import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import gc
import json
import argparse

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tinycudann as tcnn
from torch_scatter import scatter

from integrators.lhs_rhs import *
from integrators.integrator import *
from dscene.dscene import DynamicScene
from dscene.sample import *
from model.helper import *

def render_img(dscene, model, path, v):
    model.eval()
    dscene.update(v)
    integrator = LHSIntegrator(model)
    integrator.v = v
    size = scene.sensors()[0].film().size()
    img: mi.TensorXf = dr.zeros(mi.TensorXf, (size[1], size[0], 3))
    
    for i in range(16):
        img += mi.render(dscene.scene, integrator=integrator, spp=1, seed=i + 2)
        dr.flush_malloc_cache()
    img = img / 16

    img = mi.Bitmap(img)
    img.write(path)

def train(dscene, model, config, out_dir, cv):
    M = config["rhs_samples"]
    batch_size = config["batch_size"]
    steps = config["steps"]
    lr = config["learning_rate"]
    save_interval = config["save_interval"]

    subdivide_start = 5000
    subdivide_interval = 10000
    subdivide_end = 30000
    rhs_update_interval = 34000
    lr_update_interval = 60000

    indices = dr.arange(mi.UInt, 0, batch_size)
    indices = dr.repeat(indices, M)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    current_lr = lr
    loss_fn = get_loss_fn(config["loss"])
    writer = SummaryWriter()
    tqdm_iter = tqdm(range(steps))

    _, total_area, _, _ = compute_area(dscene.scene)
    fcnt = model.spatial_encoding.fcnt
    # print(model.spatial_encoding.embeddings.shape, model.spatial_encoding.vcnt)

    batch_loss = 0
    subdivide_surface_loss = np.zeros(fcnt, dtype=np.float32)
    subdivide_surface_var = np.zeros(fcnt, dtype=np.float32)
    
    warm_up = 100
    v = np.random.uniform(0, 1, dscene.var_num)

    model.train()
    for step in tqdm_iter:

        if step > warm_up:
            v = np.random.uniform(0, 1, dscene.var_num)

        dscene.update(v)
        scene = dscene.scene
        model.spatial_encoding.update_data(scene, mi.traverse(scene))

        optimizer.zero_grad()

        l_sampler: mi.Sampler = mi.load_dict({"type": "independent"})
        r_sampler: mi.Sampler = mi.load_dict({"type": "independent"})
        l_sampler.seed(step, batch_size)
        r_sampler.seed(step, batch_size * M)

        # shape_idx, sample_pdf, si_lhs = sample_si(
        #     scene, sample_weight, l_sampler.next_1d(), l_sampler.next_2d(), l_sampler.next_2d()
        # )
        face_idx, sample_pdf, si_lhs = model.spatial_encoding.sample_si(
            l_sampler.next_1d(), l_sampler.next_2d(), l_sampler.next_2d()
        )
        si_rhs = dr.gather(mi.SurfaceInteraction3f, si_lhs, indices)

        _, Le_lhs, out_lhs, valid_lhs = render_lhs(scene, v, si_lhs, model)
        L_rhs, _, _, _, _ = render_rhs(scene, v, r_sampler, si_rhs, model)

        valid_lhs = valid_lhs.torch().bool()
        out_lhs[~valid_lhs] = 0.0
        lhs = Le_lhs.torch() + out_lhs

        rhs = L_rhs.torch()
        rhs = rhs.reshape(batch_size, M, 3)

        weight = (sample_pdf * total_area).reshape(-1, 1)
        loss_tensor = loss_fn(lhs, rhs, weight)
        loss = torch.mean(loss_tensor)
        batch_loss += loss.item()

        loss.backward()
        optimizer.step()

        tqdm_iter.set_description(f"loss: {loss.item():.4f}")
        torch.cuda.empty_cache()
        dr.flush_malloc_cache()
        gc.collect()

        if config.get("use_subdivide", False) and step > subdivide_start and step < subdivide_end:
            ltensor = loss_tensor.mean(dim=1).detach()
            surface_loss = scatter(
                ltensor, face_idx.to(torch.int64), dim=0, dim_size=fcnt, reduce="mean")
            surface_radiance = scatter(
                lhs.norm(dim=1).detach(), face_idx.to(torch.int64), dim=0, dim_size=fcnt, reduce="mean")
            avg_area = model.spatial_encoding.avg_area()
            surface_loss = surface_loss * avg_area * torch.log(surface_radiance + 1)

            subdivide_surface_loss += surface_loss.cpu().numpy()
            square_diff = (ltensor - surface_loss[face_idx]) ** 2
            surface_var = scatter(
                square_diff, face_idx.to(torch.int64), dim=0, dim_size=fcnt, reduce="mean")
            subdivide_surface_var += surface_var.cpu().numpy() 

            if step % subdivide_interval == subdivide_interval - 1 and model.spatial_encoding.finish_subdivide():
                print("Finish subdivide ", step)
            # update subdivide level
            elif step % subdivide_interval == subdivide_interval - 1:

                value = subdivide_surface_loss.copy()
                value_mean = value.mean()
                value_std = value.std()
                new_level = np.zeros_like(value, dtype=np.int32)
                mask = (value > value_mean + 2 * value_std)
                new_level[mask] = np.floor((value - value_mean) / value_std - 1)[mask]
                new_level = np.maximum(new_level, 0)
                model.spatial_encoding.update(torch.from_numpy(new_level).cuda())
                
                subdivide_surface_loss = np.zeros(fcnt, dtype=np.float32)
                subdivide_surface_var = np.zeros(fcnt, dtype=np.float32)

                optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)
                # step_LR = torch.optim.lr_scheduler.StepLR(optimizer, steps // 3, 0.33)

        if step % save_interval == save_interval - 1:
            writer.add_scalar("loss", batch_loss / save_interval, step)
            batch_loss = 0
            checkpoint = {
                "model": model.state_dict(),
                "k": model.spatial_encoding.subdivide_level
            }
            torch.save(checkpoint, out_dir + "/" + config["model_name"])
            name = config["model_name"].split(".")[0]
            render_img(dscene, model, out_dir + "/{}_{}.exr".format(name, step), cv)

        if config.get("use_adaptive_rhs", False) and step > subdivide_end and step % rhs_update_interval == rhs_update_interval - 1:
            M = M * 2
            batch_size = batch_size // 2
            indices = dr.arange(mi.UInt, 0, batch_size)
            indices = dr.repeat(indices, M)

        if step > subdivide_end and step % lr_update_interval == lr_update_interval - 1:
            current_lr = current_lr * 0.33
            optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic Neural Radiosity")
    parser.add_argument("-c", type=str, default="config.json")
    parser.add_argument("-m", type=str, default="")

    args = parser.parse_args()
    config = json.load(open(args.c, "r"))

    out_dir = "result/" + config["output"]
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # prepare scene

    scene = mi.load_file(config["scene"])
    dscene = DynamicScene(scene)
    dscene.load_animation(config["animation"])

    # prepare model
    model = get_model(config["model"]["type"], args.m, dscene, config["model"])

    # training parameters

    train(dscene, model, config["train"], out_dir, np.array(config["v"]))
