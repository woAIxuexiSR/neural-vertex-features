import argparse
import json

import drjit as dr
import imgui
import mitsuba as mi
import numpy as np
import torch

from dscene.dscene import DynamicScene
from integrators.integrator import LHSIntegrator, LevelIntegrator, RHSIntegrator
from model.helper import get_model
from utils.ui import UI

mi.set_variant("cuda_rgb")


def build_model(config: dict, model_path: str, dscene: DynamicScene):
    if model_path == "":
        return None
    return get_model(config["model"]["type"], model_path, dscene, config["model"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive viewer")
    parser.add_argument("-c", type=str, required=True)
    parser.add_argument("-m", type=str, default="")

    args = parser.parse_args()
    config = json.load(open(args.c, "r"))

    scene: mi.Scene = mi.load_file(config["scene"])
    dscene = DynamicScene(scene)

    if "animation" in config:
        dscene.load_animation(config["animation"])

    initial_v = config.get("v")
    if initial_v is not None and initial_v != "":
        dscene.update(np.array(initial_v, dtype=np.float32))
    else:
        dscene.update(np.ones(dscene.var_num, dtype=np.float32) * 0.5)

    if dscene.active_moving_camera:
        cam_v = config.get("cam_v")
        if cam_v is not None and cam_v != "":
            dscene.camera_v = cam_v
            dscene.update(dscene.v, changed=False)

    sensor: mi.Sensor = dscene.scene.sensors()[0]
    width, height = sensor.film().size()
    ui = UI(width, height, dscene.camera)

    model = build_model(config, args.m, dscene)
    path_integrator = mi.load_dict({"type": "path", "max_depth": 16})
    lhs_integrator = LHSIntegrator(model) if model is not None else None
    rhs_integrator = RHSIntegrator(model) if model is not None else None
    level_integrator = LevelIntegrator(model, model.spatial_encoding.subdivide_level) if model is not None else None

    integrator_names = ["Path"]
    if model is not None:
        integrator_names.extend(["LHS", "RHS", "Level"])

    integrator_index = 0
    spp = 1
    exposure = 1.0

    while not ui.should_close():
        ui.begin_frame()
        dscene.render_ui()

        if lhs_integrator is not None:
            lhs_integrator.v = dscene.v if dscene.var_num > 0 else None
        if rhs_integrator is not None:
            rhs_integrator.v = dscene.v if dscene.var_num > 0 else None

        if imgui.tree_node("Render Options", imgui.TREE_NODE_DEFAULT_OPEN):
            _, integrator_index = imgui.combo("Integrator", integrator_index, integrator_names)
            _, spp = imgui.slider_int("SPP", spp, 1, 4)
            _, exposure = imgui.slider_float("Exposure", exposure, 0.1, 5.0)
            imgui.tree_pop()

        selected_name = integrator_names[integrator_index]
        if selected_name == "Path":
            integrator = path_integrator
        elif selected_name == "LHS":
            integrator = lhs_integrator
        elif selected_name == "RHS":
            integrator = rhs_integrator
        else:
            integrator = level_integrator

        seed = int(ui.duration * 1000)
        img = mi.render(dscene.scene, integrator=integrator, spp=spp, seed=seed).torch()
        img = torch.log1p(exposure * img)
        img = img ** (1 / 2.2)

        ui.write_texture_gpu(img)
        ui.end_frame()

    ui.close()
