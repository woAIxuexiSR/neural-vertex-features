import mitsuba as mi
import drjit as dr
import numpy as np
import json
import argparse

from dscene.dscene import DynamicScene
from integrators.integrator import *
from model.helper import *

mi.set_variant("cuda_rgb")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Static Neural Radiosity")
    parser.add_argument("-t", type=str, default="path")
    parser.add_argument("-s", type=int, default=1)
    parser.add_argument("-c", type=str, default="config.json")
    parser.add_argument("-m", type=str, default="")
    parser.add_argument("-o", type=str, default="hello.exr")

    args = parser.parse_args()
    config = json.load(open(args.c, "r"))

    # prepare scene

    scene: mi.Scene = mi.load_file(config["scene"])
    dscene = DynamicScene(scene)

    # rendering

    integrator = mi.load_dict({"type": "path", "max_depth": 16})
    if args.t == "path":
        pass
    elif args.t == "LHS" or args.t == "RHS" or args.t == "level" or args.t == "error":
        
        # prepare model
        
        model = get_model(config["model"]["type"], args.m, dscene, config["model"])
        
        # levels = torch.randint(0, 15, (model.spatial_encoding.fcnt,)).cuda()
        # model.spatial_encoding.subdivide_level = levels

        if args.t == "LHS":
            integrator = LHSIntegrator(model)
        elif args.t == "RHS":
            integrator = RHSIntegrator(model)
        elif args.t == "level":
            integrator = LevelIntegrator(model, model.spatial_encoding.subdivide_level)
        elif args.t == "error":
            errors = torch.randn(model.spatial_encoding.fcnt).abs() * 0.05 + 0.01 
            integrator = ErrorIntegrator(model, errors)
            
    else:
        raise NotImplementedError
    
    size = dscene.scene.sensors()[0].film().size()
    img: mi.TensorXf = dr.zeros(mi.TensorXf, (size[1], size[0], 3))
    
    max_spp_per_iter = 1024 if args.t == "path" else 1
    iter_num = (args.s + max_spp_per_iter - 1) // max_spp_per_iter
    spp = args.s if iter_num == 1 else max_spp_per_iter
    
    for i in range(iter_num):
        img += mi.render(dscene.scene, integrator=integrator, spp=spp, seed=i + 2)
        dr.flush_malloc_cache()
    img = img / iter_num
    
    img = mi.Bitmap(img)
    img.write(args.o)
