import drjit as dr
import mitsuba as mi
import torch
import torch.nn as nn
import tinycudann as tcnn
import numpy as np

grid_config = {
    "otype": "HashGrid",
    "n_dims_to_encode": 3,
    "n_levels": 8,
    "n_features_per_level": 8,
    "base_resolution": 32,
    "per_level_scale": 2.0,
    "interpolation": "Linear",
}

direction_normal_config = {
    "otype": "SphericalHarmonics",
    "n_dims_to_encode": 3,
    "degree": 4,
}

reflectance_config = {
    "otype": "Identity",
    "n_dims_to_encode": 3,
}

roughness_config = {
    "otype": "OneBlob",
    "n_dims_to_encode": 1,
    "n_bins": 8 
}

network_config = {
    "otype": "CutlassMLP",
    # "otype": "FullyFusedMLP",
    "activation": "ReLU",
    "output_activation": "None",
    "n_neurons": 128,
    "n_hidden_layers": 4,
}

class DNRField(nn.Module):

    def __init__(self, dscene, config: dict):
        super().__init__()
        self.bb_min = dscene.scene.bbox().min
        self.bb_max = dscene.scene.bbox().max
        self.encoding_reduce = config.get("encoding_reduce", "concatenation")
        assert self.encoding_reduce in ["concatenation", "sum", "product"]

        encoding_config = {"otype": "Composite", "nested": []}
        encoding_config["nested"].append(grid_config)
        encoding_config["nested"].append(direction_normal_config)
        encoding_config["nested"].append(direction_normal_config)
        encoding_config["nested"].append(reflectance_config)
        # encoding_config["nested"].append(roughness_config)

        network_config["n_neurons"] = config["n_neurons"]
        network_config["n_hidden_layers"] = config["n_hidden_layers"]

        self.n_input_dims = 3 + 3 + 3 + 3
        self.model = tcnn.NetworkWithInputEncoding(
            self.n_input_dims, 3, encoding_config, network_config
        )
        
    def forward(self, si, vars: np.ndarray):
        
        with dr.suspend_grad():
            pos = si.p
            dirs = si.to_world(si.wi)
            normals = si.sh_frame.n
            normals = dr.select(dr.dot(dirs, normals) < 0, -normals, normals)

            si_view = mi.SurfaceInteraction3f(si)
            si_view.wi = mi.Point3f([0.353553, 0.353553, 0.866025])
            albedo = si_view.bsdf().eval_diffuse_reflectance(si_view)

            pos = ((pos - self.bb_min) / (self.bb_max - self.bb_min)).torch()
            wi = dirs.torch()
            n = normals.torch()
            f_d = albedo.torch()
            # r = roughness.torch()[:, None]
            
            # there are some nan values due to scene.ray_intersect()
            wi[wi.isnan()] = 0.0
            n[n.isnan()] = 0.0
            
            wi = (nn.functional.normalize(wi, dim=1) + 1) * 0.5
            n = (nn.functional.normalize(n, dim=1) + 1) * 0.5

        x = torch.cat([pos, wi, n, f_d], dim=1)
        x = self.model(x).to(torch.float32).abs()
        return x