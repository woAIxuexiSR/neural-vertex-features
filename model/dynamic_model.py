import drjit as dr
import mitsuba as mi
import torch
import torch.nn as nn
import tinycudann as tcnn
import numpy as np
import time

from encoding.hash_grid import HashGrid
from encoding.vertex_feature import VertexFeature

mi.set_variant("cuda_rgb")

grid_config = {
    "otype": "HashGrid",
    "n_dims_to_encode": 3,
    "n_levels": 8,
    "n_features_per_level": 8,
    "log2_hashmap_size": 19,
    "base_resolution": 32,
    "per_level_scale": 2.0,
    "interpolation": "Linear",
}

direction_normal_config = {
    "otype": "SphericalHarmonics",
    "n_dims_to_encode": 3,
    "degree": 4,
}

# direction_normal_config = {
#     "otype": "OneBlob",
#     "n_dims_to_encode": 3,
#     "n_bins": 8,
# }

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

vvmlp_encoding_config = {
	"otype": "Frequency",
    "n_dims_to_encode": 1,
	"n_frequencies": 8,
}

vvmlp_network_config = {
    # "otype": "CutlassMLP",
    "otype": "FullyFusedMLP",
    "activation": "ReLU",
    "output_activation": "None",
    "n_neurons": 128,
    "n_hidden_layers": 4,
}

def extract_input(si: mi.SurfaceInteraction3f):
    pos = si.p
    dirs = si.to_world(si.wi)
    normals = si.sh_frame.n
    normals = dr.select(dr.dot(dirs, normals) < 0, -normals, normals)

    si_view = mi.SurfaceInteraction3f(si)
    si_view.wi = mi.Point3f([0.353553, 0.353553, 0.866025])
    albedo = si_view.bsdf().eval_diffuse_reflectance(si_view)

    return pos, dirs, normals, albedo

class DynamicModel(nn.Module):

    def __init__(self, dscene, config: dict):
        super().__init__()
        self.bb_min = dscene.scene.bbox().min
        self.bb_max = dscene.scene.bbox().max
        self.var_num = dscene.var_num

        self.encoding_type = config["encoding"]
        self.spatial_encoding = None
        self.encoding_dim = 0

        if self.encoding_type == "hash_grid":
            self.spatial_encoding = HashGrid(
                input_dim=3,
                num_levels=config["n_levels"],
                level_dim=config["feature_dim"],
                per_level_scale=config["per_level_scale"],
                base_resolution=config["base_resolution"],
                log2_hashmap_size=config["log2_hashmap_size"]
            )
            self.encoding_dim = self.spatial_encoding.output_dim
        elif self.encoding_type == "vertex":
            self.spatial_encoding = VertexFeature(
                dscene.scene,
                config["feature_dim"]
            )
            self.encoding_dim = config["feature_dim"]

        encoding_config = {"otype": "Composite", "nested": []}

        var_config = {
            "otype": config["grid_type_2d"],
            "n_dims_to_encode": 2,
            "n_levels": config["n_levels_2d"],
            "n_features_per_level": config["n_features_per_level_2d"],
            "base_resolution": config["base_resolution_2d"],
            "per_level_scale": config["per_level_scale_2d"],
            "interpolation": "Linear",
        }

        # make an integrated encoding for all variables to define its reduce type
        var_encoding_config = {
            "otype": "Composite", 
            "reduction": "concatenation",
            "nested": []
        }
        for _ in range(self.var_num * 3):
            var_encoding_config["nested"].append(var_config)

        encoding_config["nested"].append(var_encoding_config)

        encoding_config["nested"].append(direction_normal_config)
        encoding_config["nested"].append(direction_normal_config)
        encoding_config["nested"].append(reflectance_config)

        network_config["n_neurons"] = config["n_neurons"]
        network_config["n_hidden_layers"] = config["n_hidden_layers"]

        self.n_input_dims = 3 * self.var_num * 2 + 3 + 3 + 3
        self.encoding = tcnn.Encoding(self.n_input_dims, encoding_config)

        self.vvmlp_output_dims = config.get("vvmlp_output_dims", 128)
        vvmlp_encoding_config["n_dims_to_encode"] = self.var_num
        self.vvmlp = tcnn.NetworkWithInputEncoding(self.var_num, self.vvmlp_output_dims, vvmlp_encoding_config, vvmlp_network_config)
        
        self.mlp_input_dims = self.encoding_dim + self.encoding.n_output_dims + self.vvmlp_output_dims
        self.mlp = tcnn.Network(self.mlp_input_dims, 3, network_config)
        
    def forward(self, si, vars: np.ndarray):

        with dr.suspend_grad():
            pos, dirs, normals, albedo = extract_input(si)
            
            pos = ((pos - self.bb_min) / (self.bb_max - self.bb_min)).torch()
            wi = dirs.torch()
            n = normals.torch()
            f_d = albedo.torch()
            vars = torch.from_numpy(vars)
            
            # there are some nan values due to scene.ray_intersect()
            wi[wi.isnan()] = 0.0
            n[n.isnan()] = 0.0

            wi = (nn.functional.normalize(wi, dim=1) + 1) * 0.5
            n = (nn.functional.normalize(n, dim=1) + 1) * 0.5

        if self.encoding_type == "hash_grid":
            e = self.spatial_encoding(pos)
        elif self.encoding_type == "vertex":
            e = self.spatial_encoding(si)

        var_num = self.var_num
        x = torch.zeros((pos.shape[0], 3 * var_num * 2), device="cuda")
        for i in range(var_num):
            x[:, i * 6] = pos[:, 0]
            x[:, i * 6 + 1] = vars[i]
            x[:, i * 6 + 2] = pos[:, 1]
            x[:, i * 6 + 3] = vars[i]
            x[:, i * 6 + 4] = pos[:, 2]
            x[:, i * 6 + 5] = vars[i]
        x = torch.cat([x, wi, n, f_d], dim=1).to(torch.float32)
        
        x = self.encoding(x)

        v = torch.zeros((pos.shape[0], var_num), device="cuda")
        for i in range(var_num):
            v[:, i] = vars[i]
        v = self.vvmlp(v)

        x = self.mlp(torch.cat([e, x, v], dim=1)).to(torch.float32).abs()
        
        return x * f_d