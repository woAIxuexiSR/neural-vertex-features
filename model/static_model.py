import mitsuba as mi
import drjit as dr
import torch
import torch.nn as nn
import tinycudann as tcnn
import numpy as np

from encoding.frequency import Frequency
from encoding.dense_grid import DenseGrid
from encoding.hash_grid import HashGrid, TCNNGrid
from encoding.compact_hash import CompactHash
from encoding.vertex_feature import VertexFeature
from encoding.kplanes import KPlanes

mi.set_variant("cuda_rgb")

encoding_config = {
    "otype": "Composite", 
    "reduction": "concatenation",
    "nested": [
        {
            "otype": "SphericalHarmonics",
            "n_dims_to_encode": 3,
            "degree": 4,
        },
        {
            "otype": "SphericalHarmonics",
            "n_dims_to_encode": 3,
            "degree": 4,
        },
        {
            "otype": "Identity",
            "n_dims_to_encode": 3,
        },
    ]
}

network_config = {
    "otype": "CutlassMLP",
    # "otype": "FullyFusedMLP",
    "activation": "ReLU",
    "output_activation": "None",
    "n_neurons": 128,
    "n_hidden_layers": 4,
}

class StaticModel(nn.Module):
    def __init__(self, dscene, config: dict):
        super().__init__()

        self.bb_min = dscene.scene.bbox().min
        self.bb_max = dscene.scene.bbox().max

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
        elif self.encoding_type == "compact_hash":
            self.spatial_encoding = CompactHash(
                input_dim=3,
                num_levels=config["n_levels"],
                level_dim=config["feature_dim"],
                per_level_scale=config["per_level_scale"],
                base_resolution=config["base_resolution"],
                log2_hashmap_size=config["log2_hashmap_size"],
                log2_index_code_book_size=config["log2_index_code_book_size"],
                log2_index_probing_range=config["log2_index_probing_range"]
            )
            self.encoding_dim = self.spatial_encoding.output_dim
        elif self.encoding_type == "vertex":
            self.spatial_encoding = VertexFeature(
                dscene.scene,
                config["feature_dim"]
            )
            self.encoding_dim = config["feature_dim"]
        elif self.encoding_type == "frequency":
            self.spatial_encoding = Frequency(
                input_dim=3,
                n_frequencys=config["n_frequencys"]
            )
            self.encoding_dim = self.spatial_encoding.output_dim
        elif self.encoding_type == "dense_grid":
            self.spatial_encoding = TCNNGrid(
                grid_type="DenseGrid",
                input_dim=3,
                num_levels=config["n_levels"],
                level_dim=config["feature_dim"],
                per_level_scale=config["per_level_scale"],
                base_resolution=config["base_resolution"]
            )
            self.encoding_dim = self.spatial_encoding.output_dim
        elif self.encoding_type == "kplanes":
            self.spatial_encoding = KPlanes(
                input_dim=3,
                num_levels=config["n_levels"],
                level_dim=config["feature_dim"],
                per_level_scale=config["per_level_scale"],
                base_resolution=config["base_resolution"],
                log2_hashmap_size=config["log2_hashmap_size"]
            )
            self.encoding_dim = self.spatial_encoding.output_dim

        # encoding
        self.encoding = tcnn.Encoding(3 + 3 + 3, encoding_config)
        
        # mlp
        self.n_hidden_layers = config["n_hidden_layers"]
        self.n_neurons = config["n_neurons"]

        self.n_input_dims = self.encoding_dim + self.encoding.n_output_dims
        layers = []
        for _ in range(self.n_hidden_layers):
            layers.append(nn.Linear(self.n_neurons, self.n_neurons))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(
            nn.Linear(self.n_input_dims, self.n_neurons),
            nn.ReLU(),
            *layers,
            nn.Linear(self.n_neurons, 3)
        )
        # network_config["n_neurons"] = self.n_neurons
        # network_config["n_hidden_layers"] = self.n_hidden_layers
        # self.mlp = tcnn.Network(self.n_input_dims, 3, network_config)

    def forward(self, si, vars: np.ndarray):

        with dr.suspend_grad():
            pos = si.p
            dirs = si.to_world(si.wi)
            normals = si.sh_frame.n
            # upside = (dr.dot(dirs, normals) < 0).torch().to(torch.bool)
            normals = dr.select(dr.dot(dirs, normals) < 0, -normals, normals)
            si_view = mi.SurfaceInteraction3f(si)
            si_view.wi = mi.Point3f([0.353553, 0.353553, 0.866025])
            albedo = si_view.bsdf().eval_diffuse_reflectance(si_view)

            pos = ((pos - self.bb_min) / (self.bb_max - self.bb_min)).torch()
            wi = dirs.torch()
            n = normals.torch()
            f_d = albedo.torch()

            wi[wi.isnan()] = 0.0
            n[n.isnan()] = 0.0
            
            wi = (nn.functional.normalize(wi, dim=1) + 1) * 0.5
            n = (nn.functional.normalize(n, dim=1) + 1) * 0.5

        if self.encoding_type == "hash_grid":
            x = self.spatial_encoding(pos)
        elif self.encoding_type == "compact_hash":
            x = self.spatial_encoding(pos)
        elif self.encoding_type == "vertex":
            x = self.spatial_encoding(si)
        elif self.encoding_type == "frequency":
            x = self.spatial_encoding(pos)
        elif self.encoding_type == "dense_grid":
            x = self.spatial_encoding(pos)
        elif self.encoding_type == "kplanes":
            x = self.spatial_encoding(pos)

        e = self.encoding(torch.cat([wi, n, f_d], dim=1)).to(torch.float32)
        x = torch.cat([x, e], dim=1)

        return self.mlp(x).to(torch.float32).abs() * f_d