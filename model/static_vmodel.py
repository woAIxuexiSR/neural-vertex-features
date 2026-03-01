import mitsuba as mi
import drjit as dr
import torch
import torch.nn as nn
import tinycudann as tcnn
import numpy as np

from dscene.sample import *
from encoding.vertex_feature import VertexFeature

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
    "otype": "FullyFusedMLP",
    "activation": "ReLU",
    "output_activation": "None",
    "n_neurons": 128,
    "n_hidden_layers": 4,
}

class StaticVModel(nn.Module):
    def __init__(self, dscene, config: dict):
        super().__init__()

        self.feature_dim = config["feature_dim"]
        self.n_hidden_layers = config["n_hidden_layers"]
        self.n_neurons = config["n_neurons"]

        self.vertex_feature = VertexFeature(dscene.scene, self.feature_dim)

        # encoding
        self.encoding = tcnn.Encoding(3 + 3 + 3, encoding_config)
        
        # mlp
        self.n_input_dims = self.feature_dim + self.encoding.n_output_dims
        layers = []
        for i in range(self.n_hidden_layers):
            layers.append(nn.Linear(self.n_neurons, self.n_neurons))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(
            nn.Linear(self.n_input_dims, self.n_neurons),
            nn.ReLU(),
            *layers,
            nn.Linear(self.n_neurons, 3)
        )

    def forward(self, si, vars: np.ndarray):

        with dr.suspend_grad():
            dirs = si.to_world(si.wi)
            normals = si.sh_frame.n
            # upside = (dr.dot(dirs, normals) < 0).torch().to(torch.bool)
            normals = dr.select(dr.dot(dirs, normals) < 0, -normals, normals)
            si_view = mi.SurfaceInteraction3f(si)
            si_view.wi = mi.Point3f([0.353553, 0.353553, 0.866025])
            albedo = si_view.bsdf().eval_diffuse_reflectance(si_view)

            wi = dirs.torch()
            n = normals.torch()
            f_d = albedo.torch()

            wi[wi.isnan()] = 0.0
            n[n.isnan()] = 0.0
            
            wi = (nn.functional.normalize(wi, dim=1) + 1) * 0.5
            n = (nn.functional.normalize(n, dim=1) + 1) * 0.5

        x = self.vertex_feature(si)
        e = self.encoding(torch.cat([wi, n, f_d], dim=1)).to(torch.float32)
        x = torch.cat([x, e], dim=1)

        return self.mlp(x).abs() * f_d