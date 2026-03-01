import drjit as dr
import numpy as np
import torch
import torch.nn as nn
import tinycudann as tcnn

from encoding.hash_grid import HashGrid


class KPlanes(nn.Module):

    def __init__(self, input_dim=3, num_levels=8, level_dim=2, per_level_scale=2, base_resolution=16, log2_hashmap_size=19):
        '''
        input_dim: int, dimension of input
        num_levels: int, number of levels
        level_dim: int, feature dimension of each level
        per_level_scale: int, scale factor between levels
        base_resolution: int, resolution of the base level
        log2_hashmap_size: int, log2 of the size of the hash map
        '''

        super().__init__()

        self.input_dim = input_dim
        self.num_levels = num_levels
        self.level_dim = level_dim
        self.per_level_scale = per_level_scale
        self.base_resolution = base_resolution
        self.log2_hashmap_size = log2_hashmap_size

        self.output_dim = num_levels * level_dim * 3
        self.max_params = 2 ** log2_hashmap_size

        self.model = [
            HashGrid(
                input_dim=2,
                num_levels=num_levels,
                level_dim=level_dim,
                per_level_scale=per_level_scale,
                base_resolution=base_resolution,
                log2_hashmap_size=log2_hashmap_size
            ).cuda() for _ in range(3)
        ]

    def forward(self, inputs):
        '''
        inputs: torch.Tensor, shape [batch_size, input_dim], in range [0, 1]
        return: torch.Tensor, shape [batch_size, num_levels * level_dim * 3]
        '''

        output = torch.cat([
            self.model[0](inputs[:, [0, 1]]),
            self.model[1](inputs[:, [1, 2]]),
            self.model[2](inputs[:, [2, 0]])
        ], dim=1)

        return output