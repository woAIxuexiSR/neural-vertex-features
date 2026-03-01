import drjit as dr
import numpy as np
import torch
import torch.nn as nn
import tinycudann as tcnn


class Frequency(nn.Module):

    def __init__(self, input_dim=3, n_frequencys=12):
        '''
        input_dim: int, dimension of input
        n_frequencys: int, number of frequency
        '''

        super().__init__()
        self.input_dim = input_dim
        self.n_frequencys = n_frequencys
        self.output_dim = 2 * n_frequencys * input_dim

    def forward(self, inputs):
        '''
        inputs: torch.Tensor, shape [batch_size, input_dim], in range [0, 1]
        return: torch.Tensor, shape [batch_size, 2 * n_frequencys * input_dim]
        '''
        
        outputs = torch.cat([torch.sin(2 * np.pi * (i + 1) * inputs) for i in range(self.n_frequencys)] +
                            [torch.cos(2 * np.pi * (i + 1) * inputs) for i in range(self.n_frequencys)], dim=1)
        return outputs