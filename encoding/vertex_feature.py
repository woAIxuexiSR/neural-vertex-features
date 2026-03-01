import mitsuba as mi
import drjit as dr
import torch
import torch.nn as nn
import tinycudann as tcnn
import numpy as np

from dscene.sample import *

mi.set_variant("cuda_rgb")

class VertexFeature(nn.Module):
    def __init__(self, scene, feature_dim=8):
        super().__init__()

        self.feature_dim = feature_dim

        self.scene = scene
        shape_ptrs = scene.shapes_dr()
        shape_idx = dr.reinterpret_array_v(mi.Int, dr.reinterpret_array_v(mi.UInt, shape_ptrs))
        shape_ids = [[shape_ptr.id(), shape_ptr, shape_idx[i]] for i, shape_ptr in enumerate(shape_ptrs)]

        params = mi.traverse(scene)
        num_shapes = len(scene.shapes())
        vertices_offsets, faces_offsets = np.zeros(num_shapes + 1, dtype=np.int32), np.zeros(num_shapes + 1, dtype=np.int32)
        vertices_offset, faces_offset = 0, 0
        vertices, faces = [], []
        for i, [shape_id, shape_ptr, idx] in enumerate(sorted(shape_ids, key=lambda x: x[0])):
            if valid_shape(shape_ptr):
                shape_vertices = params[str(shape_id) + ".vertex_positions"].numpy().reshape((-1, 3)).astype(np.float64)
                shape_faces = params[str(shape_id) + ".faces"].numpy().reshape((-1, 3))
                vertices_offsets[idx] = vertices_offset
                faces_offsets[idx] = faces_offset
                vertices.append(shape_vertices)
                faces.append(shape_faces)
                vertices_offset += shape_vertices.shape[0]
                faces_offset += shape_faces.shape[0]
            else:
                vertices_offsets[idx] = vertices_offset
                faces_offsets[idx] = faces_offset
        
        vertices_offsets = torch.tensor(vertices_offsets, dtype=torch.int32).reshape(-1, 1)
        self.vertices_offsets = vertices_offsets.cuda()
        self.vertices_cnt = vertices_offset
        print(self.vertices_cnt)
        faces_offsets = torch.tensor(faces_offsets, dtype=torch.int32)
        self.faces_offsets = faces_offsets.cuda()
        self.faces_cnt = faces_offset

        self.n_params = int(np.ceil(vertices_offset / 32) * 32)
        self.embeddings = nn.Parameter(torch.empty(self.n_params, self.feature_dim))
        torch.nn.init.xavier_uniform_(self.embeddings)

        vertices = torch.from_numpy(np.concatenate(vertices, axis=0, dtype=np.float64))
        self.vertices = vertices.cuda()
        faces = torch.from_numpy(np.concatenate(faces, axis=0, dtype=np.int32))
        self.faces = faces.cuda()

    def forward(self, si):

        with dr.suspend_grad():
            shape_idx = dr.reinterpret_array_v(mi.Int, dr.reinterpret_array_v(mi.UInt, si.shape)).torch()
            prim_idx = dr.reinterpret_array_v(mi.Int, si.prim_index).torch()
            buv = si.buv.torch()

            mask = si.is_valid().torch().to(torch.bool)
            if mask.shape[0] != shape_idx.shape[0]:
                mask = torch.ones(shape_idx.shape[0], dtype=torch.bool, device=mask.device)
            shape_idx[~mask] = 0

        face_idx = self.faces_offsets[shape_idx] + prim_idx
        face_idx[~mask | (face_idx >= self.faces_cnt)] = 0
        vertices_idx = self.vertices_offsets[shape_idx] + self.faces[face_idx]
        vertices_idx[~mask | torch.any((vertices_idx >= self.vertices_cnt), dim=1)] = 0

        x = self.embeddings[vertices_idx]
        buv = torch.cat([1 - buv.sum(dim=1).unsqueeze(1), buv], dim=1)
        x = torch.sum(x * buv.unsqueeze(2), dim=1)

        return x