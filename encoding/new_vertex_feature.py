import mitsuba as mi
import drjit as dr
import torch
import torch.nn as nn
import numpy as np

from dscene.sample import valid_shape
from tqdm import tqdm

mi.set_variant("cuda_rgb")

def extract_data(scene: mi.Scene, params: mi.SceneParameters):

    shape_ptrs = scene.shapes_dr()
    shape_idx = dr.reinterpret_array_v(mi.Int, dr.reinterpret_array_v(mi.UInt, shape_ptrs))
    shape_ids = [[shape_ptr.id(), shape_ptr, shape_idx[i]] for i, shape_ptr in enumerate(shape_ptrs)]

    num_shapes = len(scene.shapes())
    idx_map = torch.zeros(num_shapes + 1, dtype=torch.int32)
    for i in range(num_shapes):
        idx_map[shape_idx[i]] = i

    vertices, faces, faces_ref, normals, texcoords = [], [], [], [], []
    voffsets, foffsets = np.zeros(num_shapes + 1, dtype=np.int32), np.zeros(num_shapes + 1, dtype=np.int32)
    voffset, foffset = 0, 0
    for i, [shape_id, shape_ptr, idx] in enumerate(sorted(shape_ids, key=lambda x: x[0])):
        if valid_shape(shape_ptr):

            shape_vertices = params[str(shape_id) + ".vertex_positions"].numpy().reshape((-1, 3)).astype(np.float64)
            shape_faces = params[str(shape_id) + ".faces"].numpy().reshape((-1, 3))
            shape_faces_ref = np.ones(shape_faces.shape[0], dtype=np.int32) * idx
            shape_normals = params[str(shape_id) + ".vertex_normals"].numpy().reshape((-1, 3)).astype(np.float64)
            if shape_normals.shape[0] == 0:
                shape_normals = np.zeros((shape_vertices.shape[0], 3), dtype=np.float64)
            shape_texcoords = params[str(shape_id) + ".vertex_texcoords"].numpy().reshape((-1, 2)).astype(np.float64)
            if shape_texcoords.shape[0] == 0:
                shape_texcoords = np.zeros((shape_vertices.shape[0], 2), dtype=np.float64)
            vertices.append(shape_vertices)
            faces.append(shape_faces)
            faces_ref.append(shape_faces_ref)
            normals.append(shape_normals)
            texcoords.append(shape_texcoords)

            voffsets[idx] = voffset
            foffsets[idx] = foffset
            voffset += shape_vertices.shape[0]
            foffset += shape_faces.shape[0]

        else:
            voffsets[idx] = voffset
            foffsets[idx] = foffset

    vertices = torch.from_numpy(np.concatenate(vertices, axis=0, dtype=np.float64))
    faces = torch.from_numpy(np.concatenate(faces, axis=0, dtype=np.int32))
    faces_ref = torch.from_numpy(np.concatenate(faces_ref, axis=0, dtype=np.int32))
    normals = torch.from_numpy(np.concatenate(normals, axis=0, dtype=np.float64))
    texcoords = torch.from_numpy(np.concatenate(texcoords, axis=0, dtype=np.float64))
    voffsets = torch.tensor(voffsets, dtype=torch.int32).reshape(-1, 1)
    foffsets = torch.tensor(foffsets, dtype=torch.int32)
    
    return idx_map, vertices, faces, faces_ref, normals, texcoords, voffsets, foffsets


def compute_faces_area(vertices, voffsets, faces, faces_ref):
    vertices_idx = faces + voffsets[faces_ref]
    v0 = vertices[vertices_idx[:, 0]]
    v1 = vertices[vertices_idx[:, 1]]
    v2 = vertices[vertices_idx[:, 2]]

    e0 = v1 - v0
    e1 = v2 - v0
    area = torch.cross(e0, e1, dim=1)
    area = torch.norm(area, dim=1) / 2
    return area


def sample_triangle(
    vertices: torch.Tensor,
    normals: torch.Tensor,
    texcoords: torch.Tensor,
    sample: torch.Tensor,
):
    t = torch.sqrt(1 - sample[:, 0])
    u, v = 1.0 - t, sample[:, 1] * t
    bary_coords = torch.stack([1 - u - v, u, v], dim=1)

    ps = dr.zeros(mi.PositionSample3f, sample.shape[0])
    ps.p = mi.Point3f(torch.sum(vertices * bary_coords.unsqueeze(2), dim=1).to(torch.float32))
    ps.time = mi.Float(0)
    ps.delta = mi.Bool(False)
    ps.buv = mi.Point2f(bary_coords[:, 1:].to(torch.float32))

    ps.uv = mi.Point2f(torch.sum(texcoords * bary_coords.unsqueeze(2), dim=1).to(torch.float32))

    normal = torch.sum(normals * bary_coords.unsqueeze(2), dim=1)
    mask = torch.norm(normal, dim=1) > 0
    e0, e1 = vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]
    normal[~mask] = torch.cross(e0[~mask], e1[~mask], dim=1)
    normal[torch.norm(normal, dim=1) < 1e-5] = 1
    normal = normal / torch.norm(normal, dim=1, keepdim=True)

    ps.n = mi.Vector3f(normal.to(torch.float32))

    return ps


def vertex_index(i, j, k):
    return (2 * k.view(-1) + 5 - i) * i // 2 + j

# find index in the subdivided triangle
# uv: [N, 2], k: [N, 1]
def get_fidx(uv, k):

    p = uv * (k + 1).view(-1, 1)
    pi = p.int()
    pf = p - pi.float()

    mask = pf.sum(dim=1) > 1
    i1 = vertex_index(pi[:, 0] + 1, pi[:, 1], k)
    i2 = vertex_index(pi[:, 0], pi[:, 1] + 1, k)
    i3 = vertex_index(pi[:, 0], pi[:, 1], k)
    i1[mask] = vertex_index(pi[:, 0], pi[:, 1] + 1, k)[mask]
    i2[mask] = vertex_index(pi[:, 0] + 1, pi[:, 1], k)[mask]
    i3[mask] = vertex_index(pi[:, 0] + 1, pi[:, 1] + 1, k)[mask]

    bary_coords = pf
    bary_coords[mask] = 1 - pf[mask]
    indices = torch.stack([i3, i1, i2], dim=1).reshape(-1, 3)
    return indices, bary_coords


def generate_grid(k):
    i, j = torch.meshgrid(
        torch.arange(k + 2, device=k.device), 
        torch.arange(k + 2, device=k.device),
        indexing='ij'
    )
    mask = i + j <= k + 1

    corner_mask = ~(
        ((i == 0) & (j == 0)) |
        ((i == 0) & (j == k + 1)) |
        ((i == k + 1) & (j == 0))
    )

    final_mask = mask & corner_mask
    coords = torch.stack([i[final_mask], j[final_mask]], dim=-1)
    return coords
    

class NewVertexFeature(nn.Module):
    
    def __init__(self, scene: mi.Scene, feature_dim=8):
        
        super().__init__()
        
        self.scene: mi.Scene = scene
        self.params: mi.SceneParameters = mi.traverse(scene)
        self.feature_dim = feature_dim

        idx_map, vertices, faces, faces_ref, normals, texcoords, voffsets, foffsets = extract_data(scene, self.params)

        self.idx_map = idx_map.cuda()
        self.vertices = vertices.cuda()
        self.faces = faces.cuda()
        self.faces_ref = faces_ref.cuda()
        self.faces_area = compute_faces_area(vertices, voffsets, faces, faces_ref).cuda()
        self.normals = normals.cuda()
        self.texcoords = texcoords.cuda()
        self.voffsets = voffsets.cuda()
        self.foffsets = foffsets.cuda()

        self.vcnt = vertices.shape[0]
        self.fcnt = faces.shape[0]
        self.subdivide_level = torch.zeros(self.fcnt, dtype=torch.int32).cuda()
        self.virtual_offsets = torch.zeros(self.fcnt, dtype=torch.int32).cuda()

        w_vertex = np.ones(self.fcnt, dtype=np.float32) / self.fcnt
        self.w_area = (self.faces_area / torch.sum(self.faces_area)).cpu().numpy()
        sample_weight = (w_vertex + self.w_area) / 2
        sample_weight = np.maximum(sample_weight, 1e-8)
        sample_weight = sample_weight / sample_weight.sum()
        self.sample_weight = sample_weight

        self.n_params = int(np.ceil(self.vcnt / 32) * 32)
        self.embeddings = nn.Parameter(torch.empty(self.n_params, self.feature_dim))
        torch.nn.init.xavier_uniform_(self.embeddings)

    def avg_area(self):
        w_vertices = (self.subdivide_level + 2) * (self.subdivide_level + 3) // 2
        avg = self.faces_area / w_vertices
        return avg

    def update_data(self, scene: mi.Scene, params: mi.SceneParameters):
        
        self.scene = scene
        self.params = params
        
        idx_map, vertices, faces, faces_ref, normals, texcoords, voffsets, foffsets = extract_data(scene, params)
        
        self.vertices = vertices.cuda()
        self.normals = normals.cuda()
        self.texcoords = texcoords.cuda()
    

    def sample_si(self, sample1: mi.Float, sample2: mi.Point2f, sample3: mi.Point2f):
        sampler = mi.DiscreteDistribution(self.sample_weight)
        face_idx = sampler.sample(sample1)
        fidx = mi.Int32(face_idx).torch().cuda()
        w = torch.from_numpy(self.sample_weight).cuda()
        sample_pdf = w[fidx] / self.faces_area[fidx]

        if torch.isnan(sample_pdf).any():
            i = torch.isnan(sample_pdf)
            fidx[i] = sample_pdf[~i].argmax().int()
            sample_pdf = w[fidx] / self.faces_area[fidx]
        if torch.isnan(sample_pdf).any():
            print("check w: ", torch.isnan(w).any())
            print("check area: ", torch.isnan(self.faces_area).any())
            print("check fidx: ", torch.isnan(fidx).any())
            print("nan sample pdf")
            i = torch.isnan(sample_pdf)
            print(w[fidx][i], self.faces_area[fidx][i])
            print(w[fidx][i] / self.faces_area[fidx][i])
            print(self.faces_area.max(), self.faces_area.min())
            # sample_pdf[i] = 0.0
            print(sample_pdf[~i].max(), sample_pdf[~i].min())
            exit(0)

        shape_idx = self.faces_ref[fidx]
        vertices_idx = self.voffsets[shape_idx] + self.faces[fidx]
        ps: mi.PositionSample3f = sample_triangle(
            self.vertices[vertices_idx],
            self.normals[vertices_idx],
            self.texcoords[vertices_idx],
            sample2.torch()
        )
        prim_idx = fidx - self.foffsets[shape_idx]
        ps.pdf = mi.Float(1.0)
        ps.pidx = mi.Int(prim_idx)
        
        si: mi.SurfaceInteraction3f = mi.SurfaceInteraction3f(ps, dr.zeros(mi.Color0f))
        si.shape = dr.gather(mi.ShapePtr, self.scene.shapes_dr(), mi.Int(self.idx_map[shape_idx]))
        si.prim_index = ps.pidx
        si.buv = ps.buv
        si.t = mi.Float(0)

        active_two_sided = mi.has_flag(si.bsdf().flags(), mi.BSDFFlags.BackSide)
        si.wi = dr.select(
            active_two_sided,
            mi.warp.square_to_uniform_sphere(sample3),
            mi.warp.square_to_uniform_hemisphere(sample3),
        )
        return fidx, sample_pdf, si
    

    # shape_idx: [N], face_idx: [N], buv: [N, 2]
    def compute_feature_idx(self, shape_idx, face_idx, buv):
        vertices_idx = self.voffsets[shape_idx] + self.faces[face_idx]

        k = self.subdivide_level[face_idx]
        fidx, uv = get_fidx(buv, k)

        feature_idx = torch.zeros_like(vertices_idx, dtype=torch.int32)
        
        # real vertex
        mask0 = (fidx == 0)
        feature_idx[mask0] = vertices_idx[:, 0].unsqueeze(1).expand(-1, 3)[mask0]
        mask1 = (fidx == k.view(-1, 1) + 1)
        feature_idx[mask1] = vertices_idx[:, 2].unsqueeze(1).expand(-1, 3)[mask1]
        mask2 = (fidx == ((k + 2) * (k + 3) // 2 - 1).view(-1, 1))
        feature_idx[mask2] = vertices_idx[:, 1].unsqueeze(1).expand(-1, 3)[mask2]
        fidx[mask0 | mask1 | mask2] = -1
        
        # virtual vertex
        fidx[fidx > k.view(-1, 1) + 1] -= 1
        fidx -= 1
        feature_idx[fidx >= 0] = (self.vcnt + self.virtual_offsets[face_idx].view(-1, 1) + fidx)[fidx >= 0]

        return feature_idx, uv
    
    def finish_subdivide(self):
        v_vert = (self.subdivide_level + 2) * (self.subdivide_level + 3) // 2 - 3
        return (v_vert.sum() / self.vcnt > 0.5)

    def update(self, level):

        # print(self.subdivide_level.shape, level.shape)
        new_level = torch.clamp(self.subdivide_level + level, 0, 30)
        self.generate_feature(new_level)
        self.subdivide_level = new_level

        w_vertices = (self.subdivide_level + 2) * (self.subdivide_level + 3) // 2
        w_vertices = w_vertices / w_vertices.sum()
        sample_weight = (self.w_area + w_vertices.cpu().numpy()) / 2
        sample_weight = np.maximum(sample_weight, 1e-8)
        sample_weight = sample_weight / sample_weight.sum()
        self.sample_weight = sample_weight
        # torch.nn.init.xavier_uniform_(self.embeddings)

    def generate_feature(self, new_level):
        shape_idx = torch.zeros((0, 1), dtype=torch.int32)
        face_idx = torch.zeros((0, 1), dtype=torch.int32)
        buv = torch.zeros((0, 2), dtype=torch.float32).cuda()
        offsets = torch.zeros(self.fcnt, dtype=torch.int32)
        offset = 0
        # for i in tqdm(range(self.fcnt)):
        for i in range(self.fcnt):
        # for i in range(10):
            if new_level[i] == 0:
                offsets[i] = offset
                offset += 0
                continue

            si, fi = self.faces_ref[i], i
            coords = generate_grid(new_level[i]) / (new_level[i] + 1)
            
            shape_idx = torch.cat([shape_idx, torch.full((coords.shape[0], 1), si, dtype=torch.int32)], dim=0)
            face_idx = torch.cat([face_idx, torch.full((coords.shape[0], 1), fi, dtype=torch.int32)], dim=0)
            buv = torch.cat([buv, coords], dim=0)
            
            offsets[i] = offset
            offset += coords.shape[0]

        feature_idx, uv = self.compute_feature_idx(shape_idx.view(-1).cuda(), face_idx.view(-1).cuda(), buv)
        feature_idx[torch.any((feature_idx >= self.n_params), dim=1)] = 0
        x = self.embeddings[feature_idx]
        uv = torch.cat([1 - uv.sum(dim=1).unsqueeze(1), uv], dim=1)
        x = torch.sum(x * uv.unsqueeze(2), dim=1)

        with torch.no_grad():
            old_embeddings = self.embeddings.clone()
            n_params = int(np.ceil((self.vcnt + x.shape[0]) / 32) * 32)
            self.embeddings = nn.Parameter(torch.empty(n_params, self.feature_dim).cuda())
            self.embeddings[:self.vcnt] = old_embeddings[:self.vcnt]
            self.embeddings[self.vcnt:self.vcnt+x.shape[0]] = x
            self.n_params = n_params
            self.virtual_offsets = offsets.cuda()


    def forward(self, si):

        with dr.suspend_grad():
            shape_idx = dr.reinterpret_array_v(mi.Int, dr.reinterpret_array_v(mi.UInt, si.shape)).torch()
            prim_idx = dr.reinterpret_array_v(mi.Int, si.prim_index).torch()
            buv = si.buv.torch()
            buv[:, 0] = torch.clamp(buv[:, 0], min=0, max=1)
            buv[:, 1] = torch.clamp(buv[:, 1], min=torch.zeros_like(buv[:, 1]), max=1 - buv[:, 0])

            mask = si.is_valid().torch().to(torch.bool)
            if mask.shape[0] != shape_idx.shape[0]:
                mask = torch.ones(shape_idx.shape[0], dtype=torch.bool, device=mask.device)
            # mask = mask & (buv.sum(dim=1) <= 1)
            shape_idx[~mask] = 0

        face_idx = self.foffsets[shape_idx] + prim_idx
        mask = mask & (face_idx < self.fcnt)
        face_idx[~mask] = 0
        feature_idx, uv = self.compute_feature_idx(shape_idx, face_idx, buv)
        mask = mask & ~torch.any((feature_idx >= self.n_params), dim=1)
        feature_idx[~mask] = 0

        # vertices_idx = self.voffsets[shape_idx] + self.faces[face_idx]
        # vertices_idx[~mask | torch.any((vertices_idx >= self.vcnt), dim=1)] = 0

        # x = self.embeddings[vertices_idx]
        x = self.embeddings[feature_idx]
        uv = torch.cat([1 - uv.sum(dim=1).unsqueeze(1), uv], dim=1)
        x = torch.sum(x * uv.unsqueeze(2), dim=1)
        return x


if __name__ == "__main__":

    # scene = mi.load_file("./static_scenes/veach-ajar/scene.xml")
    # vf = NewVertexFeature(scene, 8).cuda()

    # sampler = mi.load_dict({"type": "independent"})
    # sampler.seed(0, 1000)
    # sample1 = sampler.next_1d()
    # sample2 = sampler.next_2d()
    # sample3 = sampler.next_2d()
    # face_idx, sample_pdf, si = vf.sample_si(sample1, sample2, sample3)

    # y = vf(si)
    # print(y.shape)

    k = torch.tensor([2], dtype=torch.int32)
    coords = generate_grid(k[0]) / (k[0] + 1)

    print(coords)

    k = torch.ones((coords.shape[0], 1), dtype=torch.int32) * k

    t = torch.ones((coords.shape[0], 1), dtype=torch.int32) * 1
    indices, bary_coords = get_fidx(coords, t)
    print("Indices:", indices)
    print("Barycentric Coordinates:", bary_coords)

    # real vertex
    mask0 = (indices == 0)
    # indices[mask0] = vertices_idx[:, 0].unsqueeze(1).expand(-1, 3)[mask0]
    mask1 = (indices == k.view(-1, 1) + 1)
    # indices[mask1] = vertices_idx[:, 2].unsqueeze(1).expand(-1, 3)[mask1]
    mask2 = (indices == ((k + 2) * (k + 3) // 2 - 1).view(-1, 1))
    # indices[mask2] = vertices_idx[:, 1].unsqueeze(1).expand(-1, 3)[mask2]
    indices[mask0 | mask1 | mask2] = -1
    
    # virtual vertex
    indices[indices > k.view(-1, 1) + 1] -= 1
    indices -= 1

    # k = 2
    # coords = generate_grid(k)
    # print("Grid Coordinates:", coords)