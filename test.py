import mitsuba as mi
import numpy as np
from simple_integrators.ambient import *
from simple_integrators.uv import *

mi.set_variant('cuda_rgb')

# vertex_positions: np.ndarray [n, 3]
# faces: np.ndarray [m, 3]
# vertex_normals: np.ndarray [n, 3] or [0, 3]
# vertex_texcoords: np.ndarray [n, 2]
# subdivide each face to 4 triangles
def subdivide(vertex_positions, faces, vertex_normals, vertex_texcoords):
    num_vertices = vertex_positions.shape[0]
    new_vertex_positions = []
    new_faces = []
    new_vertex_normals = []
    new_vertex_texcoords = []
    for face in faces:
        v0, v1, v2 = face

        p0 = vertex_positions[v0]
        p1 = vertex_positions[v1]
        p2 = vertex_positions[v2]
        
        if vertex_normals.shape[0] > 0:
            n0 = vertex_normals[v0]
            n1 = vertex_normals[v1]
            n2 = vertex_normals[v2]
        
        uv0 = vertex_texcoords[v0]
        uv1 = vertex_texcoords[v1]
        uv2 = vertex_texcoords[v2]

        # Calculate midpoints
        p01 = (p0 + p1) / 2
        p12 = (p1 + p2) / 2
        p20 = (p2 + p0) / 2
        
        if vertex_normals.shape[0] > 0:
            n01 = (n0 + n1) / 2
            n12 = (n1 + n2) / 2
            n20 = (n2 + n0) / 2

        uv01 = (uv0 + uv1) / 2
        uv12 = (uv1 + uv2) / 2
        uv20 = (uv2 + uv0) / 2

        # Add new vertices
        new_vertex_positions.extend([p01, p12, p20])
        if vertex_normals.shape[0] > 0:
            new_vertex_normals.extend([n01, n12, n20])
        new_vertex_texcoords.extend([uv01, uv12, uv20])

        p01_idx = num_vertices + len(new_vertex_positions) - 3
        p12_idx = num_vertices + len(new_vertex_positions) - 2
        p20_idx = num_vertices + len(new_vertex_positions) - 1

        new_faces.extend([
            [v0, p01_idx, p20_idx],
            [v1, p12_idx, p01_idx],
            [v2, p20_idx, p12_idx],
            [p01_idx, p12_idx, p20_idx]
        ])
    
    new_vertex_positions = np.concatenate([vertex_positions, np.array(new_vertex_positions)])
    if vertex_normals.shape[0] > 0:
        new_vertex_normals = np.concatenate([vertex_normals, np.array(new_vertex_normals)])
    else:
        new_vertex_normals = np.zeros((0, 3))
    new_vertex_texcoords = np.concatenate([vertex_texcoords, np.array(new_vertex_texcoords)])
    new_faces = np.array(new_faces)

    # print(new_vertex_positions.shape, new_faces.shape, new_vertex_normals.shape, new_vertex_texcoords.shape)
    # print(vertex_positions.shape, faces.shape, vertex_normals.shape, vertex_texcoords.shape)
    return new_vertex_positions, new_faces, new_vertex_normals, new_vertex_texcoords


if __name__ == "__main__":

    scene = mi.load_file("./static_scenes/veach-ajar/scene.xml")
    params = mi.traverse(scene)

    integrator = mi.load_dict({"type": "uv"})
    path = mi.load_dict({"type": "path", "max_depth": 16})
    img = mi.render(scene, integrator=integrator, spp=1024)
    img = mi.Bitmap(img)
    img.write("original.exr")
    img = mi.render(scene, integrator=path, spp=1024)
    img = mi.Bitmap(img)
    img.write("original_path.exr")

    floor_v = np.array(params["Floor.vertex_positions"]).reshape(-1, 3)
    print(floor_v.shape)

    for shape in scene.shapes():

        if shape.id() != "Floor":
            continue
        if shape.id() + ".vertex_positions" not in params:
            continue

        vertex_positions = np.array(params[shape.id() + ".vertex_positions"]).reshape(-1, 3)
        faces = np.array(params[shape.id() + ".faces"]).reshape(-1, 3)
        vertex_normals = np.array(params[shape.id() + ".vertex_normals"]).reshape(-1, 3)
        vertex_texcoords = np.array(params[shape.id() + ".vertex_texcoords"]).reshape(-1, 2)

        new_vertex_positions, new_faces, new_vertex_normals, new_vertex_texcoords = subdivide(
            vertex_positions, faces, vertex_normals, vertex_texcoords
        )

        params[shape.id() + '.vertex_positions'] = mi.Float(new_vertex_positions.flatten())
        params[shape.id() + '.faces'] = mi.Int(new_faces.flatten())
        params[shape.id() + '.vertex_normals'] = mi.Float(new_vertex_normals.flatten())
        params[shape.id() + '.vertex_texcoords'] = mi.Float(new_vertex_texcoords.flatten())

        # shape = scene.shapes()[8]
        # shape.write_ply("modified_shape.ply")

        # print("Shape saved to 'modified_shape.ply'")

    params.update()

    floor_v = np.array(params["Floor.vertex_positions"]).reshape(-1, 3)
    print(floor_v.shape)

    img = mi.render(scene, integrator=integrator, spp=1024)
    img = mi.Bitmap(img)
    img.write("modified.exr")
    img = mi.render(scene, integrator=path, spp=1024)
    img = mi.Bitmap(img)
    img.write("modified_path.exr")