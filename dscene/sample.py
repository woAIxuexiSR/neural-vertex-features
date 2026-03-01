import mitsuba as mi
import drjit as dr
import numpy as np
import torch

mi.set_variant("cuda_rgb")

def valid_shape(shape: mi.Shape) -> bool:
    return not shape.is_emitter() and mi.has_flag(
        shape.bsdf().flags(), mi.BSDFFlags.Smooth
    )

def compute_area(scene: mi.Scene):
    m_area = []
    m_vertices = []

    params = mi.traverse(scene)
    for shape in scene.shapes():
        if valid_shape(shape):
            m_area.append(shape.surface_area())
            vnum = params[str(shape.id()) + ".vertex_positions"].numpy().shape[0] // 3
            m_vertices.append([vnum])
        else:
            m_area.append([0])
            m_vertices.append([0])
    m_area = np.array(m_area)[:, 0]
    total_area = m_area.sum()
    m_area = m_area / total_area
    m_vertices = np.array(m_vertices)[:, 0]
    m_vertices = m_vertices / m_vertices.sum()
    mask = (m_area > 0)

    return mask, total_area, m_area, m_vertices


def sample_si(
    scene: mi.Scene,
    sample_weight,
    sample1: mi.Float,
    sample2: mi.Point2f,
    sample3: mi.Point2f,
    active=True,
) -> mi.SurfaceInteraction3f:
    sampler = mi.DiscreteDistribution(sample_weight)
    shape_idx = sampler.sample(sample1, active)
    shapes: mi.Shape = dr.gather(mi.ShapePtr, scene.shapes_dr(), shape_idx, active)
    sample_pdf = sample_weight[shape_idx] / shapes.surface_area()

    ps = shapes.sample_position(0, sample2, active)
    si = mi.SurfaceInteraction3f(ps, dr.zeros(mi.Color0f))
    si.shape = shapes
    si.prim_index = ps.pidx
    si.buv = ps.buv
    si.t = mi.Float(0)

    active_two_sided = mi.has_flag(si.bsdf().flags(), mi.BSDFFlags.BackSide)
    si.wi = dr.select(
        active_two_sided,
        mi.warp.square_to_uniform_sphere(sample3),
        mi.warp.square_to_uniform_hemisphere(sample3),
    )

    return shape_idx, sample_pdf, si


# p : (N, 3), v : (N, 3, 3)
def barycentric_coords(p, v):
    
    rel = p - v[:, 0, :]
    du = v[:, 1, :] - v[:, 0, :]
    dv = v[:, 2, :] - v[:, 0, :]
    
    b1 = torch.sum(du * rel, axis=1)
    b2 = torch.sum(dv * rel, axis=1)
    a11 = torch.sum(du * du, axis=1)
    a12 = torch.sum(du * dv, axis=1)
    a22 = torch.sum(dv * dv, axis=1)
    inv_det = 1 / (a11 * a22 - a12 * a12)
    
    u = (a22 * b1 - a12 * b2) * inv_det
    v = (a11 * b2 - a12 * b1) * inv_det
    w = 1 - u - v
    
    return u, v, w