import mitsuba as mi
import drjit as dr
import torch
import time
import matplotlib.pyplot as plt

from integrators.lhs_rhs import *
from dscene.sample import valid_shape

mi.set_variant("cuda_rgb")


class LHSIntegrator(mi.SamplingIntegrator):

    def __init__(self, model, config: dict={}):
        super().__init__(mi.Properties())
        self.model = model
        self.v = None
        self.config = config

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True
    ) -> tuple[mi.Color3f, bool, list[float]]:

        self.model.eval()
        with torch.no_grad():

            ray = mi.Ray3f(ray)
            # si = scene.ray_intersect(ray, active)

            # si, throughput, null_face = first_smooth(scene, si, sampler)
            si, throughput, null_face, _ = first_smooth(scene, sampler, ray, active)
            dr.sync_device()
            L, _, _, valid = render_lhs(scene, self.v, si, self.model)

        torch.cuda.synchronize()
        # torch.cuda.empty_cache()

        return L * throughput, valid & ~null_face, []


mi.register_integrator("lhs", lambda props: LHSIntegrator(props))


class RHSIntegrator(mi.SamplingIntegrator):

    def __init__(self, model, config: dict={}):
        super().__init__(mi.Properties())
        self.model = model
        self.v = None
        self.config = config

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True
    ) -> tuple[mi.Color3f, bool, list[float]]:

        self.model.eval()
        with torch.no_grad():

            ray = mi.Ray3f(ray)
            # si = scene.ray_intersect(ray, active)
            
            # si, throughput, null_face = first_smooth(scene, si, sampler)
            si, throughput, null_face, _ = first_smooth(scene, sampler, ray, active)
            dr.sync_device()
            L, _, _, _, valid = render_rhs(scene, self.v, sampler, si, self.model)

        torch.cuda.synchronize()
        # torch.cuda.empty_cache()

        return L * throughput, valid & ~null_face, []


mi.register_integrator("rhs", lambda props: RHSIntegrator(props))


class PTIntegrator(mi.SamplingIntegrator):

    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.max_depth = props.get("max_depth", 16)

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True
    ) -> tuple[mi.Color3f, bool, list[float]]:

        ray = mi.Ray3f(ray)
        si = scene.ray_intersect(ray, active)
        L, valid = render_pt(scene, sampler, si)
        
        return L, valid, []


mi.register_integrator("pt", lambda props: PTIntegrator(props))


class ErrorIntegrator(mi.SamplingIntegrator):

    def __init__(self, model, error):
        super().__init__(mi.Properties())
        self.model = model.spatial_encoding
        self.error = error.cuda()
        self.cmap = plt.get_cmap("turbo")

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True
    ) -> tuple[mi.Color3f, bool, list[float]]:

        self.model.eval()
        with dr.suspend_grad():

            ray = mi.Ray3f(ray)
            si = scene.ray_intersect(ray, active)

            shape_idx = dr.reinterpret_array_v(mi.Int, dr.reinterpret_array_v(mi.UInt, si.shape)).torch()
            prim_idx = dr.reinterpret_array_v(mi.Int, si.prim_index).torch()
        
            face_idx = self.model.foffsets[shape_idx] + prim_idx
            face_idx[face_idx >= self.model.fcnt] = 0
            errors = self.error[face_idx].cpu()
            normed_errors = (errors - errors.min()) / (errors.max() - errors.min() + 1e-6)

        # colors = self.cmap(errors)[:, :3]
        colors = torch.stack([
            normed_errors,
            torch.zeros_like(errors),
            1 - normed_errors
        ], dim=1).numpy()

        result = mi.Color3f(colors)

        bsdf: mi.BSDF = si.bsdf(ray)
        Le = si.emitter(scene).eval(si)
        not_emitter = dr.eq(Le, mi.Color3f(0.0))
        valid = si.is_valid() & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth) & \
            not_emitter.x & not_emitter.y & not_emitter.z
        
        result[~valid] = mi.Color3f(0.0)

        return result, valid, []


mi.register_integrator("lhs", lambda props: ErrorIntegrator(props))


class LevelIntegrator(mi.SamplingIntegrator):

    def __init__(self, model, level):
        super().__init__(mi.Properties())
        self.model = model.spatial_encoding
        self.level = level.cuda()
        self.cmap = plt.get_cmap("tab20")

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True
    ) -> tuple[mi.Color3f, bool, list[float]]:

        self.model.eval()
        with dr.suspend_grad():

            ray = mi.Ray3f(ray)
            si = scene.ray_intersect(ray, active)

            shape_idx = dr.reinterpret_array_v(mi.Int, dr.reinterpret_array_v(mi.UInt, si.shape)).torch()
            prim_idx = dr.reinterpret_array_v(mi.Int, si.prim_index).torch()
        
            face_idx = self.model.foffsets[shape_idx] + prim_idx
            face_idx[face_idx >= self.model.fcnt] = 0
            levels = self.level[face_idx]

            # max_level = self.level.max()
            max_level = 40
            level = (levels / max_level).cpu()

        # colors = self.cmap(level)[:, :3]
        # colors = torch.stack([
        #     level,
        #     torch.zeros_like(level),
        #     torch.zeros_like(level)
        # ], dim=1).numpy()
        colors = torch.stack([
            torch.ones_like(level),              # R 分量固定为 1（红/橙基调）
            0.9 * (1 - level) + 0.1,             # G 分量：从 1 到 0.1（黄到深橙）
            0.6 * (1 - level) + 0.2              # B 分量：从 0.8 到 0.2（米黄到红）
        ], dim=1).numpy()

        result = mi.Color3f(colors)

        bsdf: mi.BSDF = si.bsdf(ray)
        Le = si.emitter(scene).eval(si)
        not_emitter = dr.eq(Le, mi.Color3f(0.0))
        valid = si.is_valid() & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth) & \
            not_emitter.x & not_emitter.y & not_emitter.z
        
        result[~valid] = mi.Color3f(0.0)

        return result, valid, []


mi.register_integrator("lhs", lambda props: LevelIntegrator(props))