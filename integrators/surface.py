import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
import numpy as np

mi.set_variant("cuda_rgb")

class SurfaceIntegrator(mi.SamplingIntegrator):
    def __init__(self, shape_weight, shapes):
        super().__init__(mi.Properties())
        self.background = mi.Color3f(0.1)
        
        colormap = plt.cm.coolwarm
        shape_color = []
        for i in range(shape_weight.shape[0]):
            color = colormap(shape_weight[i])
            shape_color.append([color[0], color[1], color[2]])
        self.shape_color = np.array(shape_color)
        self.shapes = shapes

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True
    ) -> tuple[mi.Color3f, bool, list[float]]:

        with dr.suspend_grad():

            si: mi.SurfaceInteraction3f = scene.ray_intersect(ray)

        result = mi.Color3f(0.0)
        result[~si.is_valid()] = self.background
        
        shapes = si.shape
        for i in range(len(self.shapes)):
            result[dr.eq(shapes, self.shapes[i])] = self.shape_color[i, :]

        return result, si.is_valid(), []

# mi.register_integrator("surface", lambda props: SurfaceIntegrator(props))