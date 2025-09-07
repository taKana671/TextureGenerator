import numpy as np
import random

from noise import SimplexNoise, Fractal2D
from mask.radial_gradient_generator import RadialGradientMask
from utils import output_image


class IslandHeightmap:

    def __init__(self, noise, size=256, island_center_h=None,
                 island_center_w=None, island_size=2):
        self.noise = noise
        self.size = size
        self.mask = RadialGradientMask(
            size, size, island_center_h, island_center_w, island_size
        )

    @classmethod
    def from_sfractal(cls, size=256, gain=0.5, lacunarity=2.01, octaves=4,
                      island_center_h=None, island_center_w=None, island_size=2):
        simplex = SimplexNoise()
        noise = Fractal2D(simplex.snoise2, gain, lacunarity, octaves)
        return cls(noise.fractal, size, island_center_h, island_center_w, island_size)

    def get_height(self, x, y, t):
        noise_v = self.noise(x / self.size + t, y / self.size + t)
        mask_v = self.mask.get_gradient(x, y)[0]
        v = 0 if mask_v >= noise_v else noise_v - mask_v
        return v

    def create_island_heightmap(self, t=None):
        if t is None:
            t = random.uniform(0, 1000)

        arr = np.array(
            [self.get_height(x, y, t)
                for y in range(self.size) for x in range(self.size)]
        )

        arr = arr.reshape(self.size, self.size)
        arr = np.clip(arr * 255, a_min=0, a_max=255).astype(np.uint8)
        output_image.output(arr, 'island_heightmap')
