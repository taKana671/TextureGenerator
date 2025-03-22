import random

import cv2
import numpy as np

from .cloud import Cloud, SkyColor
from .utils.noise_processing import adjust_noise_amount
from output_image import make_dir, output

try:
    from cynoise.simplex import SimplexNoise
    from cynoise.fBm import Fractal
except ImportError:
    from pynoise.simplex import SimplexNoise
    from pynoise.fBm import Fractal


class SphereMap:

    def __init__(self, noise_func, height, r):
        self.noise = noise_func
        self.height = height
        self.r = r

    @classmethod
    def from_sfractal(cls, height=256, r=0.1, gain=0.5, lacunarity=2.011, octaves=4):
        simplex = SimplexNoise()
        fract = Fractal(simplex.snoise3, gain, lacunarity, octaves)
        return cls(fract.fractal, height, r)

    def generate_spheremap(self):
        width = self.height * 2
        arr = np.zeros((self.height, width, 3), np.float32)

        li = random.sample(list('123456789'), 3)
        aa = int(''.join(li))
        bb = int(''.join([li[1], li[2], li[0]]))
        cc = int(''.join(li[::-1]))

        for j in range(self.height):
            for i in range(width):
                x = (i + 0.5) / self.height       # // added half a pixel to get the center of the pixel instead of the top-left
                y = (j + 0.5) / self.height
                rd_x = x * 2 * np.pi
                rd_y = y * np.pi
                y_sin = np.sin(rd_y + np.pi)
                a = self.r * np.sin(rd_x) * y_sin
                b = self.r * np.cos(rd_x) * y_sin
                c = self.r * np.cos(rd_y)

                v = self.noise(np.array([a + aa, b + bb, c + cc]) * 10)
                arr[j, i] = v

        # arr = np.array(arr).reshape(256, 512)
        arr = np.clip(arr * 255, a_min=0, a_max=255).astype(np.uint8)
        # arr = adjust_noise_amount(arr)
        output(arr, 'spehre_map')
        # cv2.imwrite('test4.png', arr)

    def rotate(self, x, y, z, roll, pitch, heading):
        roll = roll * np.pi / 180
        pitch = pitch * np.pi / 180
        heading = heading * np.pi / 180

        mat_r = np.array([
            [1, 0, 0],
            [0, np.cos(roll), np.sin(roll)],
            [0, -np.sin(roll), np.cos(roll)]
        ])

        mat_p = np.array([
            [np.cos(pitch), 0, -np.sin(pitch)],
            [0, 1, 0],
            [np.sin(pitch), 0, np.cos(pitch)]
        ])

        mat_h = np.array([
            [np.cos(heading), np.sin(heading), 0],
            [-np.sin(heading), np.cos(heading), 0],
            [0, 0, 1]
        ])

        mat = np.dot(mat_h, np.dot(mat_p, mat_r))
        xd = mat[0][0] * x + mat[0][1] * y + mat[0][2] * z
        yd = mat[1][0] * x + mat[1][1] * y + mat[1][2] * z
        zd = mat[2][0] * x + mat[2][1] * y + mat[2][2] * z

        return xd, yd, zd

    def create_coords_group(self, img_w):
        base_w = 0.5
        w = np.linspace(-base_w, base_w, img_w, endpoint=False)
        h = np.linspace(-base_w, base_w, img_w, endpoint=False)

        w = w + base_w / img_w
        h = h + base_w / img_w
        ww, hh = np.meshgrid(w, h)

        a1 = 2 * ww
        a2 = 2 * hh
        x = (1 / (1 + a1 ** 2 + a2 ** 2)) ** 0.5
        y = a1 * x
        z = a2 * x

        return x, y, z

    def convert_equirectangular(self, x, y, z):
        # Convert to latitude and longitude.
        phi = np.arcsin(z)
        theta = np.arcsin(np.clip(y / np.cos(phi), -1, 1))
        theta = np.where((x < 0) & (y < 0), -np.pi - theta, theta)
        theta = np.where((x < 0) & (y > 0), np.pi - theta, theta)

        return phi, theta

    def remap_equirectangular(self, img, phi, theta):
        h, w = img.shape[:2]
        # Normalize to image coordinates.
        phi = (phi * h / np.pi + h / 2).astype(np.float32) - 0.5
        theta = (theta * w / (2 * np.pi) + w / 2).astype(np.float32) - 0.5

        return cv2.remap(img, theta, phi, cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

    def convert_to_cubemap(self, spheremap, size):
        x, y, z = self.create_coords_group(size)

        phi, theta = self.convert_equirectangular(x, y, z)
        front = self.remap_equirectangular(spheremap, phi, theta)

        rx, ry, rz = self.rotate(x, y, z, 0, 0, -90)
        phi, theta = self.convert_equirectangular(rx, ry, rz)
        right = self.remap_equirectangular(spheremap, phi, theta)

        rx, ry, rz = self.rotate(x, y, z, 0, 0, 90)
        phi, theta = self.convert_equirectangular(rx, ry, rz)
        left = self.remap_equirectangular(spheremap, phi, theta)

        rx, ry, rz = self.rotate(x, y, z, 0, 0, 180)
        phi, theta = self.convert_equirectangular(rx, ry, rz)
        back = self.remap_equirectangular(spheremap, phi, theta)

        rx, ry, rz = self.rotate(x, y, z, 0, 90, 0)
        phi, theta = self.convert_equirectangular(rx, ry, rz)
        bottom = self.remap_equirectangular(spheremap, phi, theta)

        rx, ry, rz = self.rotate(x, y, z, 0, -90, 0)
        phi, theta = self.convert_equirectangular(rx, ry, rz)
        up = self.remap_equirectangular(spheremap, phi, theta)

        img = np.zeros((size * 3, size * 4, 3), dtype=spheremap.dtype)

        img[size * 1:size * 2, :size * 1] = left
        img[size * 1:size * 2, size * 1:size * 2] = front
        img[size * 1:size * 2, size * 2:size * 3] = right
        img[size * 1:size * 2, size * 3:] = back
        img[:size * 1, size * 1:size * 2] = up
        img[size * 2:, size * 1:size * 2] = bottom


def cubemap5():
    # noise_maker = SimplexNoise()
    simplex = SimplexNoise()
    noise = Fractal(simplex.snoise3)
    size = 256
    r = 0.05
    arr = []

    for y in range(256):
        for x in range(512):
            fx = (x + 0.5) / 512
            fy = (y + 0.5) / size
            frdx = fx * 2 * np.pi
            frdy = fy * np.pi
            fy_sin = np.sin(frdy + np.pi)
            a = r * np.sin(frdx) * fy_sin
            b = r * np.cos(frdx) * fy_sin
            c = r * np.cos(frdy)

            v = noise.fractal(np.array([a + 123, b + 132, c + 312]) * 10)
            # v = noise_maker.snoise3(np.array([a + 123, b + 132, c + 312]) * 10)
            arr.append(v)

    arr = np.array(arr).reshape(256, 512)
    arr = np.clip(arr * 255, a_min=0, a_max=255).astype(np.uint8)
    # arr = adjust_noise_amount(arr)
    output(arr, 'spehre_map')
    # cv2.imwrite('test4.png', arr)



