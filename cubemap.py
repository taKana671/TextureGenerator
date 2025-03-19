import pathlib
import random
from datetime import datetime

import numpy as np
import cv2

from pynoise.simplex import Simplex
from pynoise.fractal import Fractal
from .cloud import Cloud, SkyColor
from .utils.noise_processing import adjust_noise_amount


class CubeMap:
    """Create a seamless cubemap noise texture.
        Args:
            noise_func (callable): a function or method to generate noise.
            size (int): cube size.
    """

    def __init__(self, noise_func, size):
        self.noise = noise_func
        self.size = size
        self.half_size = int(size / 2)
        self.width = size * 6
        self.height = size

    def generate_cubemap(self):
        """Generate cubemap like below.

           front   right  back   left   top    bottom
            _________________________________________
           |      |      |      |      |      |      |
           |______|______|______|______|______|______|
        """
        arr = np.zeros((self.height, self.width, 3), np.float32)

        li = random.sample(list('123456789'), 3)
        aa = int(''.join(li))
        bb = int(''.join([li[1], li[2], li[0]]))
        cc = int(''.join(li[::-1]))

        for i in range(self.size * self.size):
            x = i % self.size                           # x position in image
            y = i // self.size                          # y position in image

            a = -self.half_size + x + 0.5               # x position in cube plane; addded 0.5 is to get the center of pixel
            b = -self.half_size + y + 0.5               # y position in cube plane;
            c = -self.half_size                         # z position in cube plane;

            dist_ab = (a ** 2 + b ** 2) ** 0.5
            dist_abc = (dist_ab ** 2 + c ** 2) ** 0.5
            rad = dist_abc * 0.5                        # adjust the distance a bit to get a better radius in the noise field

            a /= rad                                    # normalize the vector
            b /= rad
            c /= rad

            noise_pos = [
                [a, b, c],    # front
                [-c, b, a],   # right
                [-a, b, -c],  # back
                [c, b, -a],   # left
                [a, c, -b],   # top
                [a, -c, b]    # bottom
            ]

            for j in range(6):
                p = np.array([
                    aa + noise_pos[j][0],
                    bb + noise_pos[j][1],
                    cc + noise_pos[j][2]
                ])

                v = self.noise(p)
                arr[y, x + j * self.size] = v

        arr = np.clip(arr * 255, a_min=0, a_max=255).astype(np.uint8)
        return arr

    @classmethod
    def from_sfractal(cls, size=256, weight=0.5, lacunarity=2.011, octaves=4):
        simplex = Simplex()
        fract = Fractal(simplex.snoise3, weight, lacunarity, octaves)
        return cls(fract.fractal, size)

    @classmethod
    def from_snoise(cls, size=256):
        simplex = Simplex()
        return cls(simplex.snoise3, size)

    def create_skybox_images(self, intensity=1, sky_color=SkyColor.SKYBLUE):
        """Create skybox images.
            Args:
                intensity(int): cloud color intensity; minimum = 1
                sky_color(SkyColor): background color
        """
        img = self.generate_cubemap()
        img = adjust_noise_amount(img)
        self.generate_images(img, sky_color, intensity)

    def generate_images(self, img, sky_color, intensity=1):
        now = datetime.now()
        path = pathlib.Path(f'cubemap_{now.strftime("%Y%m%d%H%M%S")}')
        path.mkdir()

        s_color, e_color = sky_color.rgb_to_bgr()

        for i in range(6):
            start = self.size * i
            end = start + self.size
            cloud = Cloud(img[:, start: end, :])

            match i:
                case 0:
                    file_name = 'img_2.png'  # 'front.png'
                    bg_img = cloud.create_background(s_color, e_color)
                    cloud_img = cloud.generate_cloud(bg_img)

                case 1:
                    file_name = 'img_0.png'  # 'right.png'
                    bg_img = cloud.create_background(s_color, e_color)
                    cloud_img = cloud.generate_cloud(bg_img)
                    cloud_img = np.rot90(cloud_img)

                case 2:
                    file_name = 'img_3.png'  # 'back.png'
                    bg_img = cloud.create_background(s_color, e_color)
                    cloud_img = cloud.generate_cloud(bg_img)
                    cloud_img = np.rot90(cloud_img, 2)

                case 3:
                    file_name = 'img_1.png'  # 'left.png'
                    bg_img = cloud.create_background(s_color, e_color)
                    cloud_img = cloud.generate_cloud(bg_img)
                    cloud_img = np.rot90(cloud_img, 3)

                case 4:
                    file_name = 'img_4.png'  # 'top.png'
                    bg_img = cloud.create_background(s_color)
                    cloud_img = cloud.generate_cloud(bg_img)

                case 5:
                    file_name = 'img_5.png'  # 'bottom.png'
                    bg_img = cloud.create_background(e_color)
                    cloud_img = cloud.generate_cloud(bg_img)
                    cloud_img = np.rot90(cloud_img, 2)

            cloud_img = cloud_img.astype(np.uint8)
            cv2.imwrite(path / file_name, cloud_img)
