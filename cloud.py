import copy
from datetime import datetime
from enum import Enum

import cv2
import numpy as np

# from NoiseTexture.pynoise.perlin import Perlin
from cynoise.fBm import FractionalBrownianMotion as fbm
from .utils.noise_processing import round_arr, adjust_noise_amount
from .utils.color_gradient import get_gradient_3d


class SkyColor(Enum):

    SKYBLUE = ([0, 176, 240], [183, 236, 255])
    BLUE = ([1, 17, 104], [117, 169, 198])

    def __init__(self, start_color, end_color):
        self.start = start_color
        self.end = end_color

    def rgb_to_bgr(self):
        idx = [2, 1, 0]
        start_color = [self.start[i] for i in idx]
        end_color = [self.end[i] for i in idx]
        return start_color, end_color


class Cloud:
    """Create a cloud image from noise.
        Args:
            arr (numpy.ndarray): 3-dimensional array
    """

    def __init__(self, arr):
        self._arr = arr
        self.img = copy.deepcopy(arr)
        self.height, self.width = arr.shape[:2]

    def create_cloud_image(self, intensity=1, sky_color=SkyColor.SKYBLUE, bbox=None):
        """Create a cloud image from noise.
            Args:
                intensity(int): cloud color intensity; minimum = 1
                sky_color(SkyColor): background color
                bbox(list): 4 points of a rectangle in the noise image;
                            must be the format of [[ax, ay], [bx, by], [cx, cy], [dx, dy]].;
                            ax, ay, bx, by, cx, cy, dx, dy: int
        """
        self.img = adjust_noise_amount(self._arr)
        src_pts = self.get_src_pts(bbox)
        self.img = self.perspective_transform(src_pts)

        bg_img = self.create_background(*sky_color.rgb_to_bgr())
        cloud_img = self.generate_cloud(bg_img, intensity)
        cloud_img = cloud_img.astype(np.uint8)

        now = datetime.now()
        file_name = f'cloud_{now.strftime("%Y%m%d%H%M%S")}.png'
        cv2.imwrite(file_name, cloud_img)

    def get_src_pts(self, bbox):
        w = self.width - 1
        h = self.height - 1

        if bbox is None:
            pt_a = [w * 0.392, 0]
            pt_b = [w * 0.039, h]
            pt_c = [w * 0.96, h]
            pt_d = [w * 0.784, 0]
            return round_arr([pt_a, pt_b, pt_c, pt_d])

        if len(bbox) != 4:
            raise ValueError(f'pts must have 4 elements. got={len(bbox)}.')

        for pt in bbox:
            if len(pt) != 2:
                raise ValueError(f'The length of each element of pts must be 2. got={pt}.')

        return [[min(max(0, pt[0]), w), min(max(0, pt[1]), h)] for pt in bbox]

    def perspective_transform(self, src_pts):
        """Perspective transformation
            Args:
                src_pts (numpy.ndarray or list): 4 points of a rectangle in the noise image;
                            must be [[ax, ay], [bx, by], [cx, cy], [dx, dy]].;
                            ax, ay, bx, by, cx, cy, dx, dy: int

                            self.img              transformed
                           ________________     a ________________d
                          |     d_____ c   |     |                |
                          |     /    /     |     |                |
                          |    /____/      | --> |                |
                          |   a     b      |     |                |
                          |________________|    b|________________|c
        """
        src_pts = np.array(src_pts, np.float32)
        # src_pts = np.array([[100, 0], [10, 255], [245, 255], [200, 0]], np.float32)

        w, h = self.width - 1, self.height - 1
        dst_pts = np.array([[0, 0], [0, h], [w, h], [w, 0]], np.float32)
        mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
        pers_img = cv2.warpPerspective(self.img, mat, (self.width, self.height), flags=cv2.INTER_LINEAR)

        pers_img = pers_img - pers_img.min()
        pers_img = round_arr(pers_img / pers_img.max() * 255)

        return pers_img

    def create_background(self, start_color, end_color=None):
        if end_color is None:
            bg_img = np.full((self.height, self.width, 3), start_color, np.uint8)
        else:
            bg_img = get_gradient_3d(
                self.width, self.height, start_color, end_color, (False, False, False))

        return bg_img

    def generate_cloud(self, bg_img, intensity=1):
        """Create cloud image by alpha blending.
            Args:
                bg_image (numpy.ndarray): 3-dimensional
                intensity (int)
        """
        wh_img = np.full((self.height, self.width, 3), [255, 255, 255], np.uint8)

        for _ in range(intensity):
            bg_img = bg_img * (1 - self.img / 255) + wh_img * (self.img / 255)

        return bg_img

    @classmethod
    def from_file(cls, file_path):
        arr = cv2.imread(file_path)
        return cls(arr)

    @classmethod
    def from_fbm(cls, grid=8, size=256):
        maker = fbm(grid=grid, size=size)
        arr = maker.noise2()

        arr = np.clip(arr * 255, a_min=0, a_max=255).astype(np.uint8)
        arr = np.stack([arr] * 3, axis=2)
        return cls(arr)