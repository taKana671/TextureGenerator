from enum import Enum

import numpy as np


def get_gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T


def get_gradient_3d(width, height, starts, stops, is_hors):
    result = np.zeros((height, width, len(starts)), dtype=np.uint8)

    for i, (start, stop, is_horizontal) in enumerate(zip(starts, stops, is_hors)):
        result[:, :, i] = get_gradient_2d(start, stop, width, height, is_horizontal)

    return result


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