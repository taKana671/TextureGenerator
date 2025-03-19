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