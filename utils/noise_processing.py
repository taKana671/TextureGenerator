import math

import numpy as np


def round_num(n, ndigits=0):
    p = 10**ndigits
    return (abs(n) * p * 2 + 1) // 2 / p * math.copysign(1, n)


def round_arr(arr, decimals=0):
    return np.sign(arr) * np.floor(np.abs(arr) * 10**decimals + 0.5) / 10**decimals


def adjust_noise_amount(img, density=0.5, sharpness=0.1):
    """Adjust the amount of the noise,
        for example, to convert the FBM noise into cloud cover.
        Args:
            arr (numpy.ndarray): noize image
            density, sharpness (float): less than 1.
    """

    img = img - img.min()
    img = img / img.max()
    img = 1 - np.e ** (-(img - density) * sharpness)
    img[img < 0] = 0

    # Scale between 0 to 255 and quantize
    img = img / img.max()
    img = round_arr(img * 255)

    # img = img.astype(np.uint8)
    # cv2.imwrite('temp.png', img)

    return img