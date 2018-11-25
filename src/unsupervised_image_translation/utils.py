import colorsys
import resource

import numpy as np
from scipy.misc import imread

_utime = 0
_stime = 0

def start_stopwatch():
    info = resource.getrusage(resource.RUSAGE_SELF)
    global _utime, _stime
    _utime = info.ru_utime
    _stime = info.ru_stime


def stop_stopwatch(label):
    info = resource.getrusage(resource.RUSAGE_SELF)
    user = info.ru_utime - _utime
    system = info.ru_stime - _stime
    print('{} user: {:.3f}, system: {:.3f}'.format(label, user, system))


def load_image(path):
    """ 
    Loads image as 8-bit pixel grayscale image, normalizes to [0,1].
    """ 
    return imread(path, mode="L") / 255


def load_image_rgb(path):
    """ 
    Loads image as RGB image, normalizes to [0,1].
    """ 
    return imread(path, mode="RGB") / 255


def rgb2yiq(image):
    """
    Converts image from RGB colorscheme to YIQ.
    """
    return np.array(
        [colorsys.rgb_to_yiq(*pixel) for pixel in image.reshape([-1, 3])]
    ).reshape(image.shape)


def yiq2rgb(image):
    """
    Converts image from YIQ colorscheme to RGB.
    """
    return np.array(
        [colorsys.yiq_to_rgb(*pixel) for pixel in image.reshape([-1, 3])]
    ).reshape(image.shape)
