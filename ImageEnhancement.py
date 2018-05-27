
import cv2
import numpy as np

from skimage.filters import rank
import skimage.morphology as morp



def enhance_img(img):
    """actually, the enhance method is based on another Ma Li's paper.
        'Iris Recognition Based on Multichannel Gabor Filtering'
    :param img: the input img
    :return: the enhanced image
    :rtype: ndarray
    """
    kernel = morp.disk(32)
    img_local = rank.equalize(img.astype(np.uint8), selem=kernel)

    enhanced = cv2.GaussianBlur(img_local, (5, 5), 0)
    return enhanced