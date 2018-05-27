import math

import numpy as np

def iris_normalization(img, pupil_circle, iris_circle, M=64, N=512, offset=0):
    """normalize the iris.
    :param img: the input img
    :param pupil_circle: (x, y, radius)
    :param iris_circle: (x, y, radius)
    :param M, N: the normalization image size
    :param offset: the initial angle
    :return: the normalization image
    :rtype: ndarray
    """

    normalized = np.zeros((M, N))
    theta = np.linspace(0, 2 * np.pi, N)

    for i in range(N):
        curr_theta = theta[i] + offset
        if curr_theta > 2 * np.pi:
            curr_theta -= 2 * np.pi
        begin = trans_axis(pupil_circle, curr_theta)
        end = trans_axis(iris_circle, curr_theta)

        xspace = np.linspace(begin[0], end[0], M)
        yspace = np.linspace(begin[1], end[1], M)
        normalized[:, i] = [255 - img[int(y), int(x)]
                            if 0 <= int(x) < img.shape[1] and 0 <= int(y) < img.shape[0]
                            else 0
                            for x, y in zip(xspace, yspace)]
    return normalized

def trans_axis(circle, theta):
    """Changes polar coordinates to cartesian coordinate system.
    :param circle: (x, y, radius)
    :param theta: angle
    :return: new coordinates (x, y)
    :rtype: tuple (int, int)
    """

    x0, y0, r = circle
    x = int(x0 + r * math.cos(theta))
    y = int(y0 + r * math.sin(theta))
    return x, y


