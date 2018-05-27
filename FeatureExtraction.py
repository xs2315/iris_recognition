
import numpy as np

from scipy import ndimage as ndi


from skimage.util import view_as_blocks

def get_feature_vector(filtered_1, filtered_2):
    """As the paper denotes, this method generate the feature vector
    based on the two filtered image.
    :param filtered_1: the filtered image 1
    :param filtered_2: the filtered image 2
    :return: the feature vector
    :rtype: ndarray
    """

    blocks_1 = view_as_blocks(filtered_1, block_shape=(8, 8)).reshape([-1, 64])
    blocks_2 = view_as_blocks(filtered_2, block_shape=(8, 8)).reshape([-1, 64])

    def mad(array, axis):
        return np.mean(np.abs(array - np.mean(array, axis, keepdims=True)), axis)

    m_1 = blocks_1.mean(axis=-1)
    m_2 = blocks_2.mean(axis=-1)
    mad_1 = mad(blocks_1, axis=-1)
    mad_2 = mad(blocks_2, axis=-1)

    #     feature_vector = np.concatenate([np.stack([m_1, mad_1], axis=1).reshape([-1]),
    #                                     np.stack([m_2, mad_2], axis=1).reshape([-1])])
    feature_vector = np.stack([m_1, mad_1, m_2, mad_2], axis=1).reshape([-1])

    return feature_vector

def defined_gabor_kernel(frequency, sigma_x=None, sigma_y=None,
                         n_stds=3, offset=0, theta=0):
    """
    According to the codes of skimage, I directly rewrote the function gabor_kernel.

    Parameters
    ----------
    frequency : float
        Spatial frequency of the harmonic function. Specified in pixels.
    theta : float, optional
        Orientation in radians. If 0, the harmonic is in the x-direction.
    sigma_x, sigma_y : float, optional
        Standard deviation in x- and y-directions. These directions apply to
        the kernel *before* rotation. If `theta = pi/2`, then the kernel is
        rotated 90 degrees so that `sigma_x` controls the *vertical* direction.
    n_stds : scalar, optional
        The linear size of the kernel is n_stds (3 by default) standard
        deviations
    offset : float, optional
        Phase offset of harmonic function in radians.
    Returns
    -------
    g : complex array
        Complex filter kernel.
    """

    x0 = np.ceil(max(np.abs(n_stds * sigma_x * np.cos(theta)),
                     np.abs(n_stds * sigma_y * np.sin(theta)), 1))
    y0 = np.ceil(max(np.abs(n_stds * sigma_y * np.cos(theta)),
                     np.abs(n_stds * sigma_x * np.sin(theta)), 1))
    y, x = np.mgrid[-y0:y0 + 1, -x0:x0 + 1]

    g = np.zeros(y.shape, dtype=np.complex)
    g[:] = np.exp(-0.5 * (x ** 2 / sigma_x ** 2 + y ** 2 / sigma_y ** 2))
    g /= 2 * np.pi * sigma_x * sigma_y
    g *= np.cos(2 * np.pi * frequency * ((x ** 2 + y ** 2) ** 0.5))

    return g


def defined_gabor(img, frequency, sigma_x, sigma_y):
    """
    Perform gabor filter on the image using defined kernel.

    Parameters
    ----------
    param img : the input img
    frequency : float
        Spatial frequency of the harmonic function. Specified in pixels.
    sigma_x, sigma_y : float, optional
        Standard deviation in x- and y-directions. These directions apply to
        the kernel *before* rotation. If `theta = pi/2`, then the kernel is
        rotated 90 degrees so that `sigma_x` controls the *vertical* direction.
    -------
    Returns   :
    filtered_real, filtered_imag : the filtered image. Because using the evensymmetric
    filter, the filtered_imag is zero.
    """

    g = defined_gabor_kernel(frequency, sigma_x, sigma_y)
    filtered_real = ndi.convolve(img, np.real(g), mode='wrap', cval=0)
    filtered_imag = ndi.convolve(img, np.imag(g), mode='wrap', cval=0)

    return filtered_real, filtered_imag

