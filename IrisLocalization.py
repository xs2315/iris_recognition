import cv2
import numpy as np

from skimage import transform, feature


def detect_by_hough(img):
    """preprocess the image, then use hough transform to detect both pupil and iris.
    :param img: the input image
    :return: the circles of pupil and iris, (x, y, radius), (x, y, radius)
    :rtype: tuple
    """

    # in practice, I did use the roughly_localize method, since it is error-prone.

    def preprocess(img, pupil=False):
        # respectively process pupil and iris
        if pupil:
            thresh = cv2.adaptiveThreshold(img, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 5, 3)
            blur = cv2.GaussianBlur(thresh, (9, 9), 0)
            blur = cv2.medianBlur(blur, 33)
            #
            canny = cv2.Canny(blur, 15, 50)
        else:
            blured = img.copy()
            for i in range(3):
                blured = cv2.medianBlur(blured, 11)
            canny = cv2.Canny(blured, 15, 30)
            canny[:,
            pupil_circle[0] - pupil_circle[2] - 30:pupil_circle[0] + pupil_circle[2] + 30] = 0
            canny[0:pupil_circle[1] - pupil_circle[2]] = 0

        return canny

    canny = preprocess(img, pupil=True)

    pupil_circle = customed_hough_circle(canny, hough_radii=range(30, 70, 5))

    # slightly enlarge the radius of pupil
    pupil_circle[2] += 10

    canny = preprocess(img)

    iris_circle = customed_hough_circle(canny, hough_radii=range(pupil_circle[2] + 50, 150, 5))  # circles[0][0]

    # if the distance of iris center and pupil center is too far, we fix the iris center.
    if ((iris_circle - pupil_circle)[:2] ** 2).sum() ** 0.5 > pupil_circle[-1] * 0.3:
        iris_circle[:2] = pupil_circle[:2]

    return pupil_circle, iris_circle


def get_pupil_roughly(img, binarize=False):
    """roughly localize pupil, using the method introduced in the paper.
    :param img: the input image
    :param binarize: whether binarize the img firstly
    :return: the center coordinates (x, y)
    :rtype: tuple (int, int)
    """

    if binarize:
        (_, img) = cv2.threshold(img,
                                 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    x_min = img.sum(axis=0).argmin()
    y_min = img.sum(axis=1).argmin()

    return x_min, y_min


def select_region(img, x_min, y_min, size=60):
    """select a square  region centered at (x_min, y_min).
    :param img: the input image
    :param x_min, y_min: the center coordinates
    :param size: the range
    :return: the selected image
    :rtype: ndarray
    """

    size = min(x_min, y_min, size)
    return img[y_min - size:y_min + size, x_min - size:x_min + size]


def roughly_localize(img, size=60):
    """the whole procedure of roughly localizing pupil.
    :param img: the input image
    :param size: the range
    :return: the center coordinates (x, y)
    :rtype: tuple (int, int)
    """

    x_min, y_min = get_pupil_roughly(img)

    # the coordinates should not be off center too much.
    if abs(x_min - 160) > 30:
        x_min = 160
    if abs(y_min - 140) > 30:
        y_min = 140

    # localize 2 times.
    for i in range(2):
        x, y = get_pupil_roughly(select_region(img, x_min, y_min, size=size),
                                 binarize=True)
        if abs(x_min - 160) > 30 or abs(y_min - 140) > 30:
            break
        x_min -= size - x
        y_min -= size - y

    return x_min, y_min


def customed_hough_circle(img, hough_radii=range(30, 60)):
    """find circles on given img.
    :param img: the input image
    :param hough_radii: the radii of candidate circles
    :return: the best circle, (x, y, radius)
    :rtype: tuple (int, int, int)
    """

    hough_res = transform.hough_circle(img, hough_radii)
    centers = []
    accums = []
    radii = []

    for radius, h in zip(hough_radii, hough_res):
        num_peaks = 10
        peaks = feature.peak_local_max(h, num_peaks=num_peaks)
        centers.extend(peaks)
        accums.extend(h[peaks[:, 0], peaks[:, 1]])
        radii.extend([radius] * num_peaks)
    best = np.argsort(accums)[::-1][0]
    return np.array([centers[best][1], centers[best][0], radii[best]])


