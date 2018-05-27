# iris_recognition
Data can be found from http://biometrics.idealtest.org/dbDetailForUser.do?id=1

This procedure is based on Python 3.6.

To run the procedure, please first install the requirements: pip install -r requirements.txt

Then, type: python IrisRecognition.py to run codes.
The result will be printed and save in directory.

The process:

	1. read all images of train and test
	
	2. for each image in image set:
			1) (optional) roughly detect pupil
			2) using hough transform to detect pupil and iris
			3) normalization and enhancement
			4) filter and extract
			5) add to train feature set or test feature set
	3. evaluation:
		1) evaluate recognition results:
			1) calculate the CRR with different metrics 'l1', 'l2' and 'cosine'
			2) re-calculate the CRR with different dimensionalities (20, 40, 60, 80, 100)
			3) save csv file (table3)
			4) draw fig and save png file (fig10)
			
		2) evaluate identification results:
			1) generating 5000 sampling set according to the paper
			2) calculate the FMR, FNMR for each threshold for each time.
			3) calculate the mean and internal of each threshold
			4) save csv & draw fig & save fig.
	

The limitations:

	1. This paper did not process the noises such like eyelid and eyelash, 
	which really limit the performance. To remove noises such like eyelid and eyelash, 
	we may apply canny edge detection and use Hough algorithm.
	
	2. This paper only use two channel of gabor filter, which can be thought as a small value. 
	So, adding more kernals is a good idea.


Function and variable explanations:

get_pupil_roughly(img, binarize=False):

    """roughly localize pupil, using the method introduced in the paper.
    :param img: the input image
    :param binarize: whether binarize the img firstly
    :return: the center coordinates (x, y)
    :rtype: tuple (int, int)
    """

detect_by_hough(img):

    """preprocess the image, then use hough transform to detect both pupil and iris.
    :param img: the input image
    :return: the circles of pupil and iris, (x, y, radius), (x, y, radius)
    :rtype: tuple
    """

    # in practice, I did use the roughly_localize method, since it is error-prone.
	
iris_normalization(img, pupil_circle, iris_circle, M=64, N=512, offset=0):

    """normalize the iris.
    :param img: the input img
    :param pupil_circle: (x, y, radius)
    :param iris_circle: (x, y, radius)
    :param M, N: the normalization image size
    :param offset: the initial angle
    :return: the normalization image
    :rtype: ndarray
    """
	
trans_axis(circle, theta):

    """Changes polar coordinates to cartesian coordinate system.
    :param circle: (x, y, radius)
    :param theta: angle
    :return: new coordinates (x, y)
    :rtype: tuple (int, int)
    """

enhance_img(img):

    """actually, the enhance method is based on another Ma Li's paper.
        'Iris Recognition Based on Multichannel Gabor Filtering'
    :param img: the input img
    :return: the enhanced image
    :rtype: ndarray
    """	

defined_gabor_kernel(frequency, sigma_x=None, sigma_y=None,
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
	
get_feature_vector(filtered_1, filtered_2):

    """As the paper denotes, this method generate the feature vector
    based on the two filtered image.
    :param filtered_1: the filtered image 1
    :param filtered_2: the filtered image 2
    :return: the feature vector
    :rtype: ndarray
    """
	
predict_with_nearestcentroid(train_features, test_features,
                                 train_labels, test_labels,
                                 metric='cosine'):
				 
    """using nearest center classifer to evaluate the results.
    :train_features, test_features: the feature vectors of train set and test set
    :train_labels, test_labels: the labels of train set and test set.
    :metric: the metric to calculate distence.
    :return: CRR and center vectors of each class
    :rtype: tuple
    """
	
perform_fld(train_features, test_features,
                train_labels, n_components=100):
		
    """perform Fisher Linear Discriminant to get the reduced vectors.
    :train_features, test_features: the feature vectors of train set and test set
    :train_labels: we need the labels of train set to train.
    :return: the reduced vectors of train and test
    :rtype: tuple (reduced_train, reduced_test)
    """
			
