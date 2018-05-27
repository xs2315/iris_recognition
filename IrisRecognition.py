import os
import warnings

from PerformanceEvaluation import *


def read_imgs():
    """read all images of CASIA Iris Image Database (version 1.0)
    :return: the train set and test set
    :rtype: tuple (train, test)
    """

    base_dir = './CASIA Iris Image Database (version 1.0)'
    classes = os.listdir(base_dir)
    train, test = [], []
    for c in classes:
        tr_dir = '%s/%s/1/' % (base_dir, c)
        te_dir = '%s/%s/2/' % (base_dir, c)
        for f in os.listdir(tr_dir):
            if f[-3:] == 'bmp':
                train.append(cv2.imread(tr_dir + f, 0))
        for f in os.listdir(te_dir):
            if f[-3:] == 'bmp':
                test.append(cv2.imread(te_dir + f, 0))
    return train, test


def process_imgs(imgs, use_offset=False):
    """process the input raw images to feature vectors
    :imgs: the image set
    :use_offset: in the paper, it denotes unwrapping the iris with different angles can
    remove rotation invariance. However, it seems no effect.
    :return: the train set and test set
    :rtype: tuple (train, test)
    """

    processed = []
    for img in imgs:
        circles = detect_by_hough(img)
        circles = np.array(circles).reshape(1 ,2 ,3)

        # denoising
        (_, B) = cv2.threshold(img ,180 ,255 ,cv2.THRESH_BINARY)
        (_, C) = cv2.threshold(img ,100 ,255 ,cv2.THRESH_BINARY)
        img = img & ~B & C

        # upwarp the iris with different angles
        if use_offset:
            offsets = [-9 ,-6 ,-3 ,0 ,3 ,6 ,9]

            for offset in offsets:
                normalized = iris_normalization(img,
                                                circles[0][0], circles[0][1],
                                                offset=offset)
                enhanced = enhance_img(normalized)

                ROI = enhanced[0:48]

                filtered_1, _ = defined_gabor(ROI, frequency=0.1, sigma_x=3, sigma_y=1.5)

                filtered_2, _ = defined_gabor(ROI, frequency=0.07, sigma_x=4.5, sigma_y=1.5)

                feature_vector = get_feature_vector(filtered_1, filtered_2)

                processed.append(feature_vector)

        else:
            normalized = iris_normalization(img, circles[0][0], circles[0][1])
            enhanced = enhance_img(normalized)
            ROI = enhanced[0:48]

            filtered_1, _ = defined_gabor(ROI, frequency= 32 *np. pi /180, sigma_x=3, sigma_y=1.5)

            filtered_2, _ = defined_gabor(ROI, frequency= 32 *np. pi /180, sigma_x=4.5, sigma_y=1.5)

            feature_vector = get_feature_vector(filtered_1, filtered_2)

            processed.append(feature_vector)

        print ('processed imgs: %i/%i' % (len(processed), len(imgs)), end='\r')

    return processed


def main():
    train_imgs, test_imgs = read_imgs()

    train = process_imgs(train_imgs, use_offset=False)
    test = process_imgs(test_imgs, use_offset=False)

    train_labels = np.repeat(range(108), 3)
    test_labels = np.repeat(range(108), 4)

    eval(train, test, train_labels, test_labels)

if __name__ =='__main__':
    warnings.filterwarnings("ignore")
    main()