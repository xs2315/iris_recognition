


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors.nearest_centroid import NearestCentroid

def predict_with_nearestcentroid(train_features, test_features,
                                 train_labels, test_labels,
                                 metric='cosine'):
    """using nearest center classifer to evaluate the results.
    :train_features, test_features: the feature vectors of train set and test set
    :train_labels, test_labels: the labels of train set and test set.
    :metric: the metric to calculate distence.
    :return: CRR and center vectors of each class
    :rtype: tuple
    """
    clf = NearestCentroid(metric=metric)

    clf.fit(train_features, train_labels)

    predicted = clf.predict(test_features)

    return cal_crr(test_labels, predicted), clf.centroids_

def perform_fld(train_features, test_features,
                train_labels, n_components=100):
    """perform Fisher Linear Discriminant to get the reduced vectors.
    :train_features, test_features: the feature vectors of train set and test set
    :train_labels: we need the labels of train set to train.
    :return: the reduced vectors of train and test
    :rtype: tuple (reduced_train, reduced_test)
    """

    clf = LinearDiscriminantAnalysis(n_components=n_components)

    clf.fit(train_features, train_labels)

    reduced_train = clf.transform(train_features)

    reduced_test = clf.transform(test_features)

    return reduced_train, reduced_test


def cal_crr(labels, predicted):
    # calculate the CRR
    return (labels == predicted).sum() / len(labels)




