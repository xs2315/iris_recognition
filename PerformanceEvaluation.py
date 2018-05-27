import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances

from IrisLocalization import *
from IrisNormalization import *
from ImageEnhancement import *
from FeatureExtraction import *
from IrisMatching import *

def eval(train, test, train_labels, test_labels):
    # the evaluation procedure
    # read->process(detect,normalize,enhance...extract)->eval_recog->eval_ident

    # the results of table 3,4 and fig 10,13 are saved in current dir.


    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), dpi=150)

    reduced_train, reduced_test = eval_recog(train, test, train_labels, test_labels, ax=axes[0])

    _, centroids = predict_with_nearestcentroid(reduced_train, reduced_test,
                                                train_labels, test_labels,
                                                metric='cosine')

    eval_ident(reduced_train, reduced_test, train_labels, test_labels, ax=axes[1], times=5000)

    plt.savefig('figure_10_and_13.png')
    plt.show()




def eval_recog(train_features, test_features, train_labels, test_labels, ax):
    """the procedure to get eval recogination results.
    :train_features, test_features: the feature vectors of train set and test set
    :train_labels, test_labels: the labels of train set and test set.
    :return: the reduced vectors of train and test
    :rtype: tuple
    """

    o_l1, _ = predict_with_nearestcentroid(train_features, test_features,
                                           train_labels, test_labels,
                                           metric='l1')
    o_l2, _ = predict_with_nearestcentroid(train_features, test_features,
                                           train_labels, test_labels,
                                           metric='l2')
    o_cos, _ = predict_with_nearestcentroid(train_features, test_features,
                                            train_labels, test_labels,
                                            metric='cosine')

    reduced_train, reduced_test = perform_fld(train_features, test_features,
                                              train_labels, n_components=100)

    r_l1, _ = predict_with_nearestcentroid(reduced_train, reduced_test,
                                           train_labels, test_labels,
                                           metric='l1')
    r_l2, _ = predict_with_nearestcentroid(reduced_train, reduced_test,
                                           train_labels, test_labels,
                                           metric='l2')
    r_cos, _ = predict_with_nearestcentroid(reduced_train, reduced_test,
                                            train_labels, test_labels,
                                            metric='cosine')

    crrs = []
    dims = range(20, 101, 20)
    for d in dims:
        r_train, r_test = perform_fld(train_features, test_features,
                                      train_labels, n_components=d)
        crr, _ = predict_with_nearestcentroid(r_train, r_test,
                                              train_labels, test_labels,
                                              metric='cosine')

        crrs.append(crr)

    table3 = pd.DataFrame([['L1', o_l1, r_l1],
                           ['L2', o_l2, r_l2],
                           ['Cosine', o_cos, r_cos], ],
                          columns=['measure', 'original', 'reduced'])

    print(table3)
    table3.to_csv('table_3_recognition_result.csv', index=False, sep='\t')
    fig10 = pd.Series(crrs, index=dims)
    fig10 *= 100
    fig10.plot(marker='*', ax=ax)

    ax.set_xlabel('Dimensionality')
    ax.set_ylabel('CRR (%)')
    ax.set_title('Figure 10. Recognition results.')

    return reduced_train, reduced_test

def eval_ident(reduced_train, reduced_test, train_labels, test_labels, ax, times=5000):
    """the procedure to get eval recogination results.
    :reduced_train, reduced_test: the reduced vectors of train and test
    :train_labels, test_labels: the labels of train set and test set.
    :times: times for bootstrap sampling
    :return: None
    """

    _, centroids = predict_with_nearestcentroid(reduced_train, reduced_test,
                                                train_labels, test_labels,
                                                metric='cosine')

    scores = pairwise_distances(reduced_test, centroids, metric='cosine')

    labels = np.zeros_like(scores, dtype=np.bool)
    labels[range(len(labels)), test_labels] = True

    scores = scores.reshape((-1, 4, 108))
    labels = labels.reshape((-1, 4, 108))

    cls = np.random.choice(108, size=108 * times, replace=True)
    samples = np.random.randint(4, size=108 * times)

    sampled_scores = scores[cls, samples].reshape((times, -1, 108))
    sampled_labels = labels[cls, samples].reshape((times, -1, 108))

    thresh = 0.45 + 0.01 * np.arange(30, step=2)
    fmr, fnmr = [], []
    results = []
    for i in range(times):
        r = [cal_roc(sampled_labels[i], sampled_scores[i], thresh=t) for t in thresh]
        results.append(r)
    results = np.stack(results)

    fmrs, fnmrs = results[:, :, 0].T, results[:, :, 1].T

    m_fmr = fmrs.mean(axis=-1) * 100
    m_fnmr = fnmrs.mean(axis=-1) * 100

    conf = 95
    l = (100 - conf) / 2
    h = 100 - l
    int_fmr = np.percentile(fmrs, [l, h], axis=-1).T * 100
    int_fnmr = np.percentile(fnmrs, [l, h], axis=-1).T * 100

    table4 = pd.DataFrame({ 'THRESH': thresh,
                            'FMR': m_fmr, 'FMR_Interval': [tuple(i) for i in int_fmr],
                           'FNMR': m_fnmr, 'FNMR_Interval': [tuple(i) for i in int_fnmr]})

    print(table4)
    table4.to_csv('table_4_FMR_FNMR.csv', index=False, sep='\t')

    # fig, ax = plt.subplots()
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1e-0, 1e1, 1e2])
    ax.set_xscale("log")

    ax.set_xlabel('FMR (%)')
    ax.set_ylabel('FNMR (%)')
    ax.plot(m_fmr, m_fnmr, linestyle='--')
    ax.plot(int_fmr, m_fnmr)

    ax.set_title('Figure 13. ROC.')

def cal_roc(labels, scores, thresh=0.1):
    ta = ((labels == True) & (scores <= thresh)).sum()
    fa = ((labels == False) & (scores <= thresh)).sum()
    fr = ((labels == True) & (scores > thresh)).sum()
    tr = ((labels == False) & (scores > thresh)).sum()

    #     print (ta+fa+fr+tr)
    fmr = fa / (fa + tr)
    fnmr = fr / (fr + ta)

    return fmr, fnmr








