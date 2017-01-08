from sklearn import svm
import numpy as np
import kernels


def _run_test(kernel, x_train, y_train, x_test, y_test):
    """
    Test the kernel against one category.
    :return: F1, precision, recall
    """
    clf = svm.SVC(kernel=kernel)

    print 'Training the classifier... '
    clf.fit(x_train, y_train)

    print 'Testing the classifier... '
    y_pred = clf.predict(x_test)

    # todo: obtain and return F1, precision and recall
    return 0., 0., 0.


def test_performance(kernel, category, n_iter=10):
    """
    :param kernel: kernel function for the classifier
    :param category: string
    :param n_iter: number of runs of the test
    """

    f1s, ps, rs = [], [], []
    for i in range(n_iter):
        # todo: get train/test data with given category
        x_train, y_train, x_test, y_test = None, None, None, None
        f1, p, r = _run_test(kernel, x_train, y_train, x_test, y_test)
        f1s.append(f1)
        ps.append(p)
        rs.append(r)

    print "F1 score: ({}, {})".format(np.mean(f1s), np.std(f1s))
    print "Precision: ({}, {})".format(np.mean(ps), np.std(ps))
    print "Recall: ({}, {})".format(np.mean(rs), np.std(rs))


test_performance(kernels.ssk(3, 0.05), 'earn')
