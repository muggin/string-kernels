from sklearn import svm
import numpy as np
import kernels
import data_handling
import util
import random

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


def test_performance(kernel, category, n_iter=10, batch_size = 20):
    """
    :param kernel: kernel function for the classifier
    :param category: string
    :param n_iter: number of runs of the test
    """

    # Load data and make category selection
    trainData, testData = util.load_cleaned_data('../data/train_data.p', '../data/train_data.p')
    x_train = [x[0] for x in trainData if (category in x[1])]
    x_test = [x[0] for x in testData if (category in x[1])]
    
    f1s, ps, rs = [], [], []
    for i in range(n_iter):
        f1, p, r = _run_test(kernel, [random.choice(x_train)] * batch_size, [category] * batch_size, [random.choice(x_test)] * batch_size, [category] * batch_size)
        f1s.append(f1)
        ps.append(p)
        rs.append(r)

    print "F1 score: ({}, {})".format(np.mean(f1s), np.std(f1s))
    print "Precision: ({}, {})".format(np.mean(ps), np.std(ps))
    print "Recall: ({}, {})".format(np.mean(rs), np.std(rs))

test_performance(kernels.ssk(3, 0.05), 'earn', 10, 20)
