from sklearn import svm
import numpy as np
import kernels
import util
import random

def _run_test(kernel, x_train, y_train, x_test, y_test):
    """
    Test the kernel against one category.
    :return: F1, precision, recall
    """
    clf = svm.SVC(kernel='precomputed')

    print 'Training the classifier... '
    gram_train = kernels.compute_Gram_matrix(kernel, x_train)
    clf.fit(gram_train, y_train)

    print 'Testing the classifier... '
    gram_test = kernels.compute_Gram_matrix(kernel, x_test)
    y_pred = clf.predict(gram_test)

    # todo: obtain and return F1, precision and recall
    return 0., 0., 0.


def test_performance(kernel, category, n_iter=10, batch_size=20):
    """
    :param kernel: kernel function for the classifier
    :param category: string
    :param n_iter: number of runs of the test
    """

    # Load data and separate samples
    trainData, testData = util.load_cleaned_data('../data/train_data.p', '../data/train_data.p')
    x_train_positive = [x[0] for x in trainData if (category in x[1])]
    x_train_negative = [x[0] for x in trainData if (category not in x[1])]
    x_test_positive = [x[0] for x in testData if (category in x[1])]
    x_test_negative = [x[0] for x in testData if (category not in x[1])]
    
    f1s, ps, rs = [], [], []
    for i in range(n_iter):

        # Create some random batches, with both positive and negative samples
        x_train = [random.choice(x_train_positive)] * int(0.5 * batch_size) + [random.choice(x_train_negative)] * (batch_size - int(0.5 * batch_size))
        y_train = [1.0] * int(0.5 * batch_size) + [-1.0] * (batch_size - int(0.5 * batch_size))
        x_test = [random.choice(x_test_positive)] * int(0.5 * batch_size) + [random.choice(x_test_negative)] * (batch_size - int(0.5 * batch_size))
        y_test = [1.0] * int(0.5 * batch_size) + [-1.0] * (batch_size - int(0.5 * batch_size))

        f1, p, r = _run_test(kernel, x_train, y_train, x_test, y_test)
        f1s.append(f1)
        ps.append(p)
        rs.append(r)

    print "F1 score: ({}, {})".format(np.mean(f1s), np.std(f1s))
    print "Precision: ({}, {})".format(np.mean(ps), np.std(ps))
    print "Recall: ({}, {})".format(np.mean(rs), np.std(rs))

test_performance(kernels.ssk(3, 0.05), 'earn', 3, 10)
