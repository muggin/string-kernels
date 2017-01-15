import kernels
import random
import data_handling as dh
import util
from sklearn import svm
import numpy as np
import pickle
from itertools import compress


def _run_test(kernel, x_train, y_train, x_test, y_test):
    """
    Test the kernel against one category.
    :return: F1, precision, recall
    """
    clf = svm.SVC(kernel='precomputed')

    print 'Training the classifier... '
    category = "earn"
    y_train_bin = []
    for y in y_train:
        if y[0] == category:
            y_train_bin.append(1.0)
        else:
            y_train_bin.append(0.0)
    y_test_bin = []
    for y in y_test:
        if y[0] == category:
            y_test_bin.append(1.0)
        else:
            y_test_bin.append(0.0)

    gram_train = kernels.compute_Gram_matrix(kernel, x_train)
    clf.fit(gram_train, y_train_bin)

    print 'Testing the classifier... '
    gram_test = kernels.compute_Gram_matrix(kernel, x_test, x_train)
    y_pred = clf.predict(gram_test)

    return util.evaluate_pred(y_test_bin, y_pred)

def _run_test_gram(gram_train, y_train, gram_test, y_test, category):
    """
    Test the kernel against one category.
    :return: F1, precision, recall
    """
    clf = svm.SVC(kernel='precomputed', random_state=np.random.randint(0, 100))

    # print 'Training the classifier... '
    clf.fit(gram_train, y_train)

    # print 'Testing the classifier... '
    y_pred = clf.predict(gram_test.transpose())
    # print y_train
    # print y_pred

    return util.evaluate_pred(y_test, y_pred, category)


def toy_test_performance(kernel, trainData, testData, category, n_iter=10, batch_size=20):
    """
    :param kernel: kernel function for the classifier
    :param category: string
    :param n_iter: number of runs of the test
    """

    # Load data and separate samples
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

        print 'Test run {}'.format(i+1)
        f1, p, r = _run_test(kernel, x_train, y_train, x_test, y_test)
        f1s.append(f1)
        ps.append(p)
        rs.append(r)

    print "F1 score: ({}, {})".format(np.mean(f1s), np.std(f1s))
    print "Precision: ({}, {})".format(np.mean(ps), np.std(ps))
    print "Recall: ({}, {})".format(np.mean(rs), np.std(rs))


def test_performance(kernel, train_data, test_data, n_iter=10):
    """
    :param kernel: kernel function for the classifier
    :param category: string
    :param n_iter: number of runs of the test
    """

    f1s, ps, rs = [], [], []
    for i in range(n_iter):

        # Create some random batches, with both positive and negative samples
        x_train, y_train = zip(*train_data)
        x_test, y_test = zip(*test_data)

        # print 'Test run {}'.format(i+1)
        f1, p, r = _run_test(kernel, x_train, y_train, x_test, y_test)
        f1s.append(f1)
        ps.append(p)
        rs.append(r)

    print "F1 score: ({}, {})".format(np.mean(f1s), np.std(f1s))
    print "Precision: ({}, {})".format(np.mean(ps), np.std(ps))
    print "Recall: ({}, {})".format(np.mean(rs), np.std(rs))


def test_performance_gram(label, train_data, test_data, gram_train, gram_test, n_iter=10):
    """
    :param kernel: kernel function for the classifier
    :param category: string
    :param n_iter: number of runs of the test
    """
    x_train, y_train = zip(*train_data)
    x_test, y_test = zip(*test_data)

    # indices_train = [y[0] == 'corn' or y[0] == 'crude' for y in y_train]
    # indices_test = [y[0] == 'corn' or y[0] == 'crude' for y in y_test]
    # # x_train = [x_train[i] for i in indices_train]
    # y_train = list(compress(y_train, indices_train))
    # # x_test = [x_test[i] for i in indices_test]
    # y_test = list(compress(y_test, indices_test))
    # gram_train = np.array(gram_train)
    # gram_test = np.array(gram_test)
    # gram_train = gram_train[np.ix_(indices_train, indices_train)]
    # gram_test = gram_test[np.ix_(indices_train, indices_test)]
    #
    # y_train = np.array([1. if y[0] == label else -1. for y in y_train])
    # y_test = np.array([1. if y[0] == label else -1. for y in y_test])

    y_train = [y[0] for y in y_train]
    y_test = [y[0] for y in y_test]

    f1s, ps, rs = [], [], []
    for i in range(n_iter):

        # print 'Test run {}'.format(i + 1)
        f1, p, r = _run_test_gram(gram_train, y_train, gram_test, y_test, label)
        # print f1, p, r
        f1s.append(f1)
        ps.append(p)
        rs.append(r)

    print "F1 score: ({}, {})".format(np.mean(f1s), np.std(f1s))
    print "Precision: ({}, {})".format(np.mean(ps), np.std(ps))
    print "Recall: ({}, {})".format(np.mean(rs), np.std(rs))
    print '{:.3f} & {:.3f} & {:.3f}'.format(np.mean(f1s), np.mean(ps), np.mean(rs))

# trainData, testData = dh.load_pickled_data('../data/train_data_small.p', '../data/test_data_small.p')
# test_performance(kernels.wk(trainData + testData), trainData, testData, 10)
# test_performance(kernels.ssk(3, 0.9), trainData, testData, 1)

k = 3
l = '05'
category = 'earn'
with open('../data/precomp_kernels/train-ssk-{}-{}.p'.format(k, l)) as fd:
    train_gram = pickle.load(fd)
with open('../data/precomp_kernels/test-ssk-{}-{}.p'.format(k, l)) as fd:
    test_gram = pickle.load(fd)
with open('../data/train_data_small.p') as fd:
    train_data = pickle.load(fd)
with open('../data/test_data_small.p') as fd:
    test_data = pickle.load(fd)

test_performance_gram(category, train_data, test_data, train_gram, test_gram)


