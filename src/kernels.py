import ssk_kernel_c
import re
import math
import random
import ngk_kernel
import multiprocessing as mp

import numpy as np

from collections import Counter
from itertools import chain

from joblib import Parallel, delayed
import multiprocessing


def compute_Gram_matrix(kernel, X, Y=None):
    symm = False
    if Y is None:
        Y = X
        symm = True
    gram = np.empty((len(X), len(Y)))
    for i in range(0, len(X)):
        print '\rcur: ', i,
        for j in range(0, len(Y)):
            if symm and j < i:  # using symetry
                continue
            gram[i, j] = kernel(X[i], Y[j])
            if symm:
                gram[j, i] = gram[i, j]
    print '\r',
    return gram


def _ssk_picklable(s, t, k, l):
    return ssk_kernel_c.ssk_kernel(s, t, k, l)


def compute_ssk_Gram_matrix(k, l, X, Y=None):
    symm = False
    if Y is None:
        Y = X
        symm = True
    gram = np.empty((len(X), len(Y)))
    for i in range(0, len(X)):
        print '\rcur i: ', i, '/', len(X)
        x = X[i]
        if symm:
            gram[i, i:] = Parallel(n_jobs=4)(delayed(_ssk_picklable)(x, y, k, l) for y in Y[i:])
            gram[i:, i] = gram[i, i:]
        else:
            gram[i, :] = Parallel(n_jobs=4)(delayed(_ssk_picklable)(x, y, k, l) for y in Y)
    print '\r',
    return gram


def parallel_similarity(args):
    s, t, i, j = args

    if i == j:
        print 'Processing {} x {}'.format(i, j)

    return i, j, ssk_kernel_c.ssk_kernel(s, t, 5, 0.01)


def compute_Gram_matrix_par(kernel, X, Y=None):
    symm = False
    if Y is None:
        Y = X
        symm = True

    gram = np.empty((len(X), len(Y)))

    if symm:
        data = [(X[i], Y[j], i, j) for i in xrange(len(X)) for j in xrange(len(Y)) if i <= j]
    else:
        data = [(X[i], Y[j], i, j) for i in xrange(len(X)) for j in xrange(len(Y))]

    workers = mp.Pool(processes=32)
    results = workers.map(parallel_similarity, data)

    for i, j, result in results:
        gram[i, j] = result
        if symm:
            gram[j, i] = result

    return gram

def parallel_multi_similarity(args):
    s, t, i, j = args

    if i == j:
        print 'Processing {} x {}'.format(i, j)

    return i, j, ssk_kernel_c.ssk_kernel_many(s, t, range(3, 15), 0.5)


def compute_multi_Gram_matrix_par(X, Y=None):
    symm = False
    if Y is None:
        Y = X
        symm = True

    gram = np.empty((len(X), len(Y), len(range(3, 15))))

    if symm:
        data = [(X[i], Y[j], i, j) for i in xrange(len(X)) for j in xrange(len(Y)) if i <= j]
    else:
        data = [(X[i], Y[j], i, j) for i in xrange(len(X)) for j in xrange(len(Y))]

    workers = mp.Pool(processes=4)
    results = workers.map(parallel_multi_similarity, data)

    for i, j, result in results:
        gram[i, j] = result
        if symm:
            gram[j, i] = result

    return gram


def ssk(k, l):
    """
    Get SSK kernel function with given parameters.

    :param k: length
    :param l: lambda
    :return: function (X, Y) -> float
    """
    return lambda x, y: ssk_kernel_c.ssk_kernel(x, y, k, l)
    # return lambda x, y: ssk_kernel.ssk_kernel(x, y, k, l)

def ngk(n):
    """
    Get n-gram kernel function with given parameters.

    :param n: length
    :return: function (X, Y) -> float
    """

    return lambda x, y: _ngk_kernel(x, y, n)


def _ngk_kernel(x, y, n):
    """

    :param x: first string
    :param y: second string
    :param n: length
    :return: n-gram distance between two strings
    """

    return ngk_kernel.ngk(x, y, n)


def wk(data):
    """
    Get word kernel function. (tf-idf)

    :param data: 2 strings
    :return: function (X, Y) -> float
    """

    N = len(data)

    wf = [Counter(re.split(' ', x[0])) for x in data]
    df = Counter()
    map(df.update, (w.keys() for w in wf))
    docFreq = dict(df.iteritems())

    def wk_kernel(x, y):
        wf_x = Counter(re.split(' ', x))
        wf_y = Counter(re.split(' ', y))

        k = 0.0

        for word, freq_x in wf_x.iteritems():
            freq_y = [y[1] for y in wf_y.iteritems() if y[0] == word]

            if len(freq_y) == 0:
                continue

            docFrequency = 1
            if word in docFreq:
                docFrequency = docFreq[word]

            k += math.log(1. + float(freq_x)) * math.log(1. + float(freq_y[0])) * math.log(float(N) / float(docFrequency)) 
        return k

    return wk_kernel

def combine_kernels(k1, k2, w1=1., w2=1.):
    """
    Linear combination of two kernels.

    :param k1: first kernel
    :param k2: second kernel
    :param w1: weight of the first kernel
    :param w2: weight of the second kernel
    """
    return lambda x, y: w1 * k1(x, y) + w2 * k2(x, y)


def get_approximate_ssk_gram_matrix(strings, data, k, l):
    """
    Returns not kernel function, but already constructed Gram matrix (to make computation faster)
    :param strings: set of substrings for approximation
    :param data: data to build Gram matrix on
    :param k: length for ssk
    :param l: lambda for ssk
    :return: Gram matrix
    """
    ssk_kernel = ssk(k, l)
    n = len(data)
    n2 = len(strings)
    subkernels = np.empty((n, n2))
    for i in range(n):
        for j in range(n2):
            subkernels[i, j] = ssk_kernel(data[i], strings[j])
    return compute_Gram_matrix(np.dot, subkernels)


if __name__ == '__main__':
    import kernels
    import numpy as np
    import cPickle as pickle
    import data_handling as dh

    train_data, test_data = dh.load_pickled_data('../data/train_data_small.p', '../data/test_data_small.p')

    x_train, _ = zip(*train_data)
    x_test, _ = zip(*test_data)

    ks = range(3, 15)
    l = 0.5
    print 'Working on Train'
    gram_train = kernels.compute_multi_Gram_matrix_par(x_train)
    # gram_train = kernels.compute_multissk_Gram_matrix(ks, l, x_train)
    with open('../data/train-ssk-n-05.p', 'wb') as fd:
        pickle.dump(gram_train, fd)

    print 'Working on Test'
    gram_test = kernels.compute_multi_Gram_matrix_par(x_train, x_test)
    # gram_test = kernels.compute_multissk_Gram_matrix(ks, l, x_train, x_test)
    with open('../data/test-ssk-n-05.p', 'wb') as fd:
        pickle.dump(gram_test, fd)

