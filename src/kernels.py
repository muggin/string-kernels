import ssk_kernel as ssk
import numpy as np
import random
from collections import Counter
from itertools import chain
import re
import math

def compute_Gram_matrix(kernel, X):
    gram = np.empty((len(X), len(X)))
    for i in range(0, len(X)):
        for j in range(0, len(X)):
            if j < i: # using symetry
                continue
            gram[i, j] = kernel(X[i], X[j])
            gram[j, i] = gram[i, j]
    return gram

def ssk(k, l):
    """
    Get SSK kernel function with given parameters.

    :param k: length
    :param l: lambda
    :return: function (X, Y) -> float
    """
    return lambda x, y: ssk.ssk_kernel(x, y, k, l)

def _ssk_kernel(x, y, k, l):
    """

    :param x: first string
    :param y: second string
    :param k: length
    :param l: lambda
    :return: SSK distance between two strings given parameters
    """

    # todo: kernel calculation
    return random.random()

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

    # todo: kernel calculation
    return 0.


def wk(data):
    """
    Get word kernel function. (tf-idf)

    :param data: 2 strings
    :return: function (X, Y) -> float
    """

    N = len(data)

    # calculate words frequencies per document
    wf = [Counter(X[0].split()) for X in data]

    # calculate document frequency
    df = Counter()
    map(df.update, (w.keys() for w in wf))

    def wk_kernel(x, y):
        len_x = len(re.split(' ', x))
        wf_x = Counter(re.split(' ', x))
        len_y = len(re.split(' ', y))
        wf_y = Counter(re.split(' ', y))

        k = 0.0

        for word, freq_x in wf_x.iteritems():
            freq_y = [y[1] for y in wf_y.iteritems() if y[0] == word]
            if len(freq_y) == 0:
                continue

            fx = math.log(1. + float(freq_x) / float(len_x)) * math.log(float(N) / float(df[word]))
            fy = math.log(1. + float(freq_y[0] / float(len_y))) * math.log(float(N) / float(df[word])) 
            k += fx * fy    

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
