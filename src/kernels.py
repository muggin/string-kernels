import numpy as np

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
    return lambda x, y: _ssk_kernel(x, y, k, l)


def _ssk_kernel(x, y, k, l):
    """

    :param x: first string
    :param y: second string
    :param k: length
    :param l: lambda
    :return: SSK distance between two strings given parameters
    """

    # todo: kernel calculation
    return 0.


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

    :param data: list of documents
    :return: function (X, Y) -> float
    """

    # todo: data preprocessing (term extraction, etc)

    def wk_kernel(x, y):
        # todo: kernel calculation
        return 0

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
