

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
