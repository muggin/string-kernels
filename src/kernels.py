import ssk_kernel as ssk

def ssk(k, l):
    """
    Get SSK kernel function with given parameters.

    :param k: length
    :param l: lambda
    :return: function (X, Y) -> float
    """
    return lambda x, y: ssk.ssk_kernel(x, y, k, l)



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


if __name__ == '__main__':
    str_a = 'science is organized knowledge' * 5
    str_b = 'wisdom is organized life' * 5

    # str_a = 'car'
    # str_b = 'cat'
    ssk_ab = _ssk_kernel(str_a, str_b, 3, 0.9)
    ssk_a = _ssk_kernel(str_a, str_a, 3, 0.9)
    ssk_b = _ssk_kernel(str_b, str_b, 3, 0.9)

    print ssk_ab, ssk_a, ssk_b
    print ssk_ab / math.sqrt(ssk_a * ssk_b)

