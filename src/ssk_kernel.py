import math
import itertools as iter
import ssk_kernel as ssk
import numpy as np

from string import lowercase


def naive_ssk_kernel(s, t, k, l):
    """
    Naive implementation

    :param x: first string
    :param y: second string
    :param k: length
    :param l: lambda
    :return: SSK distance between two strings given parameters
    """

    kernel_sum = 0
    for subseq in iter.permutations(lowercase + ' ', k):
        for idc_s in _find_all_subsequence_indices(subseq, s):
            for idc_t in _find_all_subsequence_indices(subseq, t):
                print subseq, s, idc_s, t, idc_t
                kernel_sum += math.pow(l, _subsequence_length(idc_s)+_subsequence_length(idc_t))
    return kernel_sum


def _subsequence_length(indices):
    return indices[-1] - indices[0] + 1


def _find_all_subsequence_indices(substring, string):
    char_indices = [_find_all_char_indices(char, string) for char in substring]

    def get_all_indices(idcs, gt=-1):
        if not idcs:
            return [[]]
        return [[idx] + sufix for idx in idcs[0] for sufix in get_all_indices(idcs[1:], idx) if idx > gt]

    return get_all_indices(char_indices)


def _find_all_char_indices(char, string):
    return [idx for idx, ltr in enumerate(string) if ltr == char]


def compute_K_prime(s, t, k, l):
    K_prime = np.ones((k-1, len(s), len(t)))

    for i in xrange(1, k-1):
        for m in xrange(len(s)):
            for n in xrange(len(t)):
                if min(m, n) < i:
                    K_prime[i][m][n] = 0
                    continue

                K_prime[i][m][n] = l*K_prime[i][m-1][n] + \
                       sum([K_prime[i-1][m-1][j-1] * l**(len(t)-j+2) for j in _find_all_char_indices(s[-1], t)])

    return K_prime


def ssk_kernel(s, t, k, l, K_prime):
    if min(len(s), len(t)) < k:
        return 0

    print s, t, _find_all_char_indices(s[-1], t)
    return ssk_kernel(s[:-1], t, k, l, K_prime) + \
           l**2 * sum([K_prime[k-2][len(s)-1][j-1] for j in _find_all_char_indices(s[-1], t)])

if __name__ == '__main__':
    str_a = 'science is organized knowledge'
    str_b = 'wisdom is organized life'

    str_a = 'cats'
    str_b = 'camats'

    k = 3
    l = 0.95
    K_prime = compute_K_prime(str_a, str_b, k, l)
    print K_prime.shape
    ssk_ab = ssk_kernel(str_a, str_b, k, l, K_prime)
    ssk_a = ssk_kernel(str_a, str_a, k, l, K_prime)
    ssk_b = ssk_kernel(str_b, str_b, k, l, K_prime)
    print 'AB:', ssk_ab, 'A:', ssk_a, 'B:', ssk_b, 'NORM', ssk_ab / math.sqrt(ssk_a * ssk_b)

    ssk_ab = naive_ssk_kernel(str_a, str_b, k, l)
    print 'After AB'
    ssk_a = naive_ssk_kernel(str_a, str_a, k, l)
    ssk_b = naive_ssk_kernel(str_b, str_b, k, l)
    print 'AB:', ssk_ab, 'A:', ssk_a, 'B:', ssk_b, 'NORM', ssk_ab / math.sqrt(ssk_a * ssk_b)
