import math
import itertools as iter
import numpy as np

from string import lowercase


def naive_ssk_kernel(s, t, k, l):
    """
    Naive (direct) SSK implementation. Very unefficient, use only for short strings.
    :param s: document #1
    :param t: document #2
    :param k: subsequence length
    :param l: weight decay (lambda)
    :return: similarity of given documents
    """
    kernel_sum = 0
    for subseq in iter.permutations(lowercase + ' ', k):
        for idc_s in _find_all_subsequence_indices(subseq, s):
            for idc_t in _find_all_subsequence_indices(subseq, t):
                kernel_sum += math.pow(l, _subsequence_length(idc_s)+_subsequence_length(idc_t))
    return kernel_sum


def ssk_kernel(s, t, k, l):
    """
    Recursive SSK implementation.
    :param s: document #1
    :param t: document #2
    :param k: subsequence length
    :param l: weight decay (lambda)
    :return: similarity of given documents
    return:
    """
    K_prime = _compute_K_prime(s, t, k, l)
    K_st = _compute_K(s, t, k, l, K_prime)

    K_prime = _compute_K_prime(s, s, k, l)
    K_ss = _compute_K(s, s, k, l, K_prime)

    K_prime = _compute_K_prime(t, t, k, l)
    K_tt = _compute_K(t, t, k, l, K_prime)

    return K_st / math.sqrt(K_ss * K_tt)


def _compute_K(s, t, k, l, K_prime):
    """
    Compute and return the K in a recursive manner using precomputed K'
    """
    K_val = 0

    for m in xrange(len(s)+1):
        if min(len(s[:m]), len(t)) < k:
            continue

        K_val += l**2 * sum([K_prime[k-1][len(s[:m])-1][j] for j in _find_all_char_indices(s[m-1], t)])

    return K_val


def _compute_K_prime(s, t, k, l):
    """
    Compute and return K' using the efficient DP algorithm (K'')
    """
    K_prime = np.ones((k, len(s)+1, len(t)+1))
    K_dprime = np.zeros((k, len(s)+1, len(t)+1))

    for i in xrange(1, k):
        for m in xrange(len(s)+1):
            for n in xrange(len(t)+1):
                if min(m, n) < i:
                    K_prime[i][m][n] = 0
                    continue

                if s[m-1] != t[n-1]:
                    K_dprime[i][m][n] = l*K_dprime[i][m][n-1]
                else:
                    K_dprime[i][m][n] = l*(K_dprime[i][m][n-1] + l*K_prime[i-1][m-1][n-1])

                K_prime[i][m][n] = l*K_prime[i][m-1][n] + K_dprime[i][m][n]

    return K_prime


def _find_all_subsequence_indices(substring, string):
    char_indices = [_find_all_char_indices(char, string) for char in substring]

    def get_all_indices(idcs, gt=-1):
        if not idcs:
            return [[]]
        return [[idx] + sufix for idx in idcs[0] for sufix in get_all_indices(idcs[1:], idx) if idx > gt]

    return get_all_indices(char_indices)


def _find_all_char_indices(char, string):
    return [idx for idx, ltr in enumerate(string) if ltr == char]


def _subsequence_length(indices):
    return indices[-1] - indices[0] + 1
