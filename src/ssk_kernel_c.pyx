import math
import itertools as iter
import numpy as np
cimport numpy as np
cimport cython
from time import time

from string import lowercase

ctypedef np.float_t DTYPE_t
DTYPE = np.float

def ssk_kernel(s, t, int k, float l):
    """
    Recursive SSK implementation.
    :param s: document #1
    :param t: document #2
    :param k: subsequence length
    :param l: weight decay (lambda)
    :return: similarity of given documents
    return:
    """
    if s == t:
        return 1.
    elif min(len(s), len(t)) < k:
        return 0.
    cdef np.ndarray[DTYPE_t, ndim=3] K_prime = _compute_K_prime(s, t, k, l)
    cdef float K_st = _compute_K(s, t, k, l, K_prime)

    K_prime = _compute_K_prime(s, s, k, l)
    cdef float K_ss = _compute_K(s, s, k, l, K_prime)

    K_prime = _compute_K_prime(t, t, k, l)
    cdef float K_tt = _compute_K(t, t, k, l, K_prime)

    cdef float denominator = math.sqrt(K_ss * K_tt) if K_ss * K_tt else 10e-30
    return K_st / denominator

def ssk_kernel_many(s, t, ks, float l):
    """
    Recursive SSK implementation.
    :param s: document #1
    :param t: document #2
    :param k: subsequence length
    :param l: weight decay (lambda)
    :return: similarity of given documents
    return:
    """
    cdef np.ndarray[DTYPE_t, ndim=3] K_prime_st = _compute_K_prime(s, t, ks[-1], l)
    cdef np.ndarray[DTYPE_t, ndim=3] K_prime_ss = _compute_K_prime(s, s, ks[-1], l)
    cdef np.ndarray[DTYPE_t, ndim=3] K_prime_tt = _compute_K_prime(t, t, ks[-1], l)

    cdef int k
    result = []
    for k in ks:
        if s == t:
            result.append(1.)
        elif min(len(s), len(t)) < k:
            result.append(0.)
        else:
            K_st = _compute_K(s, t, k, l, K_prime_st)
            K_ss = _compute_K(s, s, k, l, K_prime_ss)
            K_tt = _compute_K(t, t, k, l, K_prime_tt)

            denominator = math.sqrt(K_ss * K_tt) if K_ss * K_tt else 10e-30
            result.append(K_st / denominator)
    return np.array(result)


def _compute_K(s, t, int k, float l, np.ndarray[DTYPE_t, ndim=3] K_prime):
    """
    Compute and return the K in a recursive manner using precomputed K'
    """
    cdef float K_val = 0
    cdef int m

    for m in xrange(len(s)+1):
        if min(len(s[:m]), len(t)) < k:
            continue

        K_val += l**2 * sum([K_prime[k-1][len(s[:m])-1][j] for j in _find_all_char_indices(s[m-1], t)])

    return K_val

@cython.boundscheck(False)
def _compute_K_prime(s_, t_, int k, float l):
    """
    Compute and return K' using the efficient DP algorithm (K'')
    """
    cdef int M = len(s_)
    cdef int N = len(t_)
    cdef np.ndarray[DTYPE_t, ndim=3] K_prime = np.ones((k, M+1, N+1), dtype=np.float)
    cdef np.ndarray[DTYPE_t, ndim=3] K_dprime = np.zeros((k, M+1, N+1), dtype=np.float)
    cdef int i, m, n
    cdef char* s = s_
    cdef char* t = t_

    for i in xrange(1, k):
        for m in xrange(M+1):
            for n in xrange(N+1):
                if min(m, n) < i:
                    K_prime[i, m, n] = 0
                    continue

                if s[m-1] != t[n-1]:
                    K_dprime[i, m, n] = l*K_dprime[i, m, n-1]
                else:
                    K_dprime[i, m, n] = l*(K_dprime[i, m, n-1] + l*K_prime[i-1, m-1, n-1])

                K_prime[i, m, n] = l*K_prime[i, m-1, n] + K_dprime[i, m, n]

    return K_prime


def _find_all_subsequence_indices(substring, string):
    char_indices = [_find_all_char_indices(char, string) for char in substring]

    def get_all_indices(idcs, gt=-1):
        if not idcs:
            return [[]]
        return [[idx] + sufix for idx in idcs[0] for sufix in get_all_indices(idcs[1:], idx) if idx > gt]

    return get_all_indices(char_indices)


def _find_all_char_indices(ch, string):
    return [idx for idx, ltr in enumerate(string) if ltr == ch]


def _subsequence_length(indices):
    return indices[-1] - indices[0] + 1
