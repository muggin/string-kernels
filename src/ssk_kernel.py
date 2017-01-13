import math
import itertools as iter
import ssk_kernel as ssk
import numpy as np

from string import lowercase


def naive_ssk_kernel(s, t, k, l):
    kernel_sum = 0
    for subseq in iter.permutations(lowercase + ' ', k):
        for idc_s in _find_all_subsequence_indices(subseq, s):
            for idc_t in _find_all_subsequence_indices(subseq, t):
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


def _compute_K_prime(s, t, k, l):
    K_prime = np.ones((k, len(s)+1, len(t)+1))

    for i in xrange(1, k):
        for m in xrange(len(s)+1):
            for n in xrange(len(t)+1):
                if min(m, n) < i:
                    K_prime[i][m][n] = 0
                    continue

                K_prime[i][m][n] = l*K_prime[i][m-1][n] + \
                       sum([K_prime[i-1][m-1][j] * l**(n-j+1) for j in _find_all_char_indices(s[m-1], t[:n])])

    return K_prime


def _ssk_kernel(s, t, k, l, K_prime):
    if min(len(s), len(t)) < k:
        return 0

    return _ssk_kernel(s[:-1], t, k, l, K_prime) + \
           l**2 * sum([K_prime[k-1][len(s)-1][j] for j in _find_all_char_indices(s[-1], t)])


def ssk_kernel(s, t, k, l, K_prime):
    K_prime = _compute_K_prime(str_a, str_b, k, l)


if __name__ == '__main__':
    test_set = [
        ('cats', 'cats'),
        ('dogs', 'dogs'),
        ('hats', 'cats'),
        ('wojtek', 'wojciech'),
        ('science', 'knowledge'),
        ('science is organized knowledge', 'wisdom is organized life')
    ]

    k = 3
    l = 0.9
    for str_a, str_b in test_set:
        print '{} ~ {}'.format(str_a, str_b)

        K_prime = _compute_K_prime(str_a, str_b, k, l)
        ssk_ab = _ssk_kernel(str_a, str_b, k, l, K_prime)
        K_prime = _compute_K_prime(str_a, str_a, k, l)
        ssk_a = _ssk_kernel(str_a, str_a, k, l, K_prime)
        K_prime = _compute_K_prime(str_b, str_b, k, l)
        ssk_b = _ssk_kernel(str_b, str_b, k, l, K_prime)
        print 'AB:', ssk_ab, 'A:', ssk_a, 'B:', ssk_b, 'NORM', ssk_ab / math.sqrt(ssk_a * ssk_b)

        ssk_ab = naive_ssk_kernel(str_a, str_b, k, l)
        ssk_a = naive_ssk_kernel(str_a, str_a, k, l)
        ssk_b = naive_ssk_kernel(str_b, str_b, k, l)
        print 'AB:', ssk_ab, 'A:', ssk_a, 'B:', ssk_b, 'NORM', ssk_ab / math.sqrt(ssk_a * ssk_b), '\n'

