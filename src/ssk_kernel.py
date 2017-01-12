import math
import itertools as iter
import ssk_kernel as ssk

from string import lowercase


def ssk_kernel(s, t, k, l):
    """

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
