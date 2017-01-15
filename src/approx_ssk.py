import matplotlib.pyplot as plt
import data_handling as dh
import kernels
import operator
import numpy as np
import pickle
import ssk_kernel_c
import math
import multiprocessing as mp

from joblib import Parallel, delayed
import multiprocessing



def extract_strings(docs, n):
    """
    :param docs: list of documents
    :param n: substring size
    :return: sorted list of substrings (first one is the most frequent)
    """

    strings = dict()
    for doc in docs:
        for i in range(len(doc) - n + 1):
            line = doc[i: i + n]
            if line not in strings:
                strings[line] = 1
            else:
                strings[line] += 1

    sorted_strings = sorted(strings.items(), key=operator.itemgetter(1), reverse=True)
    sorted_strings = [x[0] for x in sorted_strings]
    return sorted_strings


def gram_similarity(K1, K2):
    return np.sum(K1 * K2) / (1e-9 + np.sqrt(np.sum(K1 * K1) * np.sum(K2 * K2)))


def ssk_picklable(s, t, k, l):
    return ssk_kernel_c.ssk_kernel(s, t, k, l)


def compute_K(s, t, k, l):
    return ssk_kernel_c._compute_K(s, t, k, l, ssk_kernel_c._compute_K_prime(s, t, k, l))

def compute_K_par(args):
    s, t, k, l, i, j = args
    if i == j:
        print '\rcur:',  i,
    return i, j, ssk_kernel_c._compute_K(s, t, k, l, ssk_kernel_c._compute_K_prime(s, t, k, l))


def calc_subkernels(data, strings, k, l):
    # ssk_kernel = kernels.ssk(k, l)
    n = len(data)
    n2 = len(strings)
    subkernels = np.empty((n, n2))
    data_self = np.array(Parallel(n_jobs=4)(delayed(compute_K)(x, x, k, l) for x in data))
    strings_self = np.array(Parallel(n_jobs=4)(delayed(compute_K)(x, x, k, l) for x in strings))
    for i in range(n):
        print '\rcur: ', i, '/', n,
        subkernels[i, :] = Parallel(n_jobs=4)(delayed(compute_K)(data[i], s, k, l) for s in strings)
    for i in range(n):
        for j in range(n2):
            denominator = math.sqrt(data_self[i] * strings_self[j]) if data_self[i] * strings_self[j] else 10e-30
            subkernels[i, j] /= denominator
    print '\r',
    return subkernels


def calc_subkernels_par(data, strings, k, l):
    # ssk_kernel = kernels.ssk(k, l)
    n = len(data)
    n2 = len(strings)
    subkernels = np.empty((n, n2))
    data_self = np.array(Parallel(n_jobs=4)(delayed(compute_K)(x, x, k, l) for x in data))
    strings_self = np.array(Parallel(n_jobs=4)(delayed(compute_K)(x, x, k, l) for x in strings))

    data = [(data[i], strings[j], k, l, i, j) for i in xrange(len(data)) for j in xrange(len(strings))]

    workers = mp.Pool(processes=4)
    results = workers.map(compute_K_par, data)

    print '\r',

    for i, j, result in results:
        denominator = math.sqrt(data_self[i] * strings_self[j]) if data_self[i] * strings_self[j] else 10e-30
        subkernels[i, j] = result / denominator

    return subkernels


def _save_subkernels(k, data, strings, output_path):
    docs = [t[0] for t in data]#[:10]

    subkernels = calc_subkernels(docs, strings, k, 0.5)
    with open(outpput_path, 'wb') as fd:
        pickle.dump(subkernels, fd)


def compare_subsets():
    """
    Run tests that are shown on Figure 1 in the article
    :return:
    """
    trainData, _ = dh.load_pickled_data('../data/train_data_clean.p', '../data/test_data_clean.p')

    with open('../data/approx/true_ssk_3_05.p') as fd:
        trueGram = pickle.load(fd)
    with open('../data/approx/subkernels.p') as fd:
        subkernels = pickle.load(fd)
    print subkernels.shape[1], 'substrings total'
    shuffled_idx = np.random.permutation(subkernels.shape[1])
    subkernels_shuffled = subkernels[:, shuffled_idx]

    sizes = range(1, subkernels.shape[1], 1)
    freq_sim = []
    infreq_sim = []
    rand_sim = []
    for n in sizes:
        if n % 100 == 0:
            print n

        freq_gram = gram_similarity(kernels.compute_Gram_matrix(np.dot, subkernels[:, :n]), trueGram)
        infreq_gram = gram_similarity(kernels.compute_Gram_matrix(np.dot, subkernels[:, -n:]), trueGram)
        rand_gram = gram_similarity(kernels.compute_Gram_matrix(np.dot, subkernels_shuffled[:, :n]), trueGram)

        freq_sim.append(freq_gram)
        infreq_sim.append(infreq_gram)
        rand_sim.append(rand_gram)

    data = np.empty((len(sizes), 4))
    data[:, 0] = np.array(sizes)
    data[:, 1] = freq_sim
    data[:, 2] = infreq_sim
    data[:, 3] = rand_sim

    with open('../data/approx/stats.p', 'wb') as fd:
        pickle.dump(data, fd)
    # fig = plt.figure()
    # plt.plot(sizes, freq_sim)
    # plt.plot(sizes, infreq_sim)
    # plt.plot(sizes, rand_sim)
    # plt.legend(['Most frequent', 'Least frequent', 'Random'])
    # fig.show()



#
# if __name__ == '__main__':
#     k = 3
#     trainData, _ = dh.load_pickled_data('../data/train_data_clean.p', '../data/test_data_clean.p')
#     trainDocs = [t[0] for t in trainData]
#     strings = extract_strings(trainDocs, k)
#     print len(strings), 'substrings total'
#     strings = strings[:3000]
#     with open('../data/approx/strings-3000-{}.p'.format(k), 'wb') as fd:
#         pickle.dump(strings, fd)


# if __name__ == '__main__':
#     trainData, testData = dh.load_pickled_data('../data/train_data_clean.p', '../data/test_data_clean.p')
#     trainData = [(x[0].encode('ascii', 'ignore'), x[1]) for x in trainData]
#     testData = [(x[0].encode('ascii', 'ignore'), x[1]) for x in testData]
#     with open('../data/train_data_nounicode.p', 'wb') as fd:
#         pickle.dump(trainData, fd)
#     with open('../data/test_data_nounicode.p', 'wb') as fd:
#         pickle.dump(testData, fd)


if __name__ == '__main__':
    k = 5
    trainData, testData = dh.load_pickled_data('../data/train_data_nounicode.p', '../data/test_data_nounicode.p')
    with open('../data/approx/strings-3000-{}.p'.format(k)) as fd:
        strings = pickle.load(fd)
    _save_subkernels(k, trainData, strings, '../data/approx/train-subkernels-3000-{}-05.p'.format(k))
    _save_subkernels(k, testData, strings, '../data/approx/test-subkernels-3000-{}-05.p'.format(k))

