import matplotlib.pyplot as plt
import data_handling as dh
import kernels
import operator
import numpy as np
import pickle
import ssk_kernel_c

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


def calc_subkernels(data, strings, k, l):
    ssk_kernel = kernels.ssk(k, l)
    n = len(data)
    n2 = len(strings)
    subkernels = np.empty((n, n2))
    for i in range(n):
        print '\rcur: ', i, '/', n,

        # def func(j):
        #     print j
        #     # return ssk_kernel(data[i], strings[j])

        subkernels[i, :] = Parallel(n_jobs=4)(delayed(ssk_picklable)(data[i], s, k, l) for s in strings)
        # for j in range(n2):
        #     subkernels[i, j] = ssk_kernel(data[i], strings[j])
    print '\r',
    return subkernels


def _save_subkernels():
    trainData, _ = dh.load_pickled_data('../data/train_data_clean.p', '../data/train_data_clean.p')
    trainDocs = [t[0] for t in trainData][:100]
    strings = extract_strings(trainDocs, 3)
    print len(strings), 'substrings total'

    subkernels = calc_subkernels(trainDocs, strings, 3, 0.5)
    with open('../data/approx/_subkernels.p', 'wb') as fd:
        pickle.dump(subkernels, fd)


def compare_subsets():
    """
    Run tests that are shown on Figure 1 in the article
    :return:
    """
    trainData, _ = dh.load_pickled_data('../data/train_data_clean.p', '../data/train_data_clean.p')
    trainDocs = [t[0] for t in trainData][:100]
    strings = extract_strings(trainDocs, 3)
    strings_shuffled = np.random.choice(strings, len(strings))
    print len(strings), 'substrings total'

    trueGram = kernels.compute_ssk_Gram_matrix(3, 0.5, trainDocs)
    print 'True Gram matrix was built'
    with open('../data/approx/true_ssk_3_05.p', 'wb') as fd:
        pickle.dump(trueGram, fd)

    sizes = range(100, len(strings), 100)
    freq_sim = []
    infreq_sim = []
    rand_sim = []
    for n in sizes:
        print n
        freq_strings = strings[:n]
        infreq_strings = strings[-n:]
        rand_strings = strings_shuffled[:n]

        freq_gram = gram_similarity(kernels.get_approximate_ssk_gram_matrix(freq_strings, trainDocs, 3, 0.9), trueGram)
        with open('../data/approx/freq_{}.p'.format(n), 'wb') as fd:
            pickle.dump(freq_gram, fd)
        infreq_gram = gram_similarity(kernels.get_approximate_ssk_gram_matrix(infreq_strings, trainDocs, 3, 0.9), trueGram)
        with open('../data/approx/infreq_{}.p'.format(n), 'wb') as fd:
            pickle.dump(freq_gram, fd)
        rand_gram = gram_similarity(kernels.get_approximate_ssk_gram_matrix(rand_strings, trainDocs, 3, 0.9), trueGram)
        with open('../data/approx/rand_{}.p'.format(n), 'wb') as fd:
            pickle.dump(freq_gram, fd)

        freq_sim.append(freq_gram)
        infreq_sim.append(infreq_gram)
        rand_sim.append(rand_gram)

    fig = plt.figure()
    plt.plot(sizes, freq_sim)
    plt.plot(sizes, infreq_sim)
    plt.plot(sizes, rand_sim)
    plt.legend(['Most frequent', 'Least frequent', 'Random'])
    fig.show()

# _save_subkernels()
compare_subsets()

