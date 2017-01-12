import matplotlib.pyplot as plt
import util
import kernels
import operator
import numpy as np


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


def compare_subsets():
    """
    Run tests that are shown on Figure 1 in the article
    :return:
    """
    trainData, _ = util.load_cleaned_data('../data/train_data_clean.p', '../data/train_data_clean.p')
    trainDocs = [t[0] for t in trainData][:100]
    strings = extract_strings(trainDocs, 3)
    strings_shuffled = np.random.choice(strings, len(strings))
    print len(strings), 'substrings total'

    trueGram = kernels.compute_Gram_matrix(kernels.ssk(3, 0.1), trainDocs)
    print 'True Gram matrix was built'

    sizes = range(0, len(strings), 100)
    freq_sim = []
    infreq_sim = []
    rand_sim = []
    for n in sizes:
        print '\r', n,
        freq_strings = strings[:n]
        infreq_strings = strings[-n:]
        rand_strings = strings_shuffled[:n]

        freq_gram = gram_similarity(kernels.get_approximate_ssk_gram_matrix(freq_strings, trainDocs, 3, 0.1), trueGram)
        infreq_gram = gram_similarity(kernels.get_approximate_ssk_gram_matrix(infreq_strings, trainDocs, 3, 0.1), trueGram)
        rand_gram = gram_similarity(kernels.get_approximate_ssk_gram_matrix(rand_strings, trainDocs, 3, 0.1), trueGram)

        freq_sim.append(freq_gram)
        infreq_sim.append(infreq_gram)
        rand_sim.append(rand_gram)

    fig = plt.figure()
    plt.plot(sizes, freq_sim)
    plt.plot(sizes, infreq_sim)
    plt.plot(sizes, rand_sim)
    plt.legend(['Most frequent', 'Least frequent', 'Random'])
    fig.show()

