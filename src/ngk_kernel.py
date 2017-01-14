def create_ngrams(text, n):
    """Create a set of ngrams of length n"""
    return set(text[i:i+n] for i in range(len(text)-n+1))


def ngk(doc1, doc2, n):
    sd1 = create_ngrams(doc1, n)
    sd2 = create_ngrams(doc2, n)

    if len(sd1 | sd2) == 0:
        return 1.0

    return len(sd1 & sd2) * 1.0 / len(sd1 | sd2)


if __name__ == "__main__":
    assert ngk("Das ist ein Test", "Das ist ein Test", 4) == 1.0
    print ngk("Das ist ein Test", "Das ist ein Test", 4)
    print ngk("Das ist ein Tlub", "Das ist ein Test", 4)
