import pickle


def evaluate_pred(y, pred, cat=1.):
    TP = 0  # true positive
    FP = 0  # false positive
    FN = 0  # false negative
    for i in range(0, len(y)):
        if pred[i] == cat and y[i] == cat:
            TP += 1
        elif pred[i] == cat and y[i] != cat:
            FP += 1
        elif pred[i] != cat and y[i] == cat:
            FN += 1

    if TP + FP == 0:
        precision = 0.0
    else:
        precision = float(TP) / float(TP + FP)

    recall = float(TP) / float(TP + FN)

    if precision + recall == 0:
        F1 = 0.0
    else:
        F1 = 2. * precision * recall / (precision + recall)

    return F1, precision, recall
