import pickle

def evaluate_pred(y, pred):
	TP = 0 # true positive
	FP = 0 # false positive
	FN = 0 # false negative
	for i in range(0, len(y)):
		if pred[i] == 1. and y[i] == 1.:
			TP += 1
		elif pred[i] == 1. and y[i] == -1.:
			FP += 1
		elif pred[i] == -1. and y[i] == 1.:
			FN += 1
	precision = float(TP) / float(TP + FP)
	recall = float(TP) / float(TP + FN)
	F1 = 2. * precision * recall / (precision + recall)
	return F1, precision, recall
