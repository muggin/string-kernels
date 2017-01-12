import pickle

def load_cleaned_data(trainDataPath, testDataPath):
	with open(trainDataPath) as fd:
  		trainData = pickle.load(fd)
  	with open(testDataPath) as fd:
  		testData = pickle.load(fd)
  	return trainData, testData
  		
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