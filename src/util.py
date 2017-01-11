import pickle

def load_cleaned_data(trainDataPath, testDataPath):
	with open(trainDataPath) as fd:
  		trainData = pickle.load(fd)
  	with open(testDataPath) as fd:
  		testData = pickle.load(fd)
  	return trainData, testData
  		