from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import svm
import numpy as np
import fisiere


def normalize(data):
	scaler = preprocessing.StandardScaler()
	scaler.fit(data)
	data = scaler.transform(data)

	return data

def main():
	trainData, trainLabels = fisiere.getTrainData()#citire train
	validationData, validationLabels = fisiere.getValidationData()#citire validation

	model = svm.SVC(C = 1.05, kernel="rbf")#initializare model
	
	trainData = normalize(trainData)#normalizam train data
	validationData = normalize(validationData)#normalizam valdiation data

	model.fit(trainData, trainLabels)#antrenam
	
	pred = model.predict(validationData)#creem predictii pentru validation data
	acc = accuracy_score(pred, validationLabels)#verificam acuratetea

	testData = fisiere.getTestData()#citire test data
	testData = normalize(testData)#normalizam test data
	preds = model.predict(testData)#creem predictii

	print("Scriem predictile in fisier.")
	fisiere.scrieConcluzie(preds)#scriem in fisier predictile	


if __name__ == "__main__":
	main()
