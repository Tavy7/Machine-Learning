from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
import numpy as np
import time

import fisiere

def normalize(data, test_features, type=None):
	from sklearn import preprocessing
	print("Normalizam folosind {}".format(type))
	
	if type == 'max':
		scaler = preprocessing.MinMaxScaler()
	elif type == 'std':
		scaler = preprocessing.StandardScaler()
	elif type == 'l1':
		scaler = preprocessing.Normalizer(norm='l1')
	elif type == 'l2':
		scaler = preprocessing.Normalizer(norm='l2')
	elif type == 'x':
		return data, test_features
	else:
		exit()
	

	scaler.fit(data)#calc mean si std
	scaled_train_feats = scaler.transform(data)#standardizeaza
	scaled_test_feats = scaler.transform(test_features)

	print("Done")
	return scaled_train_feats, scaled_test_feats

def bestSVM (trainData, trainLabels, validationData, validationLabels):
	maxAcc = 0


	for norma in ['x', 'l1', 'l2', 'max', 'std']:
		print(norma)
		scaledTrainData, scaledValidationData = normalize(trainData, validationData, norma)

		for kern in ['linear', 'poly', 'rbf', 'sigmoid']:

			print("Creem modelul", kern)
			model = svm.SVC(kernel = kern, C = 1.05)

			print("Train")
			model.fit(scaledTrainData, trainLabels)

			preds = model.predict(scaledValidationData)

			print("Accuracy")
			acc = accuracy_score(preds, validationLabels)#, normalize=False)
			print(acc)

			if maxAcc < acc:
				print("!")
				maxAcc = acc
				bestModel, bestNorm = model, norma

	print("Best acc: ", maxAcc)
	print("Stergem datele.")

	return bestModel, bestNorm, maxAcc

def bestLin(trainData, trainLabels, validationData, validationLabels):
	maxAcc = 0
	from sklearn.linear_model import LogisticRegression

	for penalty in ['l2', 'none']:
		for norma in ['x', 'l1', 'l2', 'max', 'std']:
			print("Norma", norma)
			print("Model", penalty)
			scaledTrainData, scaledValidationData = normalize(trainData, validationData, norma)

			clf = LogisticRegression(penalty = penalty)

			print("Train")
			clf.fit(scaledTrainData, trainLabels)

			print("Preds")
			preds = clf.predict(scaledValidationData)

			acc = accuracy_score(preds, validationLabels)
			print("Accuracy", acc)

			if maxAcc < acc:
				print("!")
				maxAcc = acc
				bestModel, bestNorm = clf, norma

	print("Best acc: ", maxAcc)
	print("Stergem datele.")

	return bestModel, bestNorm, maxAcc

def bestRandomForest(trainData, trainLabels, validationData, validationLabels):

	maxAcc = 0
	from sklearn.ensemble import RandomForestClassifier

	for i in range (10, 800, 250):
		for norma in ['x', 'l1', 'l2', 'max', 'std']:
			scaledTrainData, scaledValidationData = normalize(trainData, validationData, norma)
			print("Norma", norma)
			print(i, "copaci")

			rf =  RandomForestClassifier(n_estimators = i)

			print("Train")
			rf.fit(scaledTrainData, trainLabels)

			print("Preds")
			preds = rf.predict(scaledValidationData)

			acc = accuracy_score(preds, validationLabels)
			print("Accuracy", acc)

			if maxAcc < acc:
				print("!")
				maxAcc = acc
				bestModel, bestNorm = rf, norma

	print("Best acc: ", maxAcc)
	print("Stergem datele.")

	return bestModel, bestNorm, maxAcc

def bestNNet(trainData, trainLabels, validationData, validationLabels):
	maxAcc = 0

	for solver in ['lbfgs', 'sgd', 'adam']:
		for norma in ['x', 'l1', 'l2', 'max', 'std']:

			scaledTrainData, scaledValidationData = normalize(trainData, validationData, norma)
			print("Norma", norma)
			print(solver)

			nn =  MLPClassifier(solver = solver, alpha=1e-5, max_iter = 1000)
			nn1 = MLPClassifier(solver = solver, max_iter = 1000)

			print("Train")
			nn.fit(scaledTrainData, trainLabels)
			nn1.fit(scaledTrainData, trainLabels)

			print("Preds")
			preds = nn.predict(scaledValidationData)
			preds1 = nn1.predict(scaledValidationData)


			acc = accuracy_score(preds, validationLabels)
			acc1 = accuracy_score(preds1, validationLabels)

			print("Accuracy", acc, acc1)

			if maxAcc < acc1:
				print("!!")
				maxAcc = acc1
				bestModel, bestNorm = nn1, norma

			if maxAcc < acc:
				print("!")
				maxAcc = acc
				bestModel, bestNorm = nn, norma


	print("Best acc: ", maxAcc)
	print("Stergem datele.")

	return bestModel, bestNorm, maxAcc

	
def main():

	trainData, trainLabels = fisiere.getTrainData()
	validationData, validationLabels = fisiere.getValidationData()

	bestSVM1, svmNorm, svmAcc = bestSVM(trainData, trainLabels, validationData, validationLabels)
	bestLi, liNorm, liAcc = bestLin(trainData, trainLabels, validationData, validationLabels)
	bestRF, rfNorm, rfAcc = bestRandomForest(trainData, trainLabels, validationData, validationLabels)
	bestNN, nnNorm, nnAcc = bestNNet(trainData, trainLabels, validationData, validationLabels)

	# model01 = svm.SVC(C = 1.05)
	# svmTrainData, svmVerificationData = normalize(trainData, validationData, 'std')
	# model01.fit(svmTrainData, trainLabels)
	# pred1 = model01.predict(svmVerificationData)
	# acc1 = accuracy_score(pred1, validationLabels)
	# print(acc1)

	del trainData
	del trainLabels
	del validationData
	del validationLabels

	bestAcc = 0

	if bestAcc < svmAcc:
		bestModel = bestSVM1
		bestNorm = svmNorm
		bestAcc = svmAcc

	if bestAcc < liAcc:
		bestModel = bestLi
		bestNorm = liNorm
		bestAcc = liAcc

	if bestAcc < rfAcc:
		bestModel = bestRF 
		bestNorm = rfNorm
		bestAcc = rfAcc


	if bestAcc < nnAcc:
		bestModel = bestNN
		bestNorm = nnNorm
		bestAcc = nnAcc

	print("Best acc", bestAcc)

	testData = fisiere.getTestData()
	test = normalize(testData, testData, bestNorm)
	preds = bestModel.predict(test[0])

	del testData

	print("Scriem predictile in fisier.")
	fisiere.scrieConcluzie(preds)
	

	print(svmAcc, liAcc, rfAcc, nnAcc)


if __name__ == "__main__":

	x = float(round(time.time()))
	main()
	print("Programul a rulat ", str((float(round(time.time())) - x)), " secunde.")
	exit()