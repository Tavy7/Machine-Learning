from python_speech_features import mfcc, ssc
from scipy.io import wavfile
import numpy as np
import librosa
import csv
import os

import scipy as sp

extragere = 2

def getTrainData():
	print("Citim train")

	path = "./data/"
	file = csv.reader(open(path + "train.txt"))
	path += "train/train/"

	c = 0

	trainData, trainLabels = [], []
	for line in file:
		trainLabels.append(int(line[1]))

		#print("Citim train " + line[0] + " " + str(line[1]))
		frecv, data = wavfile.read(path + line[0])

		if extragere == 0:
			data = mfcc(data, frecv)
		if extragere == 1:
			data = sp.signal.stft(data, frecv)
		if extragere == 2:
			data = ssc(data, frecv)

		trainData.append(data[0])

		#c += 1

		if c == 500:
			break

	return trainData, trainLabels


def getValidationData():
	print("Citim validation")


	path = "./data/"
	file = csv.reader(open(path + "validation.txt"))
	path += "validation/validation/"

	c = 0

	validationData, validationLabels = [], []
	for line in file:
		validationLabels.append(int(line[1]))

		#print("Citim validation " + line[0] + " " + str(line[1]))
		frecv, data = wavfile.read(path + line[0])

		if extragere == 0:
			data = mfcc(data, frecv)
		if extragere == 1:
			data = sp.signal.stft(data, frecv)
		if extragere == 2:
			data = ssc(data, frecv)

		validationData.append(data[0])

		#c += 1
		if c == 800:
			break

	return validationData, validationLabels

def getTestData():	
	print("Citim test")

	path = "./data/test/test/"
	nume = os.listdir(path)

	testData = []
	for i in nume:
		#print("Citim test " + str(i))
		frecv, data = wavfile.read(path + i)
		
		if extragere == 0:
			data = mfcc(data, frecv)
		if extragere == 1:
			data = sp.signal.stft(data, frecv)
		if extragere == 2:
			data = ssc(data, frecv)

		testData.append(data[0])
	
	return testData

def scrieConcluzie(labels):
	f = open("output.txt", "a")
	f.truncate(0)

	path = "./data/test/test/"
	nume = os.listdir(path)

	f.write("name,label\n")
	for i in range(len(nume)):
		f.write("{},{}\n".format(nume[i], labels[i]))
	f.close()





#59.9, max lbfgs
#nn =  MLPClassifier(solver = solver, alpha=1e-5, max_iter = 1000)

#60.3, std, rbf
#60.3 std, sigmoid 
#svm.SVC(kernel = 'rbf', C = 1.05)


def normal(data, norma):
	if norma == 'x':
		return data

	from sklearn.preprocessing import normalize
	return normalize(X = data, norm = norma)