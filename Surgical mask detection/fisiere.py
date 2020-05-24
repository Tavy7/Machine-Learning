import numpy as np
import librosa
import csv
import os

def extractFeatures(data, frecv):
	data = librosa.feature.mfcc(data, sr=frecv)#extragem mfcc din data
	data = np.mean(data.T, axis=0)#pentru fiecare feature calculam media

	return data


def getTrainData():
	path = "./data/"#initializam path
	file = csv.reader(open(path + "train.txt"))#deschidem fisierul ce contine labels
	path += "train/train/"#completam path

	trainData, trainLabels = [], []#initializam data de returnat
	for line in file:
		trainLabels.append(int(line[1]))

		data, frecv = librosa.load(path + line[0])#citim fisierul line[0]
		data = extractFeatures(data, frecv)#extragem features
		
		trainData.append(data)
	return trainData, trainLabels


def getValidationData():
	path = "./data/"#initializam path
	file = csv.reader(open(path + "validation.txt"))
	path += "validation/validation/"

	validationData, validationLabels = [], []#initializam data de returnat
	for line in file:
		validationLabels.append(int(line[1]))

		data, frecv = librosa.load(path + line[0])#citim fisierul line[0]
		data = extractFeatures(data, frecv)#extragem features

		validationData.append(data)
	return validationData, validationLabels


def getTestData():	
	path = "./data/test/test/"#initializam path
	nume = os.listdir(path)

	testData = []#initializam data de returnat
	for i in nume:
		data, frecv = librosa.load(path + i)#citim fisierul i
		data = extractFeatures(data, frecv)#extragem features

		testData.append(data)
	
	return testData


def scrieConcluzie(labels):
	f = open("output.txt", "a")
	f.truncate(0)#stergere continut fisier

	path = "./data/test/test/"
	nume = os.listdir(path)

	f.write("name,label\n")#scriere header
	for i in range(len(nume)):
		f.write("{},{}\n".format(nume[i], labels[i]))#scriere nume si label aferent
	f.close()
