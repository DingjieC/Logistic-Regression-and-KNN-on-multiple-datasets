import operator
import os
import math
from dataReader import readData
from distances import calManhattanDistance
from distances import calEuclideanDistance
from distances import calMaxDistance
from confusionMatrix import *
trainingData=[]
testingData=[]


'''
method: 1. Euclidean distance
method: 2. Manhattan distance
method: 3. Complete link
'''
def calDistance(method, row1, row2):
	if method==1:
		return calEuclideanDistance(row1, row2)
	elif method==2:
		return calManhattanDistance(row1, row2)
	elif method==3:
		return calMaxDistance(row1, row2)
'''
return list of k neighbors
'''
def findKNearestNeighbors(testSample, k, distanceMetric):
	lenTrainingDataSet = len(trainingData)
	dist=[]# this list will hold distances from all training samples to the
			# testing example
	dist = [(trainingData[i], calDistance(distanceMetric, trainingData[i], testSample)) for i in range(lenTrainingDataSet)]
	# sort the distance to get the closest 'k' training points
	dist.sort(key=operator.itemgetter(1))
	return [dist[j][0] for j in range(k)]

'''
take majority of the neighbors' class labels
'''
def getClassLabel(kNeighbors, k):
	classIndex = len(kNeighbors[0])-1
	classDict = {}
	for i in range(k):
		c = kNeighbors[i][classIndex]
		if c not in classDict:
			classDict[c] = 0
		classDict[c] += 1
	# the sorted api will sort the dictionary and return a list
	# the 0th element would be the label with highest number of votes
	return sorted(classDict, reverse=True)[0]

def predict(testRec, k, distanceMetric):
	# print("from predict():",trainingData[0])
	neighbors = findKNearestNeighbors(testRec, k, distanceMetric)
	return getClassLabel(neighbors, k)

def knn(datasetDir, trainingFileName, testingFileName, k, best_predictors, distanceMetric=1):

	if datasetDir != None:
		# here the data reading is done, since there is a dataset path variable supplied.
		# 
		if not best_predictors:
			print("No best predictors were supplied.")
			print("Considering all the columns for kNN.")

		print("Reading training data.")
		#readData(datasetDir+"trainingDataNorm.csv",trainingData)
		readData(os.path.join(datasetDir,trainingFileName), best_predictors, trainingData)
		print("Done reading")
		print("Reading testing data.")
		readData(os.path.join(datasetDir,testingFileName), best_predictors, testingData)
		print("Done reading")

	# this is a check done to test knn for various values of k. If datasetDir is None, 
	# it implies that the data has already been read before and the the other two param-
	# eters trainingFileName and testingFileName, hold the list of training and testing
	# datasets.
	
	predictions = []
	i=1
	for testRec in testingData:
		predictedValue = int(predict(testRec,k,distanceMetric))
		predictions.append(predictedValue)
		#print("Done with ",i," test record(s)")
		i+=1
	print("Done with predictions. Computing accuracy.")
	resultMetric = computeAccuracy(testingData, predictions)
	printConfMatrix(resultMetric.tp, resultMetric.fp, resultMetric.tn, resultMetric.fn)
	print resultMetric
	return resultMetric