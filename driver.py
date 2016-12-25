import knn
import sys
import os
from logistic_regression import LogisticRegression
from dataReader import readData
# import matplotlib.pyplot as plt
def usage():
	print("Wrong usage.")
	print "python driver.py <dataset dir> <training file name> <testing file name> <algo> [options] \n"+\
	"<dataset dir> - directory where the training and testing files reside\n"+\
	"<training file name> - a CSV file used for training. Our algorithms assume that the class label is the last attribute of the file\n"+\
	"<testing file name> - a CSV file used for testing.\n"+\
	"<algo> - param takes values as knn or lr\n"+\
	"knn - for kNN algorithm\n"+\
    "lr - for Logistic Regression\n"+ \
	"[options] -\n"+\
	"if <algo> is knn -\n"+\
    "options take value as\n"+\
	"<option1> - k value (this value is mandatory of knn to run)\n"+\
	"[option2] ... [option n] - rest are the indices for the best predictors\n"+\
	"if <algo> is lr -\n"+\
	"options take value as\n"+\
    "[option1] ... [option n] - the indices for the best predictors.\n"
	sys.exit(1)

if len(sys.argv) < 5:
	usage()

(datasetPath, trainingFileName, testingFileName, algo) = (sys.argv[1:5])
'''
If the user inputs no best predictors then, all the columns would be chosen for
running the algorithm

'''
if algo=="knn":
	if len(sys.argv) < 6:
		usage()
	k = int(sys.argv[5])
	best_predictors=[]
	for i in range(6,len(sys.argv)):
		best_predictors.append(int(sys.argv[i]))
	print("The best predictors chosen : ",best_predictors)
	print("datasetPath",datasetPath)
	print("trainFile:",trainingFileName)
	print("testingFile:",testingFileName)
	print("k set to ",k)
	knn.knn(datasetPath, trainingFileName, testingFileName, k, best_predictors)
elif algo=="lr":
	train_file_path = os.path.join(datasetPath, trainingFileName)
	test_file_path = os.path.join(datasetPath, testingFileName)
	train_data = []
	test_data = []
	best_predictors = []
	print "Reading train data!!!"
	readData(train_file_path, best_predictors, train_data)
	# print "Training data = ",train_data[0]
	# raw_input()
	print "Reading test data!!!"
	readData(test_file_path, best_predictors, test_data)
	logistic_regression = LogisticRegression()
	logistic_regression.compute_logistic_regression(train_data, test_data)
else:
	usage()
	sys.exit(1)