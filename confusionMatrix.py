from __future__ import division
class ResultsMetric(object):
	"""docstring for ResultsMetric"""
	def __init__(self, accuracy, tpr, fpr, tp, fp, tn, fn):
		self.accuracy =accuracy
		self.tpr = tpr
		self.fpr = fpr
		self.tp = tp
		self.fp = fp
		self.tn = tn
		self.fn = fn

	def __str__(self):
		return ("Acc - %f TPR - %f, FPR - %f, TP - %f, FP - %f, TN - %f, FN - %f"%(self.accuracy, self.tpr, self.fpr, self.tp, self.fp, self.tn, self.fn))


def printConfMatrix(tp,fp,tn,fn):
	print("Confusion Matrix: ")
	print("\t Predicted")
	print("\t   |Class (1) |Class (0) |")
	print("Actual (1) | TP - %d | FN - %d |"%(tp,fn))
	print("Actual (0) | FP - %d | TN - %d |"%(fp,tn))

def computeAccuracy(testingData, predictions):
	tp = 0
	tn = 0
	fp = 0
	fn = 0
	totalRecords = len(testingData)
	classIndex = len(testingData[0]) - 1
	for i in range(totalRecords):
		trueLabel = testingData[i][classIndex]
		if trueLabel == predictions[i]:
			# this is either the true positive or true negative
			if trueLabel == 1:
				# true positive
				tp += 1
			else:
				# true negative
				tn += 1
		else:
			# this is either the false positive or false negative
			if trueLabel == 1:
				# false negative
				fn += 1
			else:
				# false positive
				fp += 1
	accuracy = (tp+tn)/(tp+tn+fp+fn) * 100.0
	accuracy = round(accuracy,4)
	#printConfMatrix(tp,fp,tn,fn)
	#print("Accuracy : "+str(accuracy)+" %.")
	tpr = tp/(tp+fn)
	fpr = fp/(fp+tn)
	return ResultsMetric(accuracy, tpr, fpr, tp, fp, tn, fn) 