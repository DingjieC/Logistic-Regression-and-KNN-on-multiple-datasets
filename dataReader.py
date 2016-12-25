import csv
def readData(filepath, best_predictors, data=[]):
	with open(filepath,'r') as f:
		datalist = list(csv.reader(f))
		
		column_len = len(datalist[0])
		for i in range(len(datalist)):
			#print(i)
			if i!=0:
				relevantDataList = []
				# the read data may be read in string format;
				# thus converting it into float type
				for j in range(column_len-1):
					if not best_predictors or j in best_predictors:
						relevantDataList.append(round(float(datalist[i][j]),4))
				relevantDataList.append(int(float(datalist[i][column_len-1])))
				data.append(relevantDataList)