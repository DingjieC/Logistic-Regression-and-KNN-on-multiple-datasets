import math
import sys
def calEuclideanDistance(row1, row2):
	dist = 0.0
	attr_len = len(row1) - 1
	for i in range(attr_len):
		# sum of squared distances
		dist+=math.pow(row1[i]-row2[i],2)
	# square root of the distance
	return math.sqrt(dist)

def calManhattanDistance(row1, row2):
	dist = 0.0
	attr_len = len(row1) - 1
	for i in range(attr_len):
		dist+=abs(row1[i]-row2[i])
	return dist

def calMaxDistance(row1, row2):
	maxDist = float(-sys.maxsize	)
	dist = maxDist
	attr_len = len(row1) - 1
	for i in range(attr_len):
		# sum of squared distances
		dist = math.pow(row1[i]-row2[i],2)
		if dist > maxDist:
			maxDist = dist
	# square root of the distance
	return dist