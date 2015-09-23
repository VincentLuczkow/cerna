import sys
import math
import random
import featProbCalc as fpc
from functools import reduce
import numpy as np


# Draws n points from unif(0,1). Returns them in sorted order.
# used to generate random sequences of binding sites.
def createRandomSequence(n):
	# Choose n points in (0,1)
	sites = []
	for i in range(0, n):
		k = random.uniform(0, 1)
		sites.append(k)
	return sorted(sites)


# Generates an RNA string
def generateString(length, alphabet={0: 'a', 1: 'c', 2: 'g', 3: 't'}):
	string = ''
	for i in range(0, length):
		string += alphabet[random.randint(0, 3)]
	return string


# Feature 1
def NumberOfBindingSites(seqLength, motLength):
	sequence = generateString(seqLength)
	motif = generateString(motLength)
	return sequence.count(motif)


# Feature 3
# This isn't working, so it's just an empty method
def clusteringOfBindingSites(sequence, motif, sites):
	return 0


# Feature 4 Test 1
# Calculates the squared error between each distance and the average
# distance. Returns their sum
def feat4Test1(n, sites):
	temp = 1.0 / (n + 1)
	distances = list(map(lambda x, y: x - y, sites[1:], sites[:-1]))
	yValues = list(map(lambda x: (x - temp) ** 2, distances))
	total = sum(yValues)
	return total


# Feature 4 Test 2
# Divides a sequence into bins, calculates the numbers of bins filled by
# a random sequence
def feat4Test2(n, sites):
	# Binary value. Says whether or not a bin is filled.
	bins = np.ones(n)
	binPoints = []
	# Split
	for i in range(1, n + 1):
		binPoints.append(float(i) / n)
	curPointNumber = 0
	curBinNumber = 0
	for i in range(0, len(bins)):
		while curPointNumber < n and sites[curPointNumber] <= binPoints[i]:
			bins[i] = 0
			curPointNumber += 1
	binsNotFilled = reduce(lambda x, y: x + y, bins, 0)
	return binsNotFilled


# Feature 4 Test 3
# Calculates the squared distances between each binding site.
def feat4Test3(n, sites):
	distances = list(map(lambda x, y: x - y, sites[1:], sites[:-1]))
	uSqd = list(map(lambda x: x ** 2, distances))
	return uSqd


def main(trials=1000):
	# Stores feature 1 data
	numberOfBindingSites = np.zeros(trials)

	# Stores feature 2 data
	spanOfBindingSites = np.zeros(trials)

	sequenceLength = random.randint(900, 1100)
	motifLength = random.randint(4, 8)

	# For features 3 and 4
	# n = The number of times the motif appears.
	# n = random.randint(1,150)
	n = 100
	floatn = float(n)

	sumYs = np.zeros(trials)
	bins = np.zeros(trials)
	uSqds = np.zeros(trials * (n - 1))

	for i in range(trials):
		print("Trial: " + str(i))

		sites = createRandomSequence(n)

		# Feature 1
		numberOfBindingSites[i] = NumberOfBindingSites(sequenceLength, motifLength)

		# Feature 2
		try:
			spanOfBindingSites[i] = sites[len(sites) - 1] - sites[0]
		except:
			spanOfBindingSites[i] = 0

		# Feature 4 Test 1
		sumYs[i] = feat4Test1(n, sites)

		# Feature 4 Test 2
		bins[i] = feat4Test2(n, sites)

		# Feature 4 Test 3
		uSqdCur = feat4Test3(n, sites)
		j = i * (n - 1)
		for k in range(j, j + n - 1):
			uSqds[k] = uSqdCur[k - j]

	# Feature 1
	######################################################################
	feature1ExperimentalMean = np.mean(numberOfBindingSites)
	feature1PredictedMean = (sequenceLength - motifLength + 1) * (0.25 ** motifLength)
	print("Feature 1 Experimental Mean: " + str(feature1ExperimentalMean))
	print("Feature 1 Predicted Mean: " + str(feature1PredictedMean))
	######################################################################

	# Feature 2
	######################################################################
	feature2ExperimentalMean = np.mean(spanOfBindingSites)
	feature2PredictedMean = 0
	print("Feature 2 Experimental Mean: " + str(feature2ExperimentalMean))
	print("Feature 2 Predicted Mean: " + str(feature2PredictedMean))
	######################################################################

	# Feature 3
	######################################################################
	feature3ExperimentalMean = np.mean(sumYs)
	feature3PredictedMean = clusteringOfBindingSites(sequence, motif, sites))
	feature3ExperimentalVariance = np.var(sumYs)
	feature3PredictedVariance = 0
	print("Feature 3 Experimental Mean: " + str(feature3ExperimentalMean))
	print("Feature 3 Predicted Mean: " + str(feature3PredictedMean))
	print("Feature 3 Experimental Variance: " + str(feature3ExperimentalVariance))
	print("Feature 3 Predicted Variance: " + str(feature3PredictedVariance))
	######################################################################

	# Feature 4 Test 1
	######################################################################
	feature4Test1ExperimentalMean = np.mean(sumYs)
	feature4Test1PredictedMean = ((floatn - 1) * floatn) / ((floatn + 2) * (floatn + 1) ** 2)
	feature4Test1ExperimentalVariance = np.var(sumYs)
	feature4Test1PredictedVariance = (4 * floatn * (floatn - 1) * (
		2 * floatn ** 3 + 2 * floatn ** 2 - 3 * floatn + 3)) / (
	(floatn + 1) ** 4 * (floatn + 2) ** 2 * (floatn + 3) * (floatn + 4))
	print("Feature 4 Test 1 Experimental Mean: " + str(feature4Test1ExperimentalMean))
	print("Feature 4 Test 1 Predicted Mean: " + str(math.sqrt(feature4Test1PredictedMean)))
	print("Feature 4 Test 1 Experimental Variance: " + str(feature4Test1ExperimentalVariance))
	print("Feature 4 Test 1 Predicted Variance: " + str(feature4Test1PredictedVariance))
	######################################################################	

	# Feature 4 Test 2
	######################################################################
	prob = (1 - 1.0 / n) ** n
	feature4Test2ExperimentalMean = np.var(sumYs) = np.mean(bins)
	feature4Test2PredictedMean = n * prob
	feature4Test2ExperimentalVariance = np.var(bins)
	feature4Test2PredictedVariance = n * prob * (1 - prob)
	print("Feature 4 Test 2 Experimental Mean: " + str(feature4Test2ExperimentalMean))
	print("Feature 4 Test 2 Predicted Mean: " + str(feature4Test2PredictedMean))
	print("Feature 4 Test 2 Experimental Variance: " + str(feature4Test2ExperimentalVariance))
	print("Feature 4 Test 2 Predicted Variance: " + str(feature4Test2PredictedVariance))
	######################################################################

	# Feature 4 Test 3
	######################################################################
	t = 24.0 / ((n + 4) * (n + 3) * (n + 2) * (n + 1))

	feature4Test3ExperimentalMean = np.mean(uSqds)
	feature4Test3PredictedMean = 2.0 / ((n + 1) * (n + 2))
	feature4Test3ExperimentalVariance = np.var(uSqds)
	feature4Test3PredictedVariance = t - feat4Test3PredMean ** 2
	print("Feature 4 Test 3 Experimental Mean: " + str(feature4Test3ExperimentalMean))
	print("Feature 4 Test 3 Predicted Mean: " + str(feature4Test3PredictedMean))
	print("Feature 4 Test 3 Experimental Variance: " + str(feature4Test3ExperimentalVariance))
	print("Feature 4 Test 3 Predicted Variance: " + str(feature4Test3PredictedVariance))
	######################################################################

	if __name__ == "__main__":
		main()
