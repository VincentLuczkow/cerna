# This file just implements the algorithms. It does not test them

import math
from functools import reduce
import scipy.stats


# Given a sequence and a motif, find all instances of the
# motif in the sequence. Return as a list.
def findBindingSites(sequence, motif):
	# Stores location of binding sites in the sequence.
	bindingSites = []
	w = len(motif)
	i = 0
	while i < len(sequence) - w + 1:
		if motif == sequence[i:i + w]:
			bindingSites.append(i)
		i += 1
	return bindingSites


# Finds the points where the motif overlaps itself.
# So take a prefix of length k, and suffix of length k, see
# if these 'fixes are equal. If they are, we
def motifOverlap(motif):
	Im = {}
	r = 2
	w = len(motif)
	while r <= w:
		if (motif[:w - r + 1] == motif[r - 1:]):
			Im[r - 1] = 1
		else:
			Im[r - 1] = 0
		r += 1
	print(Im)
	return Im


# Calculates the probability the motif will appear. Can provide custom probabilities
# for each letter. This is the equivalent of theta in the notes.
def probOfWord(motif, baseProbs):
	p = 1
	for x in motif:
		p = p * baseProbs[x]
	return p


# The first feature is calculated from the number of binding sites.
def firstFeature(sequence, motif, baseProbs={'a': 0.25, 'c': 0.25, 'g': 0.25, 't': 0.25, 'u': 0.25}):
	# print("First Feature")
	# The number of times the motif appears in the sequence.
	numberOfSites = sequence.count(motif)

	# Theta from the notes.
	theta = probOfWord(motif, baseProbs)

	k = len(sequence) - len(motif) + 1

	# The expected value of the number of copies of the motif in the sequence.
	expectedValue = theta * k

	Im = motifOverlap(motif)

	# Holds the covariance
	covariance = 0

	r = 2
	# Build up covariance based on overlapping
	print("Sequence length " + str(len(sequence)))
	while r <= len(motif):
		# print("r is " + str(r))
		# print("Im r is " + str(Im[r-1]))
		covariance += (k + 1 - r) * Im[r - 1] * probOfWord(motif[-r:], baseProbs) * theta
		r += 1

	covariance *= 2

	# Subtract E(x) * E(y)
	covariance -= theta * theta

	variance = k * theta * (1 - theta) + covariance

	return float(numberOfSites - expectedValue) / (math.sqrt(variance))


# This returns the probability that the span of binding sites is less than baseSpan.
def secondFeature(sequence, motif, baseSpan):
	# Location of last occurring copy of motif.
	last = sequence.rfind(motif)
	# Location of first occurring copy of motif.
	first = sequence.find(motif)
	span = last - first
	return last * (1 - baseSpan) * (baseSpan ** (last + 1)) + baseSpan ** last


# Measure of the degree of clustering of binding sites.
def thirdFeature(sequence, motif):
	# print("Third Feature")
	# Locations of all the binding sites.
	sites = findBindingSites(sequence, motif)
	n = len(sites)
	# List of distances between each site. distances[i] is the distance between binding site i+2 and i+1.
	# (Offset because of the way python indexes lists)
	distances = list(map(lambda x, y: float(x - y) / len(sequence), sites[1:], sites[:-1]))
	product = reduce(lambda x, y: x * y, distances, 1)

	firstSum = 0
	for i in range(1, n):
		firstProduct = float(product) / distances[i - 1]
		firstSum += (2 * distances[i - 1] - distances[i - 1] ** 2) * float(firstProduct)
	firstSum /= 2
	return math.factorial(n) * (firstSum - product)


# Measures the evenness of the distribution.
def fourthFeature(sequence, motif, distance):
	# print("Fourth Feature")
	sites = findBindingSites(sequence, motif)
	# Number of binding sites.
	n = len(sites)
	expectedValue = float(n) / ((n + 1) ** 2 * (n + 2))

	variance = float((8 * (n ** 5) + 8 * (n ** 4) - 3 * (n ** 3) + 12 * (n ** 2))) / (
		(n + 1) ** 4 * (n + 2) ** 2 * (n + 3) * (n + 4))

	# print(variance)

	stdev = math.sqrt(variance)

	a = (distance - float(((n - 1) * n)) / ((n + 1) ** 2 * (n + 2))) / stdev

	feature = (scipy.stats.norm(0, 1).cdf(a))
	# print("Probability is: " + str(feature))
	print(feature)
	return feature


def main():
	firstFeature("acacac", "acac")
	secondFeature("acacac", "acac", 50)
	thirdFeature("acacbacbbacacacac", "ac")
	fourthFeature("asddfasdfasgsfasdfasdf", "df", 0.1)


if __name__ == "__main__":
	main()
