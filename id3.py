'''
	Erich Chu 
	ID3 Learning Algorithm
'''
import sys
import csv
import math
import operator

def loadDataSets(trainingFile, testFile):
	train_dataset = []
	test_dataset = []
	# Training data
	with open(trainingFile, 'rb') as csvfile:
		csvreader = csv.reader(csvfile)
		next(csvreader, None)
		for row in csvreader:
			train_dataset.append(row)

	with open(testFile, 'rb') as csvfile:
		csvreader = csv.reader(csvfile)
		next(csvreader, None)
		for row in csvreader:
			test_dataset.append(row)

	return train_dataset, test_dataset

def getBiggerVal(value1, value2):
	return float(value1) if float(value1) > float(value2) else float(value2)

def getSmallerVal(value1, value2):
	return float(value1) if float(value1) < float(value2) else float(value2)

def findMinAndMaxValues(dataset):
	max_population = max_population_change = max_age65plus = max_black = max_hispanic = max_edu_bachelors = max_income = max_poverty = max_density = sys.float_info.min
	min_population = min_population_change = min_age65plus = min_black = min_hispanic = min_edu_bachelors = min_income = min_poverty = min_density = sys.float_info.max

	for data in dataset:
		max_population = getBiggerVal(data[1], max_population)
		max_population_change = getBiggerVal(data[2], max_population_change)
		max_age65plus = getBiggerVal(data[3], max_age65plus)
		max_black = getBiggerVal(data[4], max_black)
		max_hispanic = getBiggerVal(data[5], max_hispanic)
		max_edu_bachelors = getBiggerVal(data[6], max_edu_bachelors)
		max_income = getBiggerVal(data[7], max_income)
		max_poverty = getBiggerVal(data[8], max_poverty)
		max_density = getBiggerVal(data[9], max_density)
		min_population = getSmallerVal(data[1], min_population)
		min_population_change = getSmallerVal(data[2], min_population_change)
		min_age65plus = getSmallerVal(data[3], min_age65plus)
		min_black = getSmallerVal(data[4], min_black)
		min_hispanic = getSmallerVal(data[5], min_hispanic)
		min_edu_bachelors = getSmallerVal(data[6], min_edu_bachelors)
		min_income = getSmallerVal(data[7], min_income)
		min_poverty = getSmallerVal(data[8], min_poverty)
		min_density = getSmallerVal(data[9], min_density)
		
	minValues = []
	maxValues = []
	maxValues += [max_population, max_population_change, max_age65plus, max_black, max_hispanic, max_edu_bachelors, max_income, max_poverty, max_density]
	minValues += [min_population, min_population_change, min_age65plus, min_black, min_hispanic, min_edu_bachelors, min_income, min_poverty, min_density]
	return minValues, maxValues

def normalizeData(dataset, minValues, maxValues):
	for datarow in dataset:
		for data in range(len(datarow)):
			if data == 0:
				continue
			else:
				datarow[data] = (float(datarow[data]) - minValues[data-1])/ (maxValues[data-1] - minValues[data-1])
	return dataset

class DataPoint:
	def __init__(self, datarow, datarow_length):
		self._attributes = [0.0 for i in range(datarow_length)]
		for data_idx in range(datarow_length):
			self._attributes[data_idx] = float(datarow[data_idx])
		self._length = datarow_length
		self._currentCluster = None
		self._distanceFromCurrentCluster = None

def importDataSetAsDataPoints(dataset):
	allDataPoints = []
	for data in dataset:
		allDataPoints.append(DataPoint(data, len(data)))
	return allDataPoints


def getAccuracy(predictionResults, test_dataset):
	correct = 0
	for test_data in range(len(test_dataset)):
		if int(test_dataset[test_data][0]) == predictionResults[test_data]:
			correct += 1
	return (float(correct) / len(test_dataset)) * 100

def calculateEntropy(indexOfAttribute, allDataPoints):
	entropyTotal = 0.0
	freqDict = {}
	for dataPoint in allDataPoints:
		dataValue = dataPoint._attributes[indexOfAttribute]
		if dataValue in freqDict:
			freqDict[dataValue] += 1.0
		else:
			freqDict[dataValue] = 1.0

	for freq in freqDict:
		entropyTotal += ((-1 * freqDict[freq] / len(allDataPoints)) * math.log(freqDict[freq]/len(allDataPoints), 2))
	return entropyTotal

# My way of doing information gain since the lowest entropy is needed to result in biggest gain
def getAllEntropies(allDataPoints):
	allAttributeEntropies = []
	for indexOfAttribute in range(1, len(allDataPoints[0]._attributes)):
		allAttributeEntropies.append((calculateEntropy(indexOfAttribute, allDataPoints), indexOfAttribute))
	return allAttributeEntropies

def getMaxVal(allDataPoints, attrIdx):
	maxVal = sys.float_info.min
	for dataPoint in allDataPoints:
		if dataPoint._attributes[attrIdx] > maxVal:
			maxVal = dataPoint._attributes[attrIdx]
	return maxVal

def getMinVal(allDataPoints, attrIdx):
	minVal = sys.float_info.max
	for dataPoint in allDataPoints:
		if dataPoint._attributes[attrIdx] < minVal:
			minVal = dataPoint._attributes[attrIdx]
	return minVal
	
def calculateQuartile(allDataPoints, attrIdx):
	maxVal = getMaxVal(allDataPoints, attrIdx)
	minVal = getMinVal(allDataPoints, attrIdx)
	rangeVal = maxVal - minVal
	allVals = []
	for dataPoint in allDataPoints:
		allVals.append(dataPoint._attributes[attrIdx])
	allVals.sort()
	firstq = allVals[len(allDataPoints)//4]
	secondq = allVals[len(allDataPoints)//2]
	thirdq = allVals[(3*len(allDataPoints))//4]
	return firstq, secondq, thirdq

def evalQuartile(allDataPoints, idx, q_1 = None, q_2 = None):
	rep = 0
	dem = 0

	for dataPoint in allDataPoints:
		attrVal = dataPoint._attributes[idx]
		partyVal = dataPoint._attributes[0]
		if q_1 == None:
			if attrVal < q_2:
				if partyVal == 0.0:
					rep += 1
				else:
					dem += 1
		elif q_2 == None:
			if attrVal >= q_1:
				if partyVal == 0.0:
					rep += 1
				else:
					dem += 1
		elif attrVal >= q_1 and attrVal < q_2:
			if partyVal == 0.0:
				rep += 1
			else:
				dem += 1
	# pure
	if rep == 0:
		return 1
	elif dem == 0:
		return 0
	else:
		# indecisive
		return -1

# q_1 is beginning
# q_2 is end
# compares [q_1, q_2)
def getDataPointsWithinQuartile(allDataPoints, attrIdx, q_1 = None, q_2 = None):
	newDataPoints = []
	for dataPoint in allDataPoints:
		attrVal = dataPoint._attributes[attrIdx]
		if q_1 == None:
			if attrVal < q_2:
				newDataPoints.append(dataPoint)
		elif q_2 == None:
			if attrVal >= q_1:
				newDataPoints.append(dataPoint)
		elif attrVal >= q_1 and attrVal < q_2:
			newDataPoints.append(dataPoint)
	return newDataPoints

def getAttributes(attributeWithEntropies):
	attributeWithEntropies.pop(0)
	allAttributes = []
	for attr in attributeWithEntropies:
		allAttributes.append(attr[1])
	return allAttributes

def filterEntropies(allAttributes, entropyList):
	new_entropy_list = []
	for entropy in entropyList:
		if entropy[1] in allAttributes:
			new_entropy_list.append(entropy)
	return new_entropy_list

def getMajority(allDataPoints):
	rep = 0
	dem = 0

	for dataPoint in allDataPoints:
		if dataPoint._attributes[0] == 0:
			rep += 1
		else:
			dem += 1
	return 0 if rep >= dem else 1

def id3Tree(allTrainDataPoints, allAttributes, attrIdx):
	tree = {attrIdx:{}}

	# Update attributes list
	remove_idx = -1
	for allAttributesIdx in range(len(allAttributes)):
		if allAttributes[allAttributesIdx] == attrIdx:
			remove_idx = allAttributesIdx

	newAllAttributes = list(allAttributes)
	newAllAttributes.pop(remove_idx)

	finished = (len(newAllAttributes) == 0)

	# Get quartiles for splitting to subsets
	firstq, secondq, thirdq = calculateQuartile(allTrainDataPoints, attrIdx)

	# Evaluate party of each subset
	party_underq1 = evalQuartile(allTrainDataPoints, attrIdx, q_1 = firstq)
	party_q1_q2 = evalQuartile(allTrainDataPoints, attrIdx, firstq, secondq)
	party_q2_q3 = evalQuartile(allTrainDataPoints, attrIdx, secondq, thirdq)
	party_above_q3 = evalQuartile(allTrainDataPoints, attrIdx, q_2 = thirdq)

	# See if tree expands further
	# If indecisive result, divide into new subsets and lowest entropy valued attribute will return
	# biggest information gain since E(1) - E(2)
	if party_underq1 == -1:
		q1 = getDataPointsWithinQuartile(allTrainDataPoints, attrIdx, q_2 = firstq)
		if finished or len(q1) == 0:
			# Append to tree default value
			tree[attrIdx][0] = getMajority(q1)
		else:
			all_entropies_q1 = getAllEntropies(q1)
			actual_entropies_left = filterEntropies(newAllAttributes, all_entropies_q1)
			actual_entropies_left.sort(key=operator.itemgetter(0))
			bestAttrib = actual_entropies_left[0][1]
			tree[attrIdx][0] = id3Tree(q1, newAllAttributes, bestAttrib)
	else:
		tree[attrIdx][0] = party_underq1

	if party_q1_q2 == -1:
		q2 = getDataPointsWithinQuartile(allTrainDataPoints, attrIdx, q_1 = firstq, q_2 = secondq)
		if finished or len(q2) == 0:
			tree[attrIdx][1] =  getMajority(q2)
		else:
			all_entropies_q2 = getAllEntropies(q2)
			actual_entropies_left = filterEntropies(newAllAttributes, all_entropies_q2)
			actual_entropies_left.sort(key=operator.itemgetter(0))
			bestAttrib = actual_entropies_left[0][1]
			tree[attrIdx][1] = id3Tree(q2, newAllAttributes, bestAttrib)
	else:
		tree[attrIdx][1] = party_q1_q2

	if party_q2_q3 == -1:
		q3 = getDataPointsWithinQuartile(allTrainDataPoints, attrIdx, q_1 = secondq, q_2 = thirdq)
		if finished or len(q3) == 0:
			tree[attrIdx][2] = getMajority(q3)
		else:
			all_entropies_q3 = getAllEntropies(q3)
			actual_entropies_left = filterEntropies(newAllAttributes, all_entropies_q3)
			actual_entropies_left.sort(key=operator.itemgetter(0))
			bestAttrib = actual_entropies_left[0][1]
			tree[attrIdx][2] = id3Tree(q3, newAllAttributes, bestAttrib)
	else:
		tree[attrIdx][2] = party_q2_q3

	if party_above_q3 == -1:
		q4 = getDataPointsWithinQuartile(allTrainDataPoints, attrIdx, q_1 = thirdq)
		if finished or len(q4) == 0:
			tree[attrIdx][3] = getMajority(q4)
		else:
			all_entropies_q4 = getAllEntropies(q4)
			actual_entropies_left = filterEntropies(newAllAttributes, all_entropies_q4)
			actual_entropies_left.sort(key=operator.itemgetter(0))
			bestAttrib = actual_entropies_left[0][1]
			tree[attrIdx][3] = id3Tree(q4, newAllAttributes, bestAttrib)
	else:
		tree[attrIdx][3] = party_above_q3
	return tree

def predictDataPoint(tree, allDataPoints, dataPoint, attrIdx):
	firstq, secondq, thirdq = calculateQuartile(allDataPoints, attrIdx)
	q_val = -1
	attrVal = dataPoint._attributes[attrIdx]
	if attrVal < firstq:
		q_val = 0
	elif attrVal >= firstq and attrVal < secondq:
		q_val = 1
	elif attrVal >= secondq and attrVal < thirdq:
		q_val = 2
	elif attrVal >= thirdq:
		q_val = 3
	if type(tree[attrIdx][q_val]) is dict:
		return predictDataPoint(tree[attrIdx][q_val], allDataPoints, dataPoint, tree[attrIdx][q_val].keys()[0])
	else:
		return tree[attrIdx][q_val]

def predictWithTree(allTestDataPoints, tree, attrIdx):
	predictionResults = []
	for dataPoint in allTestDataPoints:
		predictionResults.append(predictDataPoint(tree, allTestDataPoints, dataPoint, attrIdx))
	return predictionResults

def main():
	raw_train_dataset, raw_test_dataset = loadDataSets('votes-train.csv', 'votes-test.csv')
	trainMinValues, trainMaxValues = findMinAndMaxValues(raw_train_dataset)
	testMinValues, testMaxValues = findMinAndMaxValues(raw_train_dataset)
	allTrainDataPoints = importDataSetAsDataPoints(raw_train_dataset)
	allTestDataPoints = importDataSetAsDataPoints(raw_test_dataset)
	allAttributeEntropies = getAllEntropies(allTrainDataPoints)

	# All entropies are now sorted with the attribute index from highest to lowest
	allAttributeEntropies.sort(key=operator.itemgetter(0), reverse = True)
	first_attribute = allAttributeEntropies[0][1]

	allAttributes = []
	for attr in allAttributeEntropies:
		allAttributes.append(attr[1])

	tree = id3Tree(allTrainDataPoints, allAttributes, first_attribute)
	#print tree[1][0][4][0][5][0][2][0]
	predictionResults = predictWithTree(allTestDataPoints, tree, first_attribute)
	print getAccuracy(predictionResults, raw_test_dataset)

main()

