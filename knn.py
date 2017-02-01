'''
	Erich Chu 
	K nearest neighbor
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

def distanceBetweenTwoDataPoints(data1, data2):
	distance = 0
	# Skip over democratic party
	for data in range(1, 10):
		distance += pow((data1[data] - data2[data]), 2)
	return math.sqrt(distance)

def retrieveXNeighbors(test_data, training_dataset, x):
	allNeighbors = []
	for data in range(len(training_dataset)):
		theDistance = distanceBetweenTwoDataPoints(test_data, training_dataset[data])
		allNeighbors.append((training_dataset[data], theDistance))
	# Sorts to get closest neighbors by pairs (neighbor, distance)
	allNeighbors.sort(key=operator.itemgetter(1))

	# Retrieve x neighbors
	neighbors = []
	for currIdx in range(x):
		neighbors.append(allNeighbors[currIdx][0])
	return neighbors

def evalParty(neighbors):
	democratic = 0
	republican = 0
	for neighbor in neighbors:
		if int(neighbor[0]) == 0:
			republican += 1
		else:
			democratic += 1
	return 0 if republican > democratic else 1

def predictParty(test_dataset, train_dataset):
	predictedParties = []
	for test_data in test_dataset:
		neighbors = retrieveXNeighbors(test_data, train_dataset, 3)
		concludedParty = evalParty(neighbors)
		predictedParties.append(concludedParty)
	return predictedParties

def getAccuracy(predictionResults, test_dataset):
	correct = 0
	for test_data in range(len(test_dataset)):
		if int(test_dataset[test_data][0]) == predictionResults[test_data]:
			correct += 1
	return (float(correct) / len(test_dataset)) * 100


def main():
	raw_train_dataset, raw_test_dataset = loadDataSets('votes-train.csv', 'votes-test.csv')
	trainMinValues, trainMaxValues = findMinAndMaxValues(raw_train_dataset)
	testMinValues, testMaxValues = findMinAndMaxValues(raw_test_dataset)
	train_dataset = normalizeData(raw_train_dataset, trainMinValues, trainMaxValues)
	test_dataset = normalizeData(raw_test_dataset, testMinValues, testMaxValues)

	predictionResults = predictParty(test_dataset, train_dataset)
	print getAccuracy(predictionResults, test_dataset)	

main()

