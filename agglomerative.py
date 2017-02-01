'''
	Erich Chu 
	Agglomerative nesting
'''
import sys
import csv
import math
import operator
import random

def loadDataSets(testFile):
	test_dataset = []

	with open(testFile, 'rb') as csvfile:
		csvreader = csv.reader(csvfile)
		next(csvreader, None)
		for row in csvreader:
			test_dataset.append(row)

	return test_dataset

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

class Cluster:
	# Every cluster starts out as a single data point
	def __init__(self, point):
		self._points = [point]

	def addPointsFromCluster(self, cluster):
		for point in cluster._points:
			self._points.append(point)

	def distanceBetweenTwoDataPoints(self, point1, point2):
		distance = 0.0
		# Skip over democratic party
		for data in range(1, len(point1._attributes)):
			distance += pow((point1._attributes[data] - point2._attributes[data]), 2)
		return math.sqrt(distance)

	def evalDistanceFromCluster(self, cluster):
		shortest_distance = sys.float_info.max
		for this_point in self._points:
			for cluster_point in cluster._points:
				distance = self.distanceBetweenTwoDataPoints(this_point, cluster_point)
				if distance < shortest_distance:
					shortest_distance = distance
		return shortest_distance

class DataPoint:
	def __init__(self, datarow, datarow_length):
		self._attributes = [0.0 for i in range(datarow_length)]
		for data_idx in range(datarow_length):
			self._attributes[data_idx] = float(datarow[data_idx])
		self._length = datarow_length

def importDataSetAsDataPoints(dataset):
	allDataPoints = []
	for data in dataset:
		allDataPoints.append(DataPoint(data, len(data)))
	return allDataPoints

def convertAllDataPointsToClusters(allDataPoints):
	allClusters = []
	for dataPoint in allDataPoints:
		allClusters.append(Cluster(dataPoint))
	return allClusters

def agglomerativeCluster(allClusters):
	while(len(allClusters) != 1):
		# Just defaults
		shortest_cluster_distance = sys.float_info.max
		cluster_to_keep = allClusters[0]
		cluster_to_remove = allClusters[1]

		for cluster1 in range(len(allClusters)):
			for cluster2 in range(len(allClusters)):
				if allClusters[cluster1] != allClusters[cluster2]:
					distance = allClusters[cluster1].evalDistanceFromCluster(allClusters[cluster2])
					if distance < shortest_cluster_distance:
						cluster_to_keep = cluster1
						cluster_to_remove = cluster2
		allClusters[cluster_to_keep].addPointsFromCluster(allClusters[cluster_to_remove])
		allClusters.remove(allClusters[cluster_to_remove])
	return allClusters
			

def main():
	raw_test_dataset = loadDataSets('votes-test.csv')
	testMinValues, testMaxValues = findMinAndMaxValues(raw_test_dataset)
	normalized_test_dataset = normalizeData(raw_test_dataset, testMinValues, testMaxValues)
	allDataPoints = importDataSetAsDataPoints(normalized_test_dataset)
	allClusters = convertAllDataPointsToClusters(allDataPoints)

	# Super slow runtime
	agglomerativeClusterResult = agglomerativeCluster(allClusters)
	print agglomerativeClusterResult
	
main()

