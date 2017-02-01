'''
	Erich Chu 
	K-means algorithm
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

# Point 1 will be test data and point 2 will be centroid data
def distanceBetweenTwoDataPoints(point1, point2):
	distance = 0.0
	# Skip over democratic party
	for data in range(1, len(point1._attributes)):
		distance += pow((point1._attributes[data] - point2._attributes[data]), 2)
	return math.sqrt(distance)

def assignDataPointToCluster(datapoint, centroidPoints):
	lowestDistance = sys.float_info.max
	closestCentroid = 0.
	for centroid in range(len(centroidPoints)):
		distance = distanceBetweenTwoDataPoints(datapoint, centroidPoints[centroid])
		if distance < lowestDistance:
			lowestDistance = distance
			closestCentroid = centroid
	return closestCentroid, lowestDistance


def makeClustersWithCentroidPoints(allDataPoints, centroidPoints, num_clusters):
	allClusters = [[] for i in range(num_clusters)]
	for dataPoint in allDataPoints:
		cluster_idx, lowestDistance = assignDataPointToCluster(dataPoint, centroidPoints)
		dataPoint._currentCluster = cluster_idx
		dataPoint._distanceFromCurrentCluster = lowestDistance
		allClusters[cluster_idx].append(dataPoint)
	return allClusters

def calculateNewCentroidPoints(allClusters, num_clusters):
	newCentroidPoints = [None for i in range(num_clusters)]
	for cluster in range(len(allClusters)):
		mean_values = [0.0 for x in range(len(allClusters[cluster][0]._attributes))]
		for point in allClusters[cluster]:
			# Skip over party
			for attribute in range(1, len(point._attributes)):
				mean_values[attribute] += (point._attributes[attribute] / len(allClusters[cluster]))
		newCentroidPoints[cluster] = DataPoint(mean_values, len(mean_values))
	return newCentroidPoints

def kmeans(test_data_points, num_clusters, stop_cutoff):
	initialCentroidPoints = random.sample(test_data_points, num_clusters)
	while(True):
		biggestShiftChange = sys.float_info.min
		allClusters = makeClustersWithCentroidPoints(test_data_points, initialCentroidPoints, num_clusters)
		newCentroidPoints = calculateNewCentroidPoints(allClusters, num_clusters)
		for idx in range(len(newCentroidPoints)):
			distance = distanceBetweenTwoDataPoints(newCentroidPoints[idx], initialCentroidPoints[idx])
			if distance > biggestShiftChange:
				biggestShiftChange = distance

		if biggestShiftChange < stop_cutoff:
			break
		else:
			initialCentroidPoints = newCentroidPoints
	print 'Final shift change : ' + str(biggestShiftChange)
	return allClusters

'''
Calculating Silhouette score to evaluate clustering
Formula (b-a) / max(a,b)
B is the distance from the nearest cluster
A is the distance from the centroid in current cluster
'''
def evaluateClusters(allClusters, num_of_all_datapoints):
	silhouette_score = 0.0
	for cluster1 in allClusters:
		for point1 in cluster1:
			distanceToClosestCluster = sys.float_info.max
			for cluster2 in allClusters:
				for point2 in cluster2:
					if point1._currentCluster != point2._currentCluster:
						distance = distanceBetweenTwoDataPoints(point1, point2)
						if distance < distanceToClosestCluster:
							distanceToClosestCluster = distance
			silhouette_score += (distanceToClosestCluster - point1._distanceFromCurrentCluster) / max(distanceToClosestCluster, point1._distanceFromCurrentCluster)
	return silhouette_score / num_of_all_datapoints



def main():
	raw_test_dataset = loadDataSets('votes-test.csv')
	testMinValues, testMaxValues = findMinAndMaxValues(raw_test_dataset)
	normalized_test_dataset = normalizeData(raw_test_dataset, testMinValues, testMaxValues)
	allDataPoints = importDataSetAsDataPoints(normalized_test_dataset)
	
	k_num_of_clusters = 3
	stop_cutoff = 0.009

	allClusters = kmeans(allDataPoints, k_num_of_clusters, stop_cutoff)
	print evaluateClusters(allClusters, len(allDataPoints))

main()

