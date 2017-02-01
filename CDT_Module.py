import sys
import operator
from math import pow, sqrt, log
from random import randint
from csv import reader
from decimal import Decimal as dec
from time import time
from copy import deepcopy


# Load data set from file name
def load_data_set(file):
    dataset = []
    with open(file, 'rb') as csvfile:
        csvreader = reader(csvfile)
        next(csvreader, None)
        for row in csvreader:
            dataset.append(row)
    return dataset


# Merges two data sets into one
# @param d1 : dataset #1
# @param d2 : dataset #2
def merge_two_data_sets(d1, d2):
    return list(d1 + d2)


# Splits one data set into two subsets
# Returns two subsets of the original data
# @param dataset : the data that will be split
# @param split_percent : the percent to split the data by
def split_data_into_test_and_train(dataset, split_percent = .7):
    total = len(dataset)
    train_indices = []    
    while len(train_indices) < int(total * split_percent):
        j = randint(0,total)
        if j not in train_indices:
            train_indices.append(j)
    train_set = []
    test_set = []
    for idx, row in enumerate(dataset):
        if idx in train_indices:
            train_set.append(row)
        else:
            test_set.append(row)
    return (train_set, test_set)


# Note: First index is assumed to hold outcome value which will remain irrelevant
# Returns a list of means for each attribute value of the dataset
# @param dataset : the dataset to calculate means for
def calculate_all_means(dataset):
    # Creating list of all mean values for each column idx aka attribute
    num_cols = len(dataset[0])
    all_means = [0.0 for i in range(num_cols)]

    for row in dataset:
        for col_idx, col in enumerate(row):
            if col_idx == 0:
                continue
            else:
                all_means[col_idx] += float(col)

    num_of_rows = len(dataset)
    all_means = [num / num_of_rows for num in all_means]
    return all_means


# Returns a list of the standard deviations for each attribute value
# @param dataset : the dataset to calculate standard deviations for
# @param all_attribute_means : a list of all the means for each attribute of a row
def calculate_all_col_std_deviations(dataset, all_attribute_means):
    num_cols = len(dataset[0])
    all_std_devs = [0.0 for i in range(num_cols)]

    for row in dataset:
        for col_idx, col in enumerate(row):
            if col_idx == 0:
                continue
            else:
                all_std_devs[col_idx] += pow(float(col) - all_attribute_means[col_idx],2)

    num_of_rows = len(dataset)
    all_std_devs = [sqrt((num / num_of_rows)) for num in all_std_devs]
    return all_std_devs


# Calculates the z score of a value using the formula
# @param val : the initial value to turn into a z score
# @param mean : the mean being used for the formula
# @param stdDev : the standard deviation being used for the formula
def calculate_ZScore(val, mean, std_dev):
    return (val - mean) / std_dev


# Converts an entire dataset's values to z scores
# @param dataset : the dataset to convert
# @param all_attribute_means : a list of all the means for each attribute in a row
# @param all_std_deviations : a list of all the standard deviations for each attribute in a row
def convert_data_to_ZScores(dataset, all_attribute_means, all_std_deviations):
    num_cols = len(dataset[0])
    for row_idx,row in enumerate(dataset):
        for col_idx in range(num_cols):
            # Skips over the first column - assuming it's the outcome
            if col_idx == 0:
                continue
            else:
                # Convert data into z score
                zScore = calculate_ZScore(float(row[col_idx]), all_attribute_means[col_idx], all_std_deviations[col_idx])
                dataset[row_idx][col_idx] = zScore
    return dataset


# Finds all means and standard deviations for a dataset
# Returns a list of means and standard deviations
# @param dataset : the dataset to find all the means and std deviations for
def find_mean_and_std_dev(dataset):
    all_attribute_means = calculate_all_means(dataset)
    all_std_deviations = calculate_all_col_std_deviations(dataset, all_attribute_means)
    return all_attribute_means, all_std_deviations


# Calculates entropy of an attribute in a dataset
# @param idx_of_attribute : the index of an attribute
# @param data_set : the dataset being used for finding entropy
def calculate_entropy(idx_of_attribute, data_set):
    entropy_total = 0.0
    freq_dict = {}
    for row_idx, row in enumerate(data_set):
        colVal = data_set[row_idx][idx_of_attribute]
        if colVal in freq_dict:
            freq_dict[colVal] += 1.0
        else:
            freq_dict[colVal] = 1.0

    for freq in freq_dict:
        entropy_total += ((-1 * freq_dict[freq] / len(data_set)) * log(freq_dict[freq]/len(data_set), 2))
    return entropy_total


# Returns a list of tuples modeled (entropy_value, index_of_attribute)
# Gets all the entropies for a dataset
# @param dataset : the dataset to retrieve all entropies for
# @param exclusions : exclude retrieving entropies for certain attributes from the dataset
def get_all_entropies(dataset, exclusions):
    all_entropies = []
    for idx_of_attribute in range(1, len(dataset[0])):
        if idx_of_attribute in exclusions:
            continue
        else:
            all_entropies.append((calculate_entropy(idx_of_attribute, dataset), idx_of_attribute))
    return all_entropies


# Note : Assumes outcome result is in the first row of the dataset
# Returns the popular vote in a dataset. In a tie, the first option wins
# @param dataset : dataset being used to determine majority outcome
def get_majority_of_outcome(dataset):
    outcome1 = dataset[0][0]
    outcome2 = None
    outcome1count = 0
    outcome2count = 0
    for row in dataset:
        if row[0] == outcome1:
            outcome1count += 1
        else:
            outcome2 = row[0]
            outcome2count += 1
    return outcome1 if (outcome1count > outcome2count) else outcome2


# Debugger for `apply_bin_range_and_vals`
# Prints number of data rows that were sorted into this bin
# @param dataset : the dataset being used
# @param bin_val : the bin value that is being used for counting data rows
# @param col : attribute index for a dataset that is being used 
def num_per_category(dataset, bin_val, col):
    binvals = []
    for i, val in enumerate(bin_val):
        binvals.append(0)
        for row in dataset:
            if row[col]==val:
                binvals[i] += 1
    print binvals 


# Checks distribution of values into bins for `apply_bin_range_and_vals`
# @param dataset : the dataset being used
# @param bin_val : the bin values being used
def check_distribution_of_bin_vals(dataset, bin_val):
    num_cols = len(dataset[0])
    for col_idx, col in enumerate(num_cols):
        if col_idx == 0:
            continue
        else:
            num_per_category(dataset, bin_val, 1)


# Splits all data and assigns them into specified ranges identified by bin_val
# Skips first index which is assumed to be outcome value
# @param data: dataset being converted
# @param bin_range: example - [-1.0, -0.5, -0.1, 0.1, 0.5, 1.0]
# @param bin_val: example - ['0', '1', '2', '3', '4', '5', '6']
def curate_categories(data, bin_range, bin_val):   
    assignPosition = -1 
    for row_idx, row in enumerate(data):
        for col_idx, col in enumerate(row):
            if col_idx == 0:
                continue
            else:
                for i, val in enumerate(bin_range):
                    if i == 0:
                        if col < val:
                            assignPosition = i
                    elif i == (len(bin_range) - 1):
                        if col >= val:
                            assignPosition = i
                    else:
                        if col >= bin_range[i-1] and col < val:
                            assignPosition = i
                    data[row_idx][col_idx] = bin_val[assignPosition]
    return data


# Creates bin ranges to sort all data into. Then, convert all data into
# a value that indicates which bin value the data is in. Returns a dataset
# that is sorted into the bins.
# @param dataset : the dataset being sorted into the appropriate bin ranges
# @param sys.argv (command line input): requires command line input to be at least two numbers ranging from largest to smallest
#        e.g. - command line call: python ai_final.py 1 0.5
#        1 and 0.5 turn into bin_range [-1, -0.5, 0.5, 1] and bin_val ['0', '1', '2', '3', '4'] 
def apply_bin_range_and_vals(dataset):
    args = []
    if len(sys.argv) == 1:
        args = [1.036433, 0.67449, 0.385319, 0.125661]
    for arg in range(1, len(sys.argv)):
        args.append(sys.argv[arg])
    bin_range = []
    for idx, item in enumerate(args):
        bin_range.append(-1 * float(item)) 
    for idx, item in enumerate(reversed(args)):
        bin_range.append(float(item))
    bin_val = []
    for idx, val in enumerate(bin_range):
        bin_val.append(str(idx))
    bin_val.append(str(len(bin_range)))
    dataset = curate_categories(dataset, bin_range, bin_val)
    # Debugger.check_distribution_of_bin_vals(dataset, bin_val)
    return dataset


# CDT Algorithm for determining best result for a dataset
# Allows for using entropy. If not used, cdt iterates from left to right of column values
# @param train_dataset : training dataset used for predicting test rows 
#                        - will contain previous subset after first iteration
# @param test_row : a row from the test dataset
# @param attrib_idx : column/attribute index of a row
# @param prev_idxs : a list of previous indexes that the CDT has iterated over
# @param with_entropy : easy way to indicate whether to use entropy
def cdt(train_dataset, test_row, attrib_idx, prev_idxs, with_entropy):
    num_cols = len(test_row)
    result = train_dataset[0][0]
    pure = 1
    # Determine if cdt has found a pure result by checking
    # if the outcome values are all the same
    for row_idx, row in enumerate(train_dataset):
        if pure == 0:
            break
        # Pure result is not found
        if (result != train_dataset[row_idx][0]):
            pure = 0
    # Pure result is found
    if pure == 1:
        return result
    else:
        # For running CDT without entropy, continue running if this is a valid attribute index
        if attrib_idx < num_cols:
            # Create a new subset from the dataset where values from the same attribute index are the same
            data_subset = []
            for idx, data in enumerate(train_dataset):
                if train_dataset[idx][attrib_idx] == test_row[attrib_idx]:
                    data_subset.append(train_dataset[idx])

            # If a subset exists, find a pure value with another attribute
            if len(data_subset) > 0:
                # If using entropy, determine best attribute using information gain formula
                if with_entropy == True:
                    prev_idxs.append(attrib_idx)
                    all_entropies = get_all_entropies(train_dataset, prev_idxs)
                    all_entropies.sort(key=operator.itemgetter(0))

                    # If this isn't the last attribute being iterated over
                    if all_entropies:
                        next_best_attrib = all_entropies[0][1]
                        result = cdt(data_subset, test_row, next_best_attrib, prev_idxs, with_entropy)
                    # Determine outcome by the popular vote in the remaining dataset
                    else:
                        result = get_majority_of_outcome(train_dataset)
                # If not using entropy, go to the next attribute to the right
                else:
                    result = cdt(data_subset, test_row, attrib_idx+1, prev_idxs, with_entropy)
            # If no pure result can be found, evaluate outcome by using the popular vote
            else:
                result = get_majority_of_outcome(train_dataset)
        # For running CDT without entropy, if no pure values are found and all attributes have been iterated over
        else:
            result = get_majority_of_outcome(train_dataset)
    # Finally, return result
    return result 


# Run CDT algorithm on the test dataset using the train dataset
# Returns a tuple of number of correct predictions and number of wrong predictions
# @param train_data : train dataset being used to determine test dataset outcomes
# @param test_data : test dataset that will have its outcomes evaluated
# @param include_entropy : easy way to indicate if entropy should be used to determine next attribute
def run_cdt(train_data, test_data, include_entropy=False):
    correct = 0
    incorrect = 0
    beginning_idx = 1
    prev_idxs = []
    # Easy way of including entropy to determine next best attribute if tree hits a non pure
    if include_entropy == True:
        allEntropies = get_all_entropies(train_data, prev_idxs)
        # Initially, the entropy value we want is the one that is the highest
        allEntropies.sort(key=operator.itemgetter(0), reverse = True)
        beginning_idx = allEntropies[0][1]

    # Make prediction for each row in the test dataset
    for row in test_data:
        prediction = cdt(train_data, row, beginning_idx, prev_idxs, with_entropy=include_entropy)
        # print "Prediction: %s\t Answer: %s" % (prediction, row[0])        #for debugging purposes to manually check predictions
        if prediction == row[0]:
            correct+=1
        else:
            incorrect +=1
    return (correct, incorrect)


# Formats and prints out the number of correct and incorrect predictions
def check_results(correct, incorrect):
    print "Predictions were %.2f%% correct (%d/%d)" % (float(correct)/(incorrect + correct) * 100, correct, incorrect + correct)    

# Create a list of values of zscores to iterate over
# @param max_val : maximum value that the highest zscore can be
# @param increment : amount to increment by
def find_highest_lowest_zscores_to_iterate(max_val, increment=0.05):
    low = []
    high = []
    init_max = max_val
    while max_val > 0:
        high.append(max_val)
        if max_val != init_max:
            low.append(max_val)
        max_val = round(dec(max_val) - dec(increment), 4)
    return high, low


# Find Optimal Bin values to evaluate dataset
# @param full_raw_dataset : dataset to be evaluated for the best optimal bin range values
# @param zscore_max_dist : max value a zscore can be
# @param increment : amount to increment by
# @param include_entropy : easy way to indicate whether to use entropy or not
def find_optimal_bin_values(full_raw_dataset, zscore_max_dist = 1.65, increment = 0.1, include_entropy=False): 
    data_mean, data_std_dev = find_mean_and_std_dev(full_raw_dataset)
    zscore_data = convert_data_to_ZScores(full_raw_dataset, data_mean, data_std_dev)
    train_dataset, test_dataset = split_data_into_test_and_train(zscore_data)
    iterations = 0
    high, low = find_highest_lowest_zscores_to_iterate(zscore_max_dist, increment)
    best_correct = best_incorrect = best_percent = best_lv = best_hv = 0
    # hv = high value
    # lv = low value
    for hv in high:
        for lv in low:
            if lv < hv and hv > 0:
                sys.argv = ['', hv, lv]
                train = deepcopy(train_dataset)
                test = deepcopy(test_dataset)
                train_bins = apply_bin_range_and_vals(train)
                test_bins = apply_bin_range_and_vals(test)
                correct, incorrect = run_cdt(train_bins, test_bins, include_entropy)
                iterations += 1
                percent_correct = (float(correct) / (correct + incorrect))
                if percent_correct > best_percent:
                    best_percent = percent_correct
                    best_correct = correct
                    best_incorrect = incorrect
                    best_lv = lv
                    best_hv = hv
                # Print results realtime
                print percent_correct, best_percent, sys.argv
    return best_correct, best_incorrect, best_hv, best_lv, iterations


# Formatted print for the results of finding an optimal bin value range and its accuracy
# @params - all the stats for formatting
def optimal_bin_value_results(best_correct, best_incorrect, best_hv, best_lv, start_time, iterations):
    print("---------- Finished %d iterations in %s seconds ----------" % (iterations, time() - start_time))
    print "Optimal bin range = [ %s, %s ]" % (best_hv, best_lv)
    print "Best accuracy:"
    check_results(best_correct, best_incorrect)   


# Find the average accuracy of a dataset over a number of iterations
# @param full_data : the dataset being used to split and predict
# @param iterations : number of iterations to run CDT
def average_accuracy_of_cdt(full_data, iterations = 20):
    best_correct = best_total = avg_correct = avg_total = 0
    data_mean, data_std_dev = find_mean_and_std_dev(full_data)
    data = convert_data_to_ZScores(full_data, data_mean, data_std_dev)
    prepared_data = apply_bin_range_and_vals(data)
    for i in range(iterations):
        # Split the data into train/test
        train_bins, test_bins = split_data_into_test_and_train(list(prepared_data))
        correct, incorrect = run_cdt(train_bins, test_bins)
        if correct > best_correct:
            best_total = correct + incorrect
            best_correct = correct
        avg_correct += correct
        avg_total += correct + incorrect
    return best_correct, best_total - best_correct, avg_correct, avg_total - avg_correct


# Formatted print for the results of finding the average accuracy of cdt
# @params - all the stats for formatting
def average_cdt_accuracy_results(best_correct, best_incorrect, avg_correct, avg_incorrect, start_time, iterations):
    print("---------- Finished %d iterations in %s seconds ----------" % (iterations, time() - start_time))
    print "On average:"
    check_results(avg_correct, avg_incorrect)    
    print "Best:"
    check_results(best_correct, best_incorrect)   


def main():
    start_time = time()

    # Load dataset
    raw_train_dataset = load_data_set("votes-train.csv")
    raw_test_dataset = load_data_set("votes-test.csv")
    # Merge voting data from class  
    full_vote_data = merge_two_data_sets(raw_train_dataset, raw_test_dataset)    

    # Second optional dataset for testing how cdt algo works on other datasets
    full_sonar_data = load_data_set("sonar_train.csv")


    ##### -------- Finding Optimal Bin Range ------- #####
    ''' find the optimal bin range for a train/test set'''    
    z_score_max_dist = 1.9
    increment_val = 0.1
    #full_data = list(full_vote_data)
    full_data = list(full_sonar_data)
    best_correct, best_incorrect, best_hv, best_lv, iterations = find_optimal_bin_values(full_data, z_score_max_dist, increment_val, include_entropy=True) 
    optimal_bin_value_results(best_correct, best_incorrect, best_hv, best_lv, start_time, iterations)


    ##### -------- Determining accuracy of CDT Algorithm ------- #####
    '''Run the cdt n times, each time separating the data into a new train and test, find the average of CDT performance'''
    iterations = 20
    best_correct, best_incorrect, avg_correct, avg_incorrect = average_accuracy_of_cdt(full_vote_data, iterations)
    average_cdt_accuracy_results(best_correct, best_incorrect, avg_correct, avg_incorrect, start_time, iterations)

if __name__ == '__main__':
    main()
    
    