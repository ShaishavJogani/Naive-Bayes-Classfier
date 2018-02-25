"""
Implementation of Gaussian Naive Bayes Classification.
The code is written from scratch and does NOT use existing functions or packages which can provide the Naive Bayes
Classifier class or fit/predict function (e.g. sklearn).

I use Bank note authentication dataset, which can be downloaded from
http://archive.ics.uci.edu/ml/datasets/banknote+authentication. Complete description of the
dataset can be also found on that web-page.

The Implementation and some of it's functions are taken from this website (https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/)

Author: Shaishav Jogani (https://www.linkedin.com/in/shaishavjogani/)
"""

import numpy as np
from random import randrange
import csv
import math


def load_csv_dataset(filename):
    """Load the CSV file"""
    lines = csv.reader(open(filename, 'rb'))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]  # Convert String to Float numbers
    return dataset


def mean(numbers):
    """Returns the mean of numbers"""
    return np.mean(numbers)


def stdev(numbers):
    """Returns the std_deviation of numbers"""
    return np.std(numbers)


def sigmoid(z):
    """Returns the sigmoid number"""
    return 1.0 / (1.0 + math.exp(-z))


def cross_validation_split(dataset, n_folds):
    """Split dataset into the k folds. Returns the list of k folds"""
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def accuracy_metric(actual, predicted):
    """Calculate accuracy percentage"""
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def evaluate_algorithm(dataset, algorithm, n_folds, ):
    """Evaluate an algorithm using a cross validation split"""
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, )
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


#############################
#############################
######## Naive Bayes  #######
#############################
#############################


def separate_by_class(dataset):
    """Split training set by class value"""
    separated = {}
    for i in range(len(dataset)):
        row = dataset[i]
        if row[-1] not in separated:
            separated[row[-1]] = []
        separated[row[-1]].append(row)
    return separated


def model(dataset):
    """Find the mean and standard deviation of each feature in dataset"""
    models = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    models.pop() #Remove last entry because it is class value.
    return models


def model_by_class(dataset):
    """find the mean and standard deviation of each feature in dataset by their class"""
    separated = separate_by_class(dataset)
    class_models = {}
    for (classValue, instances) in separated.iteritems():
        class_models[classValue] = model(instances)
    return class_models


def calculate_pdf(x, mean, stdev):
    """Calculate probability using gaussian density function"""
    if stdev == 0.0:
        if x == mean:
            return 1.0
        else:
            return 0.0
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return 1 / (math.sqrt(2 * math.pi) * stdev) * exponent


def calculate_class_probabilities(models, input):
    """Calculate the class probability for input sample. Combine probability of each feature"""
    probabilities = {}
    for (classValue, classModels) in models.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classModels)):
            (mean, stdev) = classModels[i]
            x = input[i]
            probabilities[classValue] *= calculate_pdf(x, mean, stdev)
    return probabilities


def predict(models, inputVector):
    """Compare probability for each class. Return the class label which has max probability."""
    probabilities = calculate_class_probabilities(models, inputVector)
    (bestLabel, bestProb) = (None, -1)
    for (classValue, probability) in probabilities.iteritems():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(models, testSet):
    """Get class label for each value in test set."""
    predictions = []
    for i in range(len(testSet)):
        result = predict(models, testSet[i])
        predictions.append(result)
    return predictions


def naive_bayes(train, test, ):
    """Create a naive bayes model. Then test the model and returns the testing result."""
    summaries = model_by_class(train)
    predictions = getPredictions(summaries, test)
    return predictions


def main():
    # load and prepare data
    filename = 'banknote.csv'
    dataset = load_csv_dataset(filename)

    n_folds = 3

    print "---------- Gaussian Naive Bayes ---------------"
    accuracy_naive = evaluate_algorithm(dataset, naive_bayes, n_folds)
    print "Naive Bayes Classification"
    print 'Accuracy in each fold: %s' % accuracy_naive
    print 'Average Accuracy: %f' % (sum(accuracy_naive) / len(accuracy_naive))


main()
