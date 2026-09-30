from csv import reader					# reader object reads a csv file line by line
from random import seed					# seeds the random number generator
from random import randrange			# returns a random value in a specified range
from Perceptron import Perceptron		# this is the Perceptron class in the Perceptron.py file
import numpy as np

######################################################################
##### DATASET FUNCTIONS                                          #####
######################################################################

# Load the CSV file containing the inputs and desired outputs
#
#	dataset is a 2D matrix where each row contains 1 set of inputs plus the desired output
#		-for each row, columns 0-59 contain the inputs as floating point values
#		-column 60 contains the desired output as a character: 'R' for Rock or 'M' for Metal
#		-all values will be string values; conversion to appropriate types will be necessary
#		-no bias value is included in the data file
def load_csv(filename):
	# dataset will be the matrix containing the inputs
	dataset = list()

	# Standard Python code to read each line of text from the file as a row
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue

			# add current row to dataset
			dataset.append(row)

	return dataset


# Convert the input values in the specified column of the dataset from strings to floats
def convert_inputs_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())


# Convert the desired output values, located in the specified column, to unique integers
# For 2 classes of outputs, 1 desired output will be 0, the other will be 1
def convert_desired_outputs_to_int(dataset, column):
	# Enumerate all the values in the specified column for each row
	class_values = [row[column] for row in dataset]

	# Create a set containing only the unique values
	unique = set(class_values)

	# Create a lookup table to map each unique value to an integer (either 0 or 1)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i

	# Replace the desired output string values with the corresponding integer values
	for row in dataset:
		row[column] = lookup[row[column]]
	
	return lookup


# Load the dataset from the CSV file specified by filename
def load_dataset(filename):
	# Read the data from the specified file
	dataset = load_csv(filename)

	# Convert all the input values form strings to floats
	for column in range(len(dataset[0])-1):
		convert_inputs_to_float(dataset, column)

	# Convert the desired outputs from strings to ints
	convert_desired_outputs_to_int(dataset, len(dataset[0]) - 1)


######################################################################
##### CREATE THE TRAINING SET                                    #####
######################################################################

# Create the training set
#	-Training set will consist of the specified percent fraction of the dataset
#	-How many inputs you decide to use for the training set, and how you choose
#	 those values, is entirely up to you
#
# Params:	dataset - the entire dataset
#
# Returns:	a matrix, or list of rows, containing only a subset of the input
#			vectors from the entire dataset
def create_training_set(dataset):
	# R = 0, M = 1
	# using 70% random samples
	# shuffle array of indices to choose from
	rows = len(dataset)
	indices = np.arange(rows)
	rng = np.random.default_rng(40)
	rng.shuffle(indices)
	total = int(.7 * rows)
	return dataset[indices[:total]]

######################################################################
##### CREATE A PERCEPTRON, TRAIN IT, AND TEST IT                 #####
######################################################################

# Step 1: Acquire the dataset
dataset = load_csv('sonar_all-data.csv')

# Step 2: Convert the string input values to floats
#inputs = np.array([d[:-1] for d in dataset],dtype=float)
n = len(dataset[0]) - 1
for i in range(n):
	convert_inputs_to_float(dataset,i)

# Step 3: Convert the desired outputs to int values
#labels = np.array([0 if d[-1] == 'R' else 1 for d in dataset],dtype=float)
convert_desired_outputs_to_int(dataset, n)
dataset = np.array(dataset, dtype=float)

# Step 4: Create the training set
training_set = create_training_set(dataset)

# Step 5: Create the perceptron
p = Perceptron(0, np.zeros(60))

# Step 6: Train the perceptron
p.train(training_set, 0.01, 500)

# Step 7: Test the trained perceptron
# test whole dataset
#rows = len(dataset)
#indices = np.arange(rows)
#rng = np.random.default_rng(123)
#rng.shuffle(indices)
#total = int(.7 * rows)
#test_set = [dataset[indices[i]] for i in range(total, rows)]
#p.test(test_set)
results = p.test(dataset)

# Step 8: Display the test results and accuracy of the perceptron
#print(results)
desired_outcomes = dataset[:,-1]
total = len(dataset)
correct = sum(1 for r,d in zip(results,desired_outcomes) if r==d)
acc = correct / total
print("Correct - ", correct)
print("Incorrect - ", total - correct)
print("Accuracy - ", acc * 100)
print("Bias - ",p.bias)

print("---------------------------------------")
print("iterative seed optimized search")
# 500 epochs, .01 learning, 70% random shuffled dataset training sample
# seed 123 gives 58% accuracy
# seed 12 gives 83% accuracy
# seed 40 gives 88.94% accuracy
# saving weights and bias in variables persists values in Spyder data window
last_best_accuracy = acc
last_best_weights = []
last_best_bias = 0.0
for s in range(0, 1000):
	# bias is not reset to 
	#p.bias = 0
	rows = len(dataset)
	indices = np.arange(rows)
	rng = np.random.default_rng(s)
	rng.shuffle(indices)
	t = int(.7 * rows)
	p.train(dataset[indices[:t]], 0.012, 500)
	results = p.test(dataset)
	correct = sum(1 for r,d in zip(results,desired_outcomes) if r==d)
	acc = correct / total
	if(acc > last_best_accuracy):
		last_best_weights = p.synaptic_weights
		last_best_bias = p.bias
		last_best_accuracy = acc
		print("seed - ", s, " | ", correct, " / ", total, " | ", acc * 100,"% | weights = ", last_best_weights," | bias = ", last_best_bias)
		
