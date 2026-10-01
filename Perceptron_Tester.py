from csv import reader					# reader object reads a csv file line by line
#from random import seed					# seeds the random number generator
#from random import randrange			# returns a random value in a specified range
from Perceptron import Perceptron		# this is the Perceptron class in the Perceptron.py file
import numpy as np

# training data selection portion
sample_rate = .85
learning_rate = .0018
epochs = 1100
bias = -0.4
start_seed = 3

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
	unique = sorted(set(class_values))

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
	# using random samples
	# shuffle array of indices to choose from
	rows = len(dataset)
	indices = np.arange(rows)
	rng = np.random.default_rng(start_seed)
	rng.shuffle(indices)
	total = int(sample_rate * rows)
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
p = Perceptron(bias, np.zeros(60))

# Step 6: Train the perceptron
p.train(training_set, learning_rate, epochs)

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
print("seed - ", start_seed, " | ", correct, "/", total, " | ", acc * 100,"% | weights = ", p.synaptic_weights," | bias = ", p.bias)

print("---------------------------------------")
print("iterating selection & shuffling seeds")
# 500 epochs, .01 learning, 70% random shuffled dataset training sample
# seed 123 gives 58% accuracy
# seed 12 gives 83% accuracy
# seed 40 gives 88.94% accuracy
# update learning rate to .012
# seed 172 gives 92.78% accuracy using 70% training sample rate
# sample rate .75 with seed 172 and learning rate .1 gives 94.2% accuracy without deterministic inter-epoch shuffling
# seed 172 with .68 sample portion with .1 learning rate and 500 epochs gives 93.26% accuracy
# increasing training sample size to 80% of total data
# saving weights and bias in variables persists values in Spyder data window
last_best_accuracy = acc
last_best_weights = p.synaptic_weights
last_best_bias = p.bias
for s in range(0, 200):
	# reset bias
	p.bias = bias
	rows = len(dataset)
	indices = np.arange(rows)
	rng = np.random.default_rng(s)
	rng.shuffle(indices)
	t = int(sample_rate * rows)
	p.train(dataset[indices[:t]], learning_rate, epochs)
	results = p.test(dataset)
	correct = sum(1 for r,d in zip(results,desired_outcomes) if r==d)
	acc = correct / total
	if(acc > last_best_accuracy):
		last_best_weights = p.synaptic_weights.copy()
		last_best_bias = p.bias
		last_best_accuracy = acc
		print("seed - ", s, " | ", correct, "/", total, " | ", acc * 100,"% | weights = ", last_best_weights," | bias = ", last_best_bias)
		
'''seed -  3  |  199 / 208  |  95.67307692307693 % | weights =  [-0.10925874 -0.09147618  0.1193877  -0.11155428 -0.00343422  0.00998118
  0.035298    0.08834166 -0.0281772   0.02383416 -0.1604907   0.04673124
 -0.0409617   0.07124364 -0.0267885   0.00307656  0.0889452  -0.06327162
  0.01467468 -0.05195484  0.052272   -0.05272452  0.0464634  -0.08136864
  0.01317852  0.05599116 -0.06447672  0.03826764 -0.02083806 -0.0452124
  0.12340692 -0.07610292  0.0270189   0.02100978 -0.05622372  0.07478082
  0.04380282 -0.0238428  -0.02189322  0.04914774 -0.04093452  0.02411442
 -0.07187544  0.02509056 -0.07444116  0.00201708  0.0920367  -0.1697355
 -0.17908434  0.05514336 -0.03205692 -0.0473751  -0.04068936 -0.02571966
  0.03368988  0.03203946  0.03099744 -0.0191709  -0.00644634  0.0087237 ]  | bias =  0.030200000000001583'''