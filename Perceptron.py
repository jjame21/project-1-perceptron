import numpy as np

class Perceptron(object):

	# Create a new Perceptron
	# 
	# Params:	bias -	arbitrarily chosen value that affects the overall output
	#					regardless of the inputs
	#
	#			synaptic_weights -	list of initial synaptic weights for this Perceptron
	def __init__(self, bias, synaptic_weights):
		
		self.bias = bias
		self.synaptic_weights = synaptic_weights


	# Activation function
	#	Quantizes the induced local field
	#
	# Params:	z - the value of the indiced local field
	#
	# Returns:	an integer that corresponds to one of the two possible output values (usually 0 or 1)
	def activation_function(self, z):
		# R = 0, M = 1
		return 1 if z >= 0 else 0


	# Compute and return the weighted sum of all inputs (not including bias)
	#
	# Params:	inputs - a single input vector (which may contain multiple individual inputs)
	#
	# Returns:	a float value equal to the sum of each input multiplied by its
	#			corresponding synaptic weight
	def weighted_sum_inputs(self, inputs):
		return np.dot(inputs, self.synaptic_weights)

	# Compute the induced local field (the weighted sum of the inputs + the bias)
	#
	# Params:	inputs - a single input vector (which may contain multiple individual inputs)
	#
	# Returns:	the sum of the weighted inputs adjusted by the bias
	def induced_local_field(self, inputs):
		return inputs + self.bias

	# Predict the output for the specified input vector
	#
	# Params:	input_vector - a vector or row containing a collection of individual inputs
	#
	# Returns:	an integer value representing the final output, which must be one of the two
	#			possible output values (usually 0 or 1)
	def predict(self, input_vector):
		return self.activation_function(self.induced_local_field(self.weighted_sum_inputs(input_vector)))

	# Train this Perceptron
	#
	# Params:	training_set - a collection of input vectors that represents a subset of the entire dataset
	#			learning_rate_parameter - 	the amount by which to adjust the synaptic weights following an
	#										incorrect prediction
	#			number_of_epochs -	the number of times the entire training set is processed by the perceptron
	#
	# Returns:	no return value
	def train(self, training_set, learning_rate_parameter, number_of_epochs):
		n = len(training_set[0]) - 1
		self.synaptic_weights = np.zeros(n)
		for epoch in range(number_of_epochs):
			for row in training_set:
				# x
				inputs = row[:-1]
				# y-hat
				expected_output = row[-1]
				# y = w * x + b
				output = self.predict(inputs)
				# calculate loss
				error = expected_output - output
				# apply learning adjusted by signal and error
				for i in range(n):
					self.synaptic_weights[i] += learning_rate_parameter * error * inputs[i]
				self.bias += learning_rate_parameter * error
			rng = np.random.default_rng(epoch)
			rng.shuffle(training_set)
		return

	# Test this Perceptron
	# Params:	test_set - the set of input vectors to be used to test the perceptron after it has been trained
	#
	# Returns:	a collection or list containing the actual output (i.e., prediction) for each input vector
	def test(self, test_set):
		# feed each input list into the perceptron
		# remove each last element representing the expected output label
		return [self.predict(t) for t in test_set[:, :-1]]
                
