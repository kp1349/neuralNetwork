import math
import random

class NeuralNetwork:
	def __init__(self, input, hidden, output, iteration, learning_rate, momentum, decay): #paul

	def Train(self, data): #paul

	def Test(self, data): #josh

	def Forward(self, input_data): #josh
		if self.input != len(input_data):
			raise ValueError('unmatched length for inputs')

		for i in range(self.input):
			self.input_act[i] == inputs[i]

		for i in range(self.hidden):
			hidsum = 0.0
			for j in range(self.input):
				hidsum += self.input_act[j] * self.weights_in[j][i]
			self.hidden_act[i] = tanh(hidsum)

		for i in range(self.output):
			outsum = 0.0
			for j in range(self.hidden):
				outsum += self.hidden_act[j] * self.weights_out[j][i]
			self.output_act[i] = sigmoid(outsum)

	def Backward(self, correct_values): #danny

	def Error(outputs, correct_values): #danny
