import math
import random

class NeuralNetwork:
	def __init__(self, input, hidden, output, iteration, learning_rate, momentum, decay): #paul
		self.input = input
		self.hidden = hidden
		self.output = output
		self.iteration = iteration
		self.learning = learning_rate
		self.momentum = momentum
		self.decay = decay

		self.input_act = [0.0] * input
		self.hidden_act = [0.0] * hidden
		self.output_act = [0.0] * output

		self.weights_in = []
		for i in range(0, input):
			self.weights_in.append([])
			for j in range(0, hidden):
				# self.weights_in[i].append(random.uniform(0.0, 1.0))
				self.weights_in[i].append(0.5)

		self.weights_out = []
		for i in range(0, hidden):
			self.weights_out.append([])
			for j in range(0, output):
				# self.weights_out[i].append(random.uniform(0.0, 1.0))
				self.weights_out[i].append(0.5)

		self.old_change_in = []
		for i in range(0, input):
			self.old_change_in.append([])
			for j in range(0, hidden):
				self.old_change_in[i].append(0.0)

		self.old_change_out = []
		for i in range(0, hidden):
			self.old_change_out.append([])
			for j in range(0, output):
				self.old_change_out[i].append(0.0)

	def Train(self, input_data, output_data): #paul
		x=0

	def Test(self, input_data, output_data): #josh
		x=0

	def tanh(x):
		return (math.exp(x)-math.exp(-x)) / (math.exp(-x)+math.exp(x))

	def sigmoid(x):
		return 1/(1+(math.exp(-x)))

	def Forward(self, input_data): #josh
		for i in range(0, self.input):
			self.input_act[i] == inputs[i]

		for i in range(0, self.hidden):
			hidsum = 0.0
			for j in range(0, self.input):
				hidsum += self.input_act[j] * self.weights_in[j][i]
			self.hidden_act[i] = tanh(hidsum)

		for i in range(0, self.output):
			outsum = 0.0
			for j in range(0, self.hidden):
				outsum += self.hidden_act[j] * self.weights_out[j][i]
			self.output_act[i] = sigmoid(outsum)

	def Backward(self, correct_values): #danny
		x=0

	def Error(outputs, correct_values): #danny
		x=0

new = NeuralNetwork(4,3,3, 100, 0.5, 0.3, 0.01)
# print(new.weights_in)
# print(new.weights_out)
# print(new.old_change_in)
# print(new.old_change_out)


