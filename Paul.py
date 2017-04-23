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

		input_act = [0.0] * input
		hidden_act = [0.0] * hidden
		output_act = [0.0] * output

		weights_in = []
		for i in range(0, input):
			weights_in.append([])
			for j in range(0, hidden):
				weights_in[i].append(0.0)

		weights_out = []
		for i in range(0, hidden):
			weights_out.append([])
			for j in range(0, output):
				weights_out[i].append(0.0)

		old_change_in = []
		for i in range(0, input):
			old_change_in.append([])
			for j in range(0, hidden):
				old_change_in[i].append(0.0)

		old_change_out = []
		for i in range(0, hidden):
			old_change_out.append([])
			for j in range(0, output):
				old_change_out[i].append(0.0)



	def Train(self, data): #paul
		x=0

	def Test(self, data): #josh
		x=0

	def Forward(self, input_data): #josh
		x=0

	def Backward(self, correct_values): #danny
		x=0

	def Error(outputs, correct_values): #danny
		x=0
