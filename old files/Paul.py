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
				self.weights_in[i].append(random.uniform(0.0, 1.0))
				# self.weights_in[i].append(0.5)

		self.weights_out = []
		for i in range(0, hidden):
			self.weights_out.append([])
			for j in range(0, output):
				self.weights_out[i].append(random.uniform(0.0, 1.0))
				# self.weights_out[i].append(0.5)

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
		if (len(input_data) != len(output_data)):
			print("input and output aren't the same size")

		outputs = []

		for k in range(self.iteration):
			self.learning = self.learning * (self.learning / (self.learning + (self.learning * self.decay)))
			totalError = 0.0
			for i in range(len(input_data)):
				outputs = self.Forward(input_data[i])
				Error=0.0
				for j in range(2,3):
					Error += abs(output_data[i][j] - outputs[j])
				totalError += (Error / 1.0)
				self.Backward(output_data[i])
			print("Iteration "+str(k)+" Error: " + str(totalError / len(input_data)))

	def Test(self, input_data, output_data): #josh
		for i in range(len(input_data)):
			print(str(output_data[i])+" --> "+str(self.Forward(input_data[i])))

	def Tanh(self, x):
		return (math.exp(x)-math.exp(-x)) / (math.exp(-x)+math.exp(x))
	
	def dTanh(self, y):
		return 4 / ((math.exp(-y) + math.exp(y))**2)

	def Sigmoid(self, x):
		return 1/(1+(math.exp(-x)))

	def dSigmoid(self, y):
		return (math.exp(y)) / ((math.exp(y) + 1)**2)

	def Forward(self, input_data): #josh
		for i in range(0, self.input):
			# print(inputs[i])
			self.input_act[i] = input_data[i]

		for i in range(0, self.hidden):
			hidsum = 0.0
			for j in range(0, self.input):
				# print(self.input_act[j])
				# print(self.weights_in[j][i])
				hidsum += self.input_act[j] * self.weights_in[j][i]
			self.hidden_act[i] = self.Tanh(hidsum)
			# print(self.hidden_act[i])


		for i in range(self.output):
			outsum = 0.0
			for j in range(self.hidden):
				outsum += self.hidden_act[j] * self.weights_out[j][i]
			self.output_act[i] = self.Sigmoid(outsum)

		return self.output_act

	def Backward(self, correct_values): #danny
		#array of output changes
		output_delta = [0.0] * self.output
		for k in range(self.output):
			error = -(correct_values[k] - self.output_act[k])
			output_delta[k] = self.dSigmoid(self.output_act[k]) * error
			# output_delta[k] = (self.output_act[k] - correct_values[k]) * self.dSigmoid(self.output_act[k])
		# print(output_delta)
		
		#array of hidden changes
		hidden_delta = [0.0] * self.hidden
		# tot_error = 0.0
		#hidden to input layer of Derivatives
		for i in range(self.hidden):
			error=0.0
			for j in range(self.output):
				error += output_delta[j] * self.weights_out[i][j]
			hidden_delta[i] = self.dTanh(self.hidden_act[i]) * error
		
		#updating weights for hidden weights
		for m in range(self.hidden):
			for n in range(self.output):
				change = output_delta[n]*self.hidden_act[m]
				self.weights_out[m][n] -= self.learning * change + self.old_change_out[m][n] * self.momentum
				self.old_change_out[m][n] = change
		
		#updating weights for input weights
		for s in range(self.input):
			for t in range(self.hidden):
				change = hidden_delta[t] * self.input_act[s]
				self.weights_in[s][t] -= self.learning * change + self.old_change_in[s][t] * self.momentum
				self.old_change_in[s][t] = change

	def Error(outputs, correct_values): #danny
		x=0

new = NeuralNetwork(4,4,3, 200, 0.5, 0.3, 0.0)
# print(new.weights_in)
# print(new.weights_out)
# print(new.old_change_in)
# print(new.old_change_out)

with open('iris.train') as file:
	lines = file.readlines()

inputs=[]
outputs=[]

for line in lines:
	x=line.split(',')
	inputs.append([float(x[0]), float(x[1]), float(x[2]), float(x[3])])
	outputs.append([float(x[4]), float(x[5]), float(x[6])])
	# print outputs

new.Train(inputs, outputs)

with open('iris.test') as file:
	lines = file.readlines()

for line in lines:
	x=line.split(',')
	inputs.append([float(x[0]), float(x[1]), float(x[2]), float(x[3])])
	outputs.append([float(x[4]), float(x[5]), float(x[6])])
	# print outputs

# new.Forward(inputs)

new.Test(inputs, outputs)

# print(new.output_act)



