# CODE WRITTEN BY: KOUSHIK PAUL, JAI SHENG FU, DANNY WU
# SCROLL ALL THE WAY TO THE BOTTOM TO SEE INSTRUCTIONS

import math
import random

TRAINPERCENTAGE = 0.8
THRESHOLD = 0.1
NUM_OF_BIAS_NODES = 1

class NeuralNetwork:
	def __init__(self, input, hidden, output, iteration, learning_rate, momentum, decay): #paul
		self.bias = NUM_OF_BIAS_NODES
		self.input = input + self.bias
		self.hidden = hidden
		self.output = output
		self.iteration = iteration
		self.learning = learning_rate
		self.momentum = momentum
		self.decay = decay

		self.input_act = [1.0] * self.input
		self.hidden_act = [1.0] * self.hidden
		self.output_act = [1.0] * self.output

		self.weights_in = []
		for i in range(self.input):
			self.weights_in.append([])
			for j in range(self.hidden):
				self.weights_in[i].append(random.uniform(0.0, 1.0))

		self.weights_out = []
		for i in range(self.hidden):
			self.weights_out.append([])
			for j in range(self.output):
				self.weights_out[i].append(random.uniform(0.0, 1.0))

		self.old_change_in = []
		for i in range(self.input):
			self.old_change_in.append([])
			for j in range(self.hidden):
				self.old_change_in[i].append(0.0)

		self.old_change_out = []
		for i in range(self.hidden):
			self.old_change_out.append([])
			for j in range(self.output):
				self.old_change_out[i].append(0.0)

	def Train(self, input_data, output_data): #paul
		if (len(input_data) != len(output_data)):
			print("input and output aren't the same size")
			print("input: "+str(len(input_data)))
			print("output: "+str(len(output_data)))

		outputs = []

		for k in range(self.iteration):
			totalError = 0.0
			for i in range(len(input_data)):
				outputs = self.Forward(input_data[i])
				Error=0.0
				for j in range(len(output_data[i])):
					Error += abs(output_data[i][j] - outputs[j])
				totalError += (Error / len(output_data[i]))
				self.Backward(output_data[i])
			print("Iteration "+str(k+1)+" Error: " + str(totalError / len(input_data)))
			self.learning = self.learning * (self.learning / (self.learning + (self.learning * self.decay))) #update the learning rate before each cycle

	def Train_debug(self, input_data, output_data): #paul
		if (len(input_data) != len(output_data)):
			print("input and output aren't the same size")

		outputs = []

		with open("testing.data.txt", "w") as outfile:
			for k in range(self.iteration):
				totalError = 0.0
				for i in range(len(input_data)):
					outputs = self.Forward(input_data[i])
					caclculatedError = 0.0
					realError = 0.0
					possibleError = 0.0
					for j in range(len(output_data[i])):
						caclculatedError += 0.5 * abs(output_data[i][j] - outputs[j])**2
						realError += abs(output_data[i][j] - outputs[j])
						possibleError += 1
					self.Backward(output_data[i])
				# print("Iteration "+str(k)+" Error: " + str(totalError / len(input_data)))
				outfile.write(str(k+1)+"\t"+str(caclculatedError)+"\t"+str(realError/possibleError)+"\n")
				# print(str(Error/possibleError))
				self.learning = self.learning * (self.learning / (self.learning + (self.learning * self.decay)))

	def Test(self, input_data, output_data): #josh
		for i in range(len(input_data)):
			print(str(output_data[i])+" --> "+str(self.Forward(input_data[i])))

	def Test_debug(self, input_data, output_data): #josh
		# Error = 0.0
		max_value = 0.0
		min_value = 1.0
		max_ndx = 0
		min_ndx = 0

		totalError = 0.0

		outliers = []

		lines = []
		for i in range(len(input_data)):
			lines.append(str(i)+": "+str(output_data[i])+" --> "+str(self.Forward(input_data[i])))
			sum = 0.0
			output = self.Forward(input_data[i])
			for j in range(len(output_data[i])):
				temp = abs(output_data[i][j] - output[j])
				sum += temp
				if (temp>max_value):
					max_value = temp
					max_ndx = i
				if (temp<min_value):
					min_value = temp
					min_ndx = i
				if (temp > THRESHOLD):
					outliers.append(str(i)+": "+str(temp)+"\t\t"+str(output_data[i][j])+" - "+str(output[j]))
			Error = sum / len(output_data[i])
			totalError += Error
			print(str(i)+": "+str(Error))
		for line in lines:
			print(line)

		print("max: "+str(max_value)+" ["+str(max_ndx)+"]")
		print("min: "+str(min_value)+" ["+str(min_ndx)+"]")
		print("avg: "+str(totalError / len(input_data)))
		print("outliers:")
		for lie in outliers:
			print(lie)

	def Tanh(self, x):
		return (math.exp(x)-math.exp(-x)) / (math.exp(-x)+math.exp(x))
	
	def derivativeOfTanh(self, y):
		return 4 / ((math.exp(-y) + math.exp(y))**2)

	def Sigmoid(self, x):
		return 1/(1+(math.exp(-x)))

	def derivativeOfSigmoid(self, y):
		return (math.exp(y)) / ((math.exp(y) + 1)**2)

	def Forward(self, input_data):
		# INPUT NODES
		for i in range(0, self.input - self.bias):
			self.input_act[i] = input_data[i]

		# HIDDEN LAYER NODES
		for i in range(0, self.hidden):
			hidsum = 0.0
			for j in range(0, self.input):
				hidsum += self.input_act[j] * self.weights_in[j][i]
			self.hidden_act[i] = self.Tanh(hidsum)

		# OUTPUT LAYER NODES
		for i in range(self.output):
			outsum = 0.0
			for j in range(self.hidden):
				outsum += self.hidden_act[j] * self.weights_out[j][i]
			self.output_act[i] = self.Sigmoid(outsum)

		return self.output_act

	def Backward(self, correct_values):
		# ARRAY OF DELTAS FOR SECOND LAYER WEIGHTS
		output_delta = [0.0] * self.output 
		for k in range(0, self.output):
			output_delta[k] = (self.output_act[k] - correct_values[k]) * self.derivativeOfSigmoid(self.output_act[k])
		
		# ARRAY OF DELTAS FOR FIRST LAYER WIEGHTS
		hidden_delta = [0.0] * self.hidden
		for i in range(0, self.hidden):
			error=0.0
			for j in range(0, self.output):
				error += output_delta[j] * self.weights_out[i][j] # PARTS 1 AND 3 OF DERIVATIVE FORMULA
			hidden_delta[i] = self.derivativeOfTanh(self.hidden_act[i]) * error # PART 2 ADDED
		
		#updating weights for hidden weights
		for m in range(0, self.hidden):
			for n in range(0, self.output):
				self.weights_out[m][n] -= self.learning * output_delta[n] * self.hidden_act[m] + self.old_change_out[m][n] * self.momentum
				self.old_change_out[m][n] = output_delta[n] * self.hidden_act[m]
		
		#updating weights for input weights
		for s in range(0, self.input):
			for t in range(0, self.hidden):
				self.weights_in[s][t] -= self.learning * hidden_delta[t] * self.input_act[s] + self.old_change_in[s][t] * self.momentum
				self.old_change_in[s][t] = hidden_delta[t] * self.input_act[s]

	def PrintWeights(self):
		for weight in self.weights_in:
			print(weight)
		print("\n")
		for weight in self.weights_out:
			print(weight)
		print("\n===========\n")

def IrisData(filename, input_start, input_end, output):
	data = []
	classification = []

	with open(filename) as file:
		templines = file.readlines()
	templines = random.sample(templines, len(templines))

	for line in templines:
		temp = line.split(',')

		t = []
		for i in temp[input_start:input_end]:
			t.append(float(i))
		data.append(t)
		if (temp[output] == "Iris-setosa\n"):
			classification.append([1.0, 0.0, 0.0])
		elif (temp[output] == "Iris-versicolor\n"):
			classification.append([0.0, 1.0, 0.0])
		elif (temp[output] == "Iris-virginica\n"):
			classification.append([0.0, 0.0, 1.0])

	max_value=[-100.0] * (input_end - input_start)
	min_value=[100.0] * (input_end - input_start)
	for i in range(input_end - input_start): #number of input layer
		for j in range(len(data)): #number of iris data rows
			if(max_value[i]<data[j][i]):
				max_value[i]=data[j][i]
			if(min_value[i]>data[j][i]):
				min_value[i]=data[j][i]

	for j in range(0, len(data)):
		for i in range(0, len(data[i])):
			data[j][i]=(data[j][i] - min_value[i])/(max_value[i]-min_value[i])

	train_inputs = data[0:int(len(data) * TRAINPERCENTAGE)]
	train_outputs = classification[0:int(len(classification) * TRAINPERCENTAGE)]

	test_inputs = data[int(len(data) * TRAINPERCENTAGE): len(data)]
	test_outputs = classification[int(len(classification) * TRAINPERCENTAGE): len(classification)]
	net = NeuralNetwork(4,10,3, 200, 0.6, 0.3, 0.001)

	net.Train(train_inputs, train_outputs)

	net.Test_debug(test_inputs, test_outputs)

def WineData(filename, input_start, input_end, output):
	data = []
	classification = []

	with open(filename) as file:
		templines = file.readlines()
	templines = random.sample(templines, len(templines))

	for line in templines:
		temp = line.split(',')

		t = []
		for i in temp[input_start:input_end]:
			t.append(float(i))
		data.append(t)

		if (temp[output] == "1"):
			classification.append([1.0, 0.0, 0.0])
		elif (temp[output] == "2"):
			classification.append([0.0, 1.0, 0.0])
		elif (temp[output] == "3"):
			classification.append([0.0, 0.0, 1.0])

	max_value=[-100.0] * (input_end - input_start)
	min_value=[100.0] * (input_end - input_start)
	for i in range(input_end - input_start): #number of input layer
		for j in range(len(data)): #number of iris data rows
			if(max_value[i]<data[j][i]):
				max_value[i]=data[j][i]
			if(min_value[i]>data[j][i]):
				min_value[i]=data[j][i]

	for j in range(0, len(data)):
		for i in range(0, len(data[i])):
			data[j][i]=(data[j][i] - min_value[i])/(max_value[i]-min_value[i])

	train_inputs = data[0:int(len(data) * TRAINPERCENTAGE)]
	train_outputs = classification[0:int(len(classification) * TRAINPERCENTAGE)]

	test_inputs = data[int(len(data) * TRAINPERCENTAGE): len(data)]
	test_outputs = classification[int(len(classification) * TRAINPERCENTAGE): len(classification)]

	net = NeuralNetwork((input_end - input_start),3,3, 500, 0.6, 0.3, 0.001)

	net.Train(train_inputs, train_outputs)

	net.Test_debug(test_inputs, test_outputs)

def BreastCancerWisconsinData(filename, input_start, input_end, output):
	data = []
	classification = []

	with open(filename) as file:
		templines = file.readlines()
	templines = random.sample(templines, len(templines))

	for line in templines:
		temp = line.split(',')

		t = []
		for i in temp[input_start:input_end]:
			# print(line)
			t.append(float(i))
		data.append(t)

		# print(temp[output])
		if (temp[output] == "2\n"):
			classification.append([1.0, 0.0])
		elif (temp[output] == "4\n"):
			classification.append([0.0, 1.0])

	max_value=[-100.0] * (input_end - input_start)
	min_value=[100.0] * (input_end - input_start)
	for i in range(input_end - input_start): #number of input layer
		for j in range(len(data)): #number of iris data rows
			if(max_value[i]<data[j][i]):
				max_value[i]=data[j][i]
			if(min_value[i]>data[j][i]):
				min_value[i]=data[j][i]

	for j in range(0, len(data)):
		for i in range(0, len(data[i])):
			data[j][i]=(data[j][i] - min_value[i])/(max_value[i]-min_value[i])

	train_inputs = data[0:int(len(data) * TRAINPERCENTAGE)]
	train_outputs = classification[0:int(len(classification) * TRAINPERCENTAGE)]

	test_inputs = data[int(len(data) * TRAINPERCENTAGE): len(data)]
	test_outputs = classification[int(len(classification) * TRAINPERCENTAGE): len(classification)]

	net = NeuralNetwork((input_end - input_start),6,2, 500, 0.5, 0.3, 0.01)

	net.Train(train_inputs, train_outputs)

	net.Test_debug(test_inputs, test_outputs)

def HeartDiseaseData(filename, input_start, input_end, output):
	data = []
	classification = []

	with open(filename) as file:
		templines = file.readlines()
	templines = random.sample(templines, len(templines))

	for line in templines:
		temp = line.split(',')

		t = []
		for i in temp[input_start:input_end]:
			# print(line)
			t.append(float(i))
		data.append(t)

		# print(temp[output])
		if (temp[output] == "0\n"):
			classification.append([1.0, 0.0, 0.0, 0.0, 0.0])
		elif (temp[output] == "1\n"):
			classification.append([0.0, 1.0, 0.0, 0.0, 0.0])
		elif (temp[output] == "2\n"):
			classification.append([0.0, 0.0, 1.0, 0.0, 0.0])
		elif (temp[output] == "3\n"):
			classification.append([0.0, 0.0, 0.0, 1.0, 0.0])
		elif (temp[output] == "4\n"):
			classification.append([0.0, 0.0, 0.0, 0.0, 1.0])


	max_value=[-100.0] * (input_end - input_start)
	min_value=[100.0] * (input_end - input_start)
	for i in range(input_end - input_start): #number of input layer
		for j in range(len(data)): #number of iris data rows
			if(max_value[i]<data[j][i]):
				max_value[i]=data[j][i]
			if(min_value[i]>data[j][i]):
				min_value[i]=data[j][i]

	for j in range(0, len(data)):
		for i in range(0, len(data[i])):
			data[j][i]=(data[j][i] - min_value[i])/(max_value[i]-min_value[i])

	train_inputs = data[0:int(len(data) * TRAINPERCENTAGE)]
	train_outputs = classification[0:int(len(classification) * TRAINPERCENTAGE)]

	test_inputs = data[int(len(data) * TRAINPERCENTAGE): len(data)]
	test_outputs = classification[int(len(classification) * TRAINPERCENTAGE): len(classification)]

	net = NeuralNetwork((input_end - input_start),5,5, 1000, 0.5, 0.3, 0.005)

	net.Train(train_inputs, train_outputs)

	net.Test_debug(test_inputs, test_outputs)

# UNCOMMENT TO RUN:
IrisData("iris.data.txt",0, 4, 4)
# WineData("wine.data.txt", 1, 13, 0)
# BreastCancerWisconsinData("breast-cancer-wisconsin.data.txt", 1, 10, 10)
# HeartDiseaseData("processed.cleveland.data.txt", 0,13,13)