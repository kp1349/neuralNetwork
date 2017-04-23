import random
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# derivative of sigmoid
def dsigmoid(y):
    return (math.exp(y)) / ((math.exp + 1)^2)

# using tanh over logistic sigmoid is recommended   
def tanh(x):
    return math.tanh(x)
    
# derivative for tanh sigmoid
def dtanh(y):
    return 4 / ((math.exp(-y) + math.exp(y))^2)


class NeuralNetwork:
	def __init__(self, input, hidden, output, iteration, learning_rate, momentum, decay): #paul
		x=0
	def Train(self, data): #paul
		x=0

	def Test(self, data): #josh
		x=0

	def Forward(self, input_data): #josh
		x=0

	def Backward(self, correct_values): #danny
		x=0
		#array of output changes
		output_delta = [0.0]* self.output
		#output to hidden layer Derivatives
		for k in range(self.output):
			get_error = self.output[k]-self.correct_values[k]
			output_delta[k] = get_error*dsigmoid(self.output[k])*self.hidden_act[k]
		
		#array of hidden changes
		hidden_delta = [0.0]*self.hidden
		tot_error = 0.0
		#hidden to input layer of Derivatives
		for i in range(self.hidden):
			error=0.0;
			for j in range(self.output)
				error += output_delta[j] * weights_out[i][j]
				hidden_delta[i] = dtanh(self.hidden[i])*self.input_act


		

	def Error(outputs, correct_values): #danny
		x=0
