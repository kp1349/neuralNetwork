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
		#array of output changes
		output_delta = self.Error(self,correct_values)
		
		#array of hidden changes
		hidden_delta = [0.0]*self.hidden
		tot_error = 0.0
		#hidden to input layer of Derivatives
		for i in range(self.hidden):
			error=0.0
			for j in range(self.output)
				error += output_delta[j] * weights_out[i][j]
				hidden_delta[i] = self.dtanh(self.hidden[i])*error
		
		#updating weights for hidden weights
		 for m in range(self.hidden):
            for n in range(self.output):
                change = output_delta[n]*self.hidden[m]
				self.weights_out[m][n] -= self.learning_rate * change + self.old_change_out[m][n] * self.momentum
                self.old_change_out[m][n] = change
		
		#updating weights for input weights
		for s in range(self.input):
            for t in range(self.hidden):
                change = hidden_delta[t] * self.input_act[s]
                self.weights_in[s][t] -= self.learning_rate * change + self.old_change_in[s][t] * self.momentum
                self.old_change_in[s][t] = change


	def Error(outputs, correct_values): #danny
		output_delta = [0.0]* self.output
		for k in range(self.output):
			get_error = self.output[k]-self.correct_values[k]
		
		return output_delta
		
