#Normalization script for Iris Data Set
import random

array=[]
classification=[]

with open("iris.data.txt") as infile:
	tempLines = infile.readlines()
lines=random.sample(tempLines, len(tempLines))

for line in lines:
	x=line.split(',')
	array.append([float(x[0]),float(x[1]),float(x[2]),float(x[3])])
	classification.append(x[4])

# array[j][i]
max_value=[-100.0, -100.0, -100.0, -100.0]
min_value=[100.0, 100.0, 100.0, 100.0]
for i in range(0, 4): #number of input layer
	for j in range(0, len(array)): #number of iris data rows
		if(max_value[i]<array[j][i]):
			max_value[i]=array[j][i]
		if(min_value[i]>array[j][i]):
			min_value[i]=array[j][i]

print("max: "+str(max_value[0])+" "+str(max_value[1])+" "+str(max_value[2])+" "+str(max_value[3])+"\n")

print("min: "+str(min_value[0])+" "+str(min_value[1])+" "+str(min_value[2])+" "+str(min_value[3])+"\n")


for j in range(0, len(array)):
	for i in range(0, len(array[i])):
		array[j][i]=(array[j][i] - min_value[i])/(max_value[i]-min_value[i])
		# array[j][i]-=0.5

with open("iris.data", 'w') as outfile:
	for j in range(0, len(array)):
		temp=""
		if(classification[j]=="Iris-setosa\n"):
			temp="1,0,0"
		elif(classification[j]=="Iris-versicolor\n"):
			temp="0,1,0"
		elif(classification[j]=="Iris-virginica\n"):
			temp="0,0,1"
		if(temp!=""):
			outfile.write(str(array[j][0])+","+str(array[j][1])+","+str(array[j][2])+","+str(array[j][3])+","+str(temp)+"\n")

