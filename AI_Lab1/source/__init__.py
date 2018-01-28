'''from source.Layer import layer

from source.Neuron import neuron
from asyncore import loop
'''
from source.Network import network
from array import array
inputNeuronsCount = 3
hiddenNeuronsCount = 10
outputNeuronsCount = 1
learnrate = 0.001



#First, we get the data from the file
print("Reading testdata")
datainput = list()
dataoutput = list()
file = open("Titanic.dat")
for line in file:
    Class,Age,Sex,Survived = line.split(',')
    Survived = float(Survived)
    datainput.append([float(Class),float(Age),float(Sex)])
    if Survived == 1.0:
        dataoutput.append(0.75)
    else:
        dataoutput.append(0.25)
#Then, we setup the network by creating neurons, connect them and randomize startweights
my_little_network = network(inputNeuronsCount,hiddenNeuronsCount,outputNeuronsCount,learnrate)
#Start training with the data
my_little_network.Learn(datainput, dataoutput)
