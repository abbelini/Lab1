'''from source.Layer import layer

from source.Neuron import neuron
from asyncore import loop
'''
from source.Network import network
inputNeuronsCount = 3
hiddenNeuronsCount = 10
outputNeuronsCount = 1
i=0




print("Hello world")
   

my_little_network = network(inputNeuronsCount,hiddenNeuronsCount,outputNeuronsCount)

my_little_network.Activate()
