'''
Created on 22 jan. 2018

@author: Albert
'''
from source.Layer import layer
from source.Neuron import NN

class network(object):
    '''
    classdocs
    '''
    '''Add the 3 layers, '''


    def __init__(self, num_input,num_hidden,num_output):
        '''
        Constructor
        '''
        self.inputLayer = layer()
        self.hiddenLayer = layer()
        self.outputLayer = layer()
        
        for x in range(0, num_input):
            self.inputLayer.neuronlist.append(NN())
        for x in range(0, num_hidden):
            self.hiddenLayer.neuronlist.append(NN())
        for x in range(0, num_output):
            self.outputLayer.neuronlist.append(NN())
        
    def Activate(self):
        self.inputLayer.Activate();
        self.hiddenLayer.Activate();
        self.outputLayer.Activate();
        
