'''
Created on 22 jan. 2018

@author: Albert
'''
import numpy as np
class NN():

    
    def __init__(self):
        '''
        Constructor
        '''
        self.inputs = dict()
        self.w0 = 0
        self.output = 0
        

    def Activate(self):
        tempoutput = 0
        for neuron in self.inputs:
            tempoutput += neuron.weight * neuron.output
        self.output = tempoutput
