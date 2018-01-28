'''
Created on 22 jan. 2018

@author: Albert
'''
from source.Neuron import NN
import numpy as np
class layer(object):


    def __init__(self):
        '''
        Constructor
        '''
        self.neuronlist = list()
        

    def Activate(self):
        for ne in self.neuronlist:
            ne.Activate()
    def decayweights(self,decay):
        for ne in self.neuronlist:
            ne.decayweight(decay)
    def ApplyWeight(self):
        for ne in self.neuronlist:
            ne.ApplyWeight()
        