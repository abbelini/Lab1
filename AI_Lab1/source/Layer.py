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
        
    def nonlin(self,x,deriv = False):
        if deriv == True: 
            return x*(1-x)
        return 1/(1+np.exp(-x))  
    def Activate(self):
        for ne in self.neuronlist:
            ne.Activate()
