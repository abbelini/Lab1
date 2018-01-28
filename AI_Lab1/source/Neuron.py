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
        self.inputs = dict() # Every inputting neuron for this neuron
        self.inputweightchanges = dict() # The sum of all changes to the weights in input. Applies after training
        self.w0 = np.random.uniform(0.0,1.0)
        self.w0changes = 0.0 # Sum of all changes to w0. Applies after training
        self.output = 0.0
        self.error = 0.0
        

    def Activate(self):
        tempoutput = 0.0
        tempoutput += self.w0
        for neuron, weight in self.inputs.items():
            tempoutput += weight * neuron.output
        self.output = 1.0/(1.0+np.exp(-tempoutput))
    
    def Connectneuron(self,neo):
        #Add neo as in input neuron with a random value from 0.0 to 1.0 as weight
        self.inputs[neo] = np.random.uniform(0.0,1.0)
        self.inputweightchanges[neo] = 0.0
    
    def weightof(self,neo):
        return self.inputs.get(neo)
    
    def decayweight(self,decay):
        for weight in self.inputs:
            weight *= decay
        self.w0 *= decay
    
    def ApplyWeight(self):
        for neuron, weight in self.inputs.items():
            self.inputs[neuron] += self.inputweightchanges[neuron]
            self.inputweightchanges[neuron]=0.0
        self.w0 += self.w0changes
        self.w0changes = 0.0
        
        