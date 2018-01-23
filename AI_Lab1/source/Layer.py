'''
Created on 22 jan. 2018

@author: Albert
'''
from source.Neuron import NN
class layer(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.neuronlist = list()
        
        
    def Activate(self):
        for ne in self.neuronlist:
            ne.Activate()
