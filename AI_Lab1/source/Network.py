'''
Created on 22 jan. 2018

@author: Albert
'''
from source.Layer import layer
from source.Neuron import NN
from functools import total_ordering


class network(object):
    '''
    classdocs
    '''
    '''Add the 3 layers, '''


    def __init__(self, num_input,num_hidden,num_output,LearningRate):
        '''
        Constructor
        '''
        self.inputLayer = layer()
        self.hiddenLayer = layer()
        self.outputLayer = layer()
        self.LearningRate = LearningRate
        
        for x in range(0, num_input):
            self.inputLayer.neuronlist.append(NN())
            
        #Create hiddenlayer neurons and add connections to input layer
        for x in range(0, num_hidden):
            self.hiddenLayer.neuronlist.append(NN())
            for n in self.inputLayer.neuronlist:
                self.hiddenLayer.neuronlist[x].Connectneuron(n)
                
        #Create outputlayer neuron and add connections to hidden layer    
        for x in range(0, num_output):
            self.outputLayer.neuronlist.append(NN())
            for n in self.hiddenLayer.neuronlist:
                self.outputLayer.neuronlist[x].Connectneuron(n)
        
    def Learn(self,datainput,dataoutput):
        K=0
        K_SIZE=20
        #K data will be use to test the network
        Rounds = 0
        while Rounds < 10:
            Rounds +=1
            Right = 0
            Wrong = 0
            while K * K_SIZE < len(datainput):
                Rounds+=1
                for i in range(0,len(datainput)):
                    #If this is part of the test data, skip
                    if i == K * K_SIZE:
                        i += K_SIZE - 1
                        continue
                    self.train(datainput[i], dataoutput[i])
                #After being trained trough the data, apply the changes
                self.ApplyWeight()
                #Then, test the system

                for i in range(K * K_SIZE,K_SIZE):
                    if i == len(datainput):
                        break
                    Answer = self.Use(datainput[i])
                    if Answer < 0.5:
                        if dataoutput[i] == 0.25:
                            Right +=1
                        else:
                            Wrong +=1
                    else:
                        if dataoutput[i] == 0.75:
                            Right +=1
                        else:
                            Wrong +=1
                    
                K = K + 1
                print(".",end=" ")
            print("\nRight answers:" + str(Right) + "\nWrong answers:" + str(Wrong) +"\n\n")
        
    def train(self,indata,output):
        #Setup the inputlayer
        i = 0
        for value in indata:
            self.inputLayer.neuronlist[i].output = value
            i=i+1
            
        #Activate to see the result
        self.Activate()
        
        #Backpropegate
        self.BP(output)
        
    def Activate(self):
        self.hiddenLayer.Activate();
        self.outputLayer.Activate();
        
    def BP(self,exptoutput):

        #Error in outputlayer
        for neuron in self.outputLayer.neuronlist:
            o = neuron.output
            #neuron.error = (exptoutput - o) * o * (1.0 - o)
            neuron.error = (exptoutput - o) * o * (1.0 - o)

        #Error in hiddenlayer
        for neuron in self.hiddenLayer.neuronlist:
            neuron.error = 0.0
            error = 0.0
            for out in self.outputLayer.neuronlist:
                error += out.error * out.weightof(neuron) * (1.0 - neuron.output) * neuron.output
            neuron.error = error
        
        #Change weight of outputlayer
        for neuron in self.outputLayer.neuronlist:
            for hid in self.hiddenLayer.neuronlist:
                neuron.inputweightchanges[hid] += self.LearningRate * neuron.error * hid.output
            neuron.w0changes += self.LearningRate * neuron.error
            
        #Change weight of hiddenlayer
        for neuron in self.hiddenLayer.neuronlist:
            for inp in self.inputLayer.neuronlist:
                neuron.inputweightchanges[inp] += self.LearningRate * neuron.error * inp.output
            neuron.w0changes += self.LearningRate * neuron.error
        
    def ApplyWeight(self):
        self.hiddenLayer.ApplyWeight()
        self.outputLayer.ApplyWeight()
        
    def Use(self,indata):
        i = 0
        for value in indata:
            self.inputLayer.neuronlist[i].output = value
            i=i+1
        self.Activate()
        return self.outputLayer.neuronlist[0].output
        