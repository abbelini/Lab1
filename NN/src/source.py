import numpy as np
import time as ti
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))

TEST_SIZE = 700
LEARN_RATE = 0.005
DECAY_RATE = 0.98
NUM_HIDDENLAYER_1=5
NUM_HIDDENLAYER_2=5

#First, we get the data from the file
print("Reading testdata")
traininginput = list()
trainingoutput = list()

file = open("Titanic.dat")
for line in file:
    Class,Age,Sex,Survived = line.split(',')
    Survived = float(Survived)
    traininginput.append([float(Class),float(Age),float(Sex)])
    if Survived == 1.0:
        trainingoutput.append([0.75])
    else:
        trainingoutput.append([0.25])
        
testinput = traininginput[-700:] # Get the last 700
testoutput = trainingoutput[-700:] # Get the last 700
traininginput = traininginput[:-700] # Get the rest
trainingoutput = trainingoutput[:-700] # Get the rest

# Convert to array to allow for matrix functions
traininginput = np.asarray(traininginput)
trainingoutput = np.asarray(trainingoutput)
testinput = np.asarray(testinput)
testoutput = np.asarray(testoutput)

np.random.seed()

# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((3,NUM_HIDDENLAYER_1)) - 1
#w0_1 = 2*np.random.random(NUM_HIDDENLAYER_1) - 1

syn1 = 2*np.random.random((NUM_HIDDENLAYER_1,NUM_HIDDENLAYER_2)) - 1
#w0_2 = 2*np.random.random(NUM_HIDDENLAYER_2) - 1
syn2 = 2*np.random.random((NUM_HIDDENLAYER_2,1)) - 1
#w0_3 = 2*np.random.random(1) - 1

for j in range(120000):



    # Feed forward through layers 0, 1, 2 and 3
    Input_Layer = traininginput
    Hidden_Layer_1 = nonlin((np.dot(Input_Layer,syn0)))#w0_1
    Hidden_Layer_2 = nonlin((np.dot(Hidden_Layer_1,syn1)))#w0_2
    Output_Layer = nonlin((np.dot(Hidden_Layer_2,syn2)))#w0_3
    


    l3_error = trainingoutput - Output_Layer
    l3_delta = l3_error*nonlin(Output_Layer,deriv=True)
    #w0_3_delta = l3_error*LEARN_RATE
    
    if (j% 10000) == 0:
        print ("Error:" + str(np.mean(np.abs(l3_error))))
        
    l2_error = l3_delta.dot(syn2.T)
    l2_delta = l2_error * nonlin(Hidden_Layer_2,deriv=True)
    #w0_2_delta = l2_error*LEARN_RATE
    
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(Hidden_Layer_1,deriv=True)
    #w0_1_delta = l1_error*LEARN_RATE
    #w0_3 += np.sum(w0_3_delta, axis=0)
    syn2 += (Hidden_Layer_2.T.dot(l3_delta)*LEARN_RATE)
    #w0_2 += np.sum(w0_2_delta, axis=0)
    syn1 += (Hidden_Layer_1.T.dot(l2_delta)*LEARN_RATE)
    #w0_1 += np.sum(w0_1_delta, axis=0)
    syn0 += (Input_Layer.T.dot(l1_delta)*LEARN_RATE)
print("Done,Starting test")

rightanswers=0
wronganswers=0
Input_Layer = testinput
Hidden_Layer_1 = nonlin((np.dot(Input_Layer,syn0)))#w0_1
Hidden_Layer_2 = nonlin((np.dot(Hidden_Layer_1,syn1)))#w0_2
Output_Layer = nonlin((np.dot(Hidden_Layer_2,syn2)))#w0_3

for j in range(700):
    if Output_Layer[j]>=0.5 and testoutput[j] == 0.75:
        rightanswers = rightanswers + 1 #Right
    elif Output_Layer[j]<0.5 and testoutput[j] == 0.25:
        rightanswers = rightanswers + 1 #Right
    else:
        wronganswers = wronganswers + 1 #Wrong
testinput = np.asarray(testinput)
testoutput = np.asarray(testoutput)
print("Right")
print(rightanswers)
print("Wrong")
print(wronganswers)