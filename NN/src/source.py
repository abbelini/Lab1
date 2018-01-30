import numpy as np
import time as ti
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))

TEST_SIZE = 700
LEARN_RATE = 0.01
DECAY_RATE = 0.95
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
traininginput = traininginput[:-700]
trainingoutput = trainingoutput[:-700]

# Convert to array to allow for matrix functions
traininginput = np.asarray(traininginput)
trainingoutput = np.asarray(trainingoutput)
testinput = np.asarray(testinput)
testoutput = np.asarray(testoutput)

np.random.seed(1)

# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((3,10)) - 1
syn1 = 2*np.random.random((10,10)) - 1
syn2 = 2*np.random.random((10,1)) - 1

for j in range(160000):



    # Feed forward through layers 0, 1, and 2
    l0 = traininginput
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))
    l3 = nonlin(np.dot(l2,syn2))
    

    l3_error = trainingoutput - l3
    
    if (j% 10000) == 0:
        print ("Error:" + str(np.mean(np.abs(l3_error))))
        

    l3_delta = l3_error*nonlin(l3,deriv=True)*LEARN_RATE

    l2_error = l3_delta.dot(syn2.T)
    l2_delta = l2_error * nonlin(l2,deriv=True)*LEARN_RATE

    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1,deriv=True)*LEARN_RATE

    syn2 += (l2.T.dot(l3_delta))
    syn1 += (l1.T.dot(l2_delta))
    syn0 += (l0.T.dot(l1_delta))
print("Done")