import os
import argparse
import numpy as np

from preprocessor import Preprocessor
from network import NeuralNetwork

def changeLabel(data, label):
    new_index = label.index(data[-1]) - 1
    data[-1] = label[new_index]

def addNoise(data, rate, label):
    num = int(rate/100 * len(data))
    for i in range(num):
        changeLabel(data[i], label)
#        print(data[i])
    return data

def testIdentity(trainDataFile):
    trainData = np.genfromtxt(trainDataFile)

    numInput = 8
    numHidden = 3
    numOutput = 8
    seed = 3
    learningRate = 0.3
    maxEpochs = 5000
    momentum = 0.0

    print("Generating %d-%d-%d neural network " % (numInput, numHidden, numOutput))
    nn = NeuralNetwork(numInput, numHidden, numOutput, seed)
    nn.train(trainData, maxEpochs, learningRate, momentum, showHidden=True)
    print("Training complete")

    accTrain = nn.accuracy(trainData)

    print("\nAccuracy on train data = %0.4f " % accTrain)

    numHidden = 4
    print("\nGenerating %d-%d-%d neural network " % (numInput, numHidden, numOutput))
    nn = NeuralNetwork(numInput, numHidden, numOutput, seed)
    nn.train(trainData, maxEpochs, learningRate, momentum, showHidden=True)
    print("Training complete")

    accTrain = nn.accuracy(trainData)

    print("\nAccuracy on train data = %0.4f " % accTrain)


def testTennisOrIris(trainDataFile, testDataFile, attrDataFile):
    data = Preprocessor(trainDataFile, testDataFile, attrDataFile)
    data.loadData()
    trainData = data.getMatrix(data.getTrainData())
    testData = data.getMatrix(data.getTestData())
 
    numInput = data.getNumInput()
    numOutput = len(data.getClasses())
    numHidden = 3
    seed = 4 
    learningRate = 0.1
    maxEpochs = 5000
    momentum = 0.0

    print("Generating neural network: %d-%d-%d" % (numInput, numHidden,numOutput))
    nn = NeuralNetwork(numInput, numHidden, numOutput, seed)
    nn.train(trainData, maxEpochs, learningRate, momentum)
    print("Training complete")

 #   accTrain = nn.accuracy(trainData)
    accTest = nn.accuracy(testData)

 #   print("\nAccuracy on train data = %0.4f " % accTrain)
   
    print("Accuracy on test data   = %0.4f " % accTest)
  

def testIrisNoisy(trainDataFile, testDataFile, attrDataFile):
    data = Preprocessor(trainDataFile, testDataFile, attrDataFile)
    data.loadData()
    testData = data.getMatrix(data.getTestData()) 
    numInput = data.getNumInput() 
    numOutput = len(data.getClasses())
    numHidden = 3
    seed = 4 
    learningRate = 0.1
    maxEpochs = 5000
    momentum = 0.0
 
    for rate in range(0, 21, 2):
        noisyData = addNoise(data.getTrainData(), rate, data.getClasses())
        trainData = data.getMatrix(noisyData) 
        print("\nNoise Rate (%): " + str(rate)) 
        print("Generating neural network: %d-%d-%d" % (numInput, numHidden,numOutput)) 
        nn = NeuralNetwork(numInput, numHidden, numOutput, seed)
        nn.train(trainData, maxEpochs, learningRate, momentum, showEpochs=False, vRatio=0.85)
        print("Training complete")

        accTrain = nn.accuracy(trainData)
        accTest = nn.accuracy(testData)

        accValidTrain = nn.accuracy(trainData, validationOn=True)
        accValidTest = nn.accuracy(testData, validationOn=True)
        print("w/o validation set:")
        print("Accuracy on train data = %0.4f " % accTrain)
        print("Accuracy on test data   = %0.4f " % accTest)
    
        print("w/ validation set:")
        print("Accuracy on train data = %0.4f " % accValidTrain)
        print("Accuracy on test data   = %0.4f " % accValidTest)
    

def main():
    parser = argparse.ArgumentParser(description="Decision Tree")
    parser.add_argument("-e", "--experiment", required=True, dest="experiment", 
            choices=["testIdentity", "testTennis", "testIris", "testIrisNoisy"], help='experiment name.')
    args = parser.parse_args()
    print("Experiment: " + args.experiment)
    if args.experiment == "testIdentity":
        testIdentity("data/identity/identity-train.txt")
    elif args.experiment == "testTennis":
        testTennisOrIris("data/tennis/tennis-train.txt", "data/tennis/tennis-test.txt", "data/tennis/tennis-attr.txt")
    elif args.experiment == "testIris":
        testTennisOrIris("data/iris/iris-train.txt", "data/iris/iris-test.txt", "data/iris/iris-attr.txt")
    elif args.experiment == "testIrisNoisy":
        testIrisNoisy("data/iris/iris-train.txt", "data/iris/iris-test.txt", "data/iris/iris-attr.txt") 
  
if __name__ == '__main__':
    main()
