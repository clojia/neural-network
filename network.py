import numpy as np
import random
import math
from scipy.stats import pearsonr
from itertools import combinations

class NNUtil(object):
    """
    neural network utility functions: activation functions.
    to do: 
        more activation functions
        loss functions
    """
    @staticmethod
    def softmax(sums):
        result = np.zeros(shape=[len(sums)])
        eSums = 0.0
        for k in range(len(sums)):
            eSums += math.exp(sums[k])
        for k in range(len(result)):
            result[k] = math.exp(sums[k]) / eSums
        return result

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-1 * x))


class NeuralNetwork:
    """
    neural network class (one hidden layer): model training (backpropagation), predicting and calculating accuracy 
    customize: learning rate, momentum, epochs, validation, weight decay, etc.

    """
    def __init__(self, numInput, numHidden, numOutput, seed):
        self.numInput = numInput
        self.numHidden = numHidden
        self.numOutput = numOutput 

        self.inputNodes = np.zeros(shape=[self.numInput])
        self.hiddenNodes = np.zeros(shape=[self.numHidden])
        self.outputNodes = np.zeros(shape=[self.numOutput])

        self.inputHiddenWeights = np.zeros(shape=[self.numInput, self.numHidden])
        self.hiddenOutputWeights = np.zeros(shape=[self.numHidden, self.numOutput])

        self.validInputHiddenWeights = np.zeros(shape=[self.numInput, self.numHidden])
        self.validHiddenOutputWeights = np.zeros(shape=[self.numHidden, self.numOutput])
        self.hiddenBiases = np.zeros(shape=[self.numHidden])
        self.outputBiases = np.zeros(shape=[self.numOutput]) 
        self.validHiddenBiases = np.zeros(shape=[self.numHidden])
        self.validOutputBiases = np.zeros(shape=[self.numOutput])
        self.rnd = random.Random(seed)  # allows multiple instances
        self.initializeWeights()

    def setWeights(self, weights):
        idx = 0
        for i in range(self.numInput):
            for j in range(self.numHidden):
                self.inputHiddenWeights[i, j] = weights[idx]
                idx += 1

        for j in range(self.numHidden):
            self.hiddenBiases[j] = weights[idx]
            idx += 1
        for i in range(self.numHidden):
            for j in range(self.numOutput):
                self.hiddenOutputWeights[i, j] = weights[idx]
                idx += 1

        for k in range(self.numOutput):
            self.outputBiases[k] = weights[idx]
            idx += 1

    def getWeights(self):
       
        numWts = (self.numInput * self.numHidden) + (self.numHidden * self.numOutput) + self.numHidden + self.numOutput 
        result = np.zeros(shape=[numWts])
        idx = 0  # points into result

        for i in range(self.numInput):
            for j in range(self.numHidden):
                result[idx] = self.inputHiddenWeights[i, j]
                idx += 1

        for j in range(self.numHidden):
            result[idx] = self.hiddenBiases[j]
            idx += 1

        for i in range(self.numHidden):
            for k in range(self.numOutput):
                result[idx] = self.hiddenOutputWeights[i, k]
                idx += 1

        for k in range(self.numOutput):
            result[idx] = self.outputBiases[k]
            idx += 1

        return result

    def initializeWeights(self):
        numWts = (self.numInput * self.numHidden) + (self.numHidden * self.numOutput) + self.numHidden + self.numOutput 
        wts = np.zeros(shape=[numWts])
        lo = -0.1
        hi = 0.1
        for idx in range(len(wts)):
            wts[idx] = (hi - lo) * self.rnd.random() + lo
        self.setWeights(wts)

    def predict(self, xValues, validationOn=False):

        hSums = np.zeros(shape=[self.numHidden])
        oSums = np.zeros(shape=[self.numOutput])
   
        for i in range(self.numInput):   
            self.inputNodes[i] = xValues[i] 

        for j in range(self.numHidden):
            for i in range(self.numInput):
                if validationOn:
                    hSums[j] += self.inputNodes[i] * self.validInputHiddenWeights[i, j]
                else:
                    hSums[j] += self.inputNodes[i] * self.inputHiddenWeights[i, j]
 
        for j in range(self.numHidden):
            if validationOn:
                hSums[j] += self.validHiddenBiases[j]
            else:
                hSums[j] += self.hiddenBiases[j]

        for j in range(self.numHidden):
            self.hiddenNodes[j] = NNUtil.sigmoid(hSums[j])

        for k in range(self.numOutput):
            for j in range(self.numHidden):
                if validationOn:
                    oSums[k] += self.hiddenNodes[j] * self.validHiddenOutputWeights[j, k] 
                else:
                    oSums[k] += self.hiddenNodes[j] * self.hiddenOutputWeights[j, k]

        for k in range(self.numOutput):
            if validationOn:
                oSums[k] += self.validOutputBiases[k] 
            else:
                oSums[k] += self.outputBiases[k]

        for k in range(self.numOutput):
            self.outputNodes[k] = NNUtil.sigmoid(oSums[k])

        result = np.zeros(shape=self.numOutput)
        for k in range(self.numOutput):
            result[k] = self.outputNodes[k]

        return self.outputNodes

    def predictLabel(self, xValues):
        predictP = self.predict(xValues)
        predictL = np.zeros(len(predictP))
        max_index = np.argmax(predictP)
        predictL[max_index] = 1
        return predictL

    def train(self,
              trainData, 
              maxEpochs, 
              learningRate=0.1, 
              momentum=0, 
              showEpochs=True,
              showHidden=False,  
              vRatio=1.0, 
              weightDecay=1.0):
        """
        update weights and biases using backpropagation
        """
        hiddenOutputGrads = np.zeros(shape=[self.numHidden, self.numOutput])
        outputBiaseGrads = np.zeros(shape=[self.numOutput])
        inputHiddenGrads = np.zeros(shape=[self.numInput, self.numHidden])
        hiddenBiaseGrads = np.zeros(shape=[self.numHidden])

        outputErrorTerm = np.zeros(shape=[self.numOutput])
        hiddenErrorTerm = np.zeros(shape=[self.numHidden])

        #keep previous delta for momentum
        input_hidden_prev_weights_delta = np.zeros(
            shape=[self.numInput, self.numHidden])
        hidden_prev_biases_delta = np.zeros(shape=[self.numHidden])
        hidden_output_prev_weights_delta = np.zeros(
            shape=[self.numHidden, self.numOutput])
        output_prev_biases_delta = np.zeros(shape=[self.numOutput])

        epoch = 0
        maxValidationAcc = 0
        x_values = np.zeros(shape=[self.numInput])
        y_values = np.zeros(shape=[self.numOutput])
        vData = trainData[int(vRatio*len(trainData)):]
        tData = trainData[:int(vRatio*len(trainData))]

        numTrainingExamples = len(tData)
        indices = np.arange(numTrainingExamples)  #shuffle training examples each epoch
        pearsonSum = 0    
        pearsonMax = 0
        while epoch <= maxEpochs :
            self.rnd.shuffle(indices)
            x_dim = len(tData)
            y_dim = self.numHidden
            hiddenMatrix = np.zeros(shape=(x_dim,y_dim))
     
            for it in range(len(tData)):
                idx = indices[it]

                for j in range(self.numInput):
                    x_values[j] = tData[idx, j]

                for j in range(self.numOutput):
                    y_values[j] = tData[idx, j + self.numInput]

                #1.compute the output O_u of every unit u
                self.predict(x_values)

                maxOut = max(self.outputNodes)
                #2i.calculate error term for each output unit
               # print(pearsonSum)
                for k in range(self.numOutput):
                    derivative = (1 - self.outputNodes[k]) * self.outputNodes[k]
                    outputErrorTerm[k] = derivative * (y_values[k] - self.outputNodes[k])

                #2ii. calculate hidden-output weight gradients
                for j in range(self.numHidden):
                    for k in range(self.numOutput):
                        hiddenOutputGrads[j, k] = outputErrorTerm[k] * self.hiddenNodes[j]

                #2iii. calculate hidden-output bias gradients
                for k in range(self.numOutput):
                    outputBiaseGrads[k] = outputErrorTerm[k] * 1.0  #x_0 is always 1

                #3i.calculate error term for each hidden unit
                for j in range(self.numHidden):
                    sumW = 0.0
                    for k in range(self.numOutput):
                        sumW += outputErrorTerm[k] * self.hiddenOutputWeights[j, k]
                    derivative = (1 - self.hiddenNodes[j]) * self.hiddenNodes[j]
                    hiddenErrorTerm[j] = derivative * sumW

                #3ii. calculate input-hidden weight gradients
                for i in range(self.numInput):
                    for j in range(self.numHidden):
                        inputHiddenGrads[i, j] = hiddenErrorTerm[j] * self.inputNodes[i]

                #3iii. calculate input-hidden bias gradients
                for j in range(self.numHidden):
                    hiddenBiaseGrads[j] = hiddenErrorTerm[j] * 1.0

                #4. update each weight
                for i in range(self.numInput):
                    for j in range(self.numHidden):
                        delta = learningRate * inputHiddenGrads[i, j]
                        self.inputHiddenWeights[i, j] += delta
                        # add momentum
                        self.inputHiddenWeights[i,j] += momentum * input_hidden_prev_weights_delta[i,j]
                        self.inputHiddenWeights[i,j] *= weightDecay
                        input_hidden_prev_weights_delta[i, j] = delta

                for j in range(self.numHidden):
                    delta = learningRate * hiddenBiaseGrads[j]
                    self.hiddenBiases[j] += delta
                    self.hiddenBiases[j] += momentum * hidden_prev_biases_delta[j]
                    self.hiddenBiases[j] *= weightDecay
                    hidden_prev_biases_delta[j] = delta

                for j in range(self.numHidden):
                    for k in range(self.numOutput):
                        delta = learningRate * hiddenOutputGrads[j, k]
                        self.hiddenOutputWeights[j, k] += delta
                        self.hiddenOutputWeights[j,k] += momentum * hidden_output_prev_weights_delta[j, k]
                        self.hiddenOutputWeights[j, k]*= weightDecay
                        hidden_output_prev_weights_delta[j, k] = delta

                for k in range(self.numOutput):
                    delta = learningRate * outputBiaseGrads[k]
                    self.outputBiases[k] += delta
                    self.outputBiases[k] += momentum * output_prev_biases_delta[k]
                    self.outputBiases[k] *= weightDecay
                    output_prev_biases_delta[k] = delta

                hiddenMatrix[it] = self.hiddenNodes 
            hiddenPairs = list(combinations(hiddenMatrix.T, 2)) 
            pearsonSum = 0
            pearsons = []
            for t, pair in enumerate(hiddenPairs):
                r,p = pearsonr(pair[0], pair[1])
                pearsonSum += abs(r)
                pearsons.append(abs(r))
            pearsonMax = max(pearsons)
            if epoch == maxEpochs:
                print("pearsons max: " + str(pearsonMax))

          #  print(str(hiddenMatrix.T))
            epoch += 1
            validationAcc = 0
            if vRatio < 1:
                validationAcc = self.accuracy(vData)
            if validationAcc >= maxValidationAcc:    
                self.validInputHiddenWeights = np.copy(self.inputHiddenWeights)
                self.validHiddenOutputWeights = np.copy(self.hiddenOutputWeights)
                self.validHiddenBiases = np.copy(self.hiddenBiases)
                self.validOutputBiases = np.copy(self.outputBiases)
                maxValidationAcc = validationAcc 
            if epoch % 10 == 0 and showEpochs:
                acc = self.accuracy(tData)
                print("epoch = " + str(epoch) + "   acc = %0.4f" % acc)

        if showHidden:
            for it in range(numTrainingExamples):
                print(str(trainData[it, :self.numInput]), end = " -> ") 
                predict = self.predict(trainData[it, :self.numInput])
                print(str(self.hiddenNodes), end = " -> ")
        #        print(str(self.outputNodes), end = " -> ")
                print(str(self.predictLabel(predict)))


        return self.getWeights()

    def accuracy(self, tdata, validationOn=False): 
        num_correct = 0
        num_wrong = 0
        x_values = np.zeros(shape=[self.numInput])
        t_values = np.zeros(shape=[self.numOutput])

        for i in range(len(tdata)): 
            for j in range(self.numInput):  
                x_values[j] = tdata[i, j]
            for j in range(self.numOutput):
                t_values[j] = tdata[i, j + self.numInput]

            y_values = self.predict(x_values, validationOn)
       #     print(str(self.hiddenNodes))
           
            y_values_label = self.predictLabel(x_values)
         #   if (str(t_values) != str(y_values_label)):
             #   print(str(t_values) + "  vs   "  + str(y_values))
            max_index = np.argmax(y_values) 
           # print(max_index)
            if abs(t_values[max_index] - 1.0) < 1.0e-5:
                num_correct += 1
            else:
                num_wrong += 1

        return (num_correct * 1.0) / (num_correct + num_wrong)
