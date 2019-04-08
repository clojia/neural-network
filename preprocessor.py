import numpy as np

class Preprocessor(object):
    """
    Preprocessor class: loading data as matrix, one-hot encoding, etc
    """ 
    def __init__(self, trainDataFile, testDataFile, attrFile):
        self.trainDataFile = trainDataFile
        self.testDataFile = testDataFile
        self.attrFile = attrFile
        self.trainData = []
        self.testData = []
        self.numAttributes = -1
        self.label = {}
        self.classes = []
        self.attrValues = {}
        self.attributes = []
        self.numInput = 0
    
    def getNumInput(self):
        return self.numInput

    def getTrainData(self):
        return self.trainData

    def getTestData(self):
        return self.testData

    def getClasses(self):
        return self.classes

    def getAttributes(self):
        return self.attributes

    def getAttrValues(self):
        return self.attrValues

    def loadData(self):
        with open(self.attrFile, "r") as file: #load attribute file
            for line in file:
                if line.strip() == '':
                    line = next(file)
                    [label, values] = line.split(" ", 1)
                    self.classes = [x.strip() for x in values.split(" ")]
                    self.label[label] = self.classes
                    break
                [attribute, values] = [x.strip() for x in line.split(" ", 1)]
                values = [x.strip() for x in values.split(" ")]
                self.attrValues[attribute] = values
        self.numAttributes = len(self.attrValues.keys())
        self.attributes = list(self.attrValues.keys())
        with open(self.trainDataFile, "r") as file: #load training data
            for line in file:
                row = [x.strip() for x in line.split(" ")]
                if row != [] or row != [""]:
                    self.trainData.append(row)
        with open(self.testDataFile, "r") as file:
            for line in file:
                row = [x.strip() for x in line.split(" ")]
                if row!= [] or row != [""]:
                    self.testData.append(row)

    def isAttrContinuous(self, attribute):
        if attribute not in self.attributes:
            raise ValueError("Attribute not listed")
        elif self.attrValues[attribute][0] == "continuous":
            return True
        else:
            return False

    def convertCategoricalAttr(self, attribute, value):
        if self.isAttrContinuous(attribute):
            encodedAttr = [float(value)]
        else:
            encodedAttr = [0.0] * len(self.attrValues[attribute]) 
            encodedAttr[self.attrValues[attribute].index(value)] = 1.0
        return encodedAttr

    def encodeData(self, data):
        encodedData = []
        for counter, value in enumerate(data[:-1]):
            encodedData += self.convertCategoricalAttr(self.attributes[counter], value) 
        self.numInput = len(encodedData)
        encodedLabel = [0] * len(self.classes)
        encodedLabel[self.classes.index(data[-1])] = 1
        encodedData += encodedLabel
        return encodedData

    def convertDataMatrix(self, data):
        dataMatrix = []
        for row in data:
            dataMatrix.append(self.encodeData(row))
        npDataMatrix = np.asmatrix(dataMatrix)
        return npDataMatrix

    def getMatrix(self, data):
        return self.convertDataMatrix(data)

