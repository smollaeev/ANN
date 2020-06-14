import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from independent_Variables import IndependentVariables

class Solution:
    def __init__ (self, X, Y):
        self.independentVariables = X
        self.dependentVariable = Y
        # self.numberOfNeurons = numberOfNeurons
        # self.numberOfHiddenLayers = numberOfHiddenLayers
        # self.numberOfInputs = numberOfInputs
        # self.learningRate = learningRate
        self.weights = []
        self.X_testSet, self.Y_testSet, self.X_trainingSet, self.Y_trainingSet = self.determine_TestAndTrainingSets ()

    def determine_TestAndTrainingSets (self):
        self.independentVariables.encode_CategoricalVariables ()
        X_trainingSetArray, X_testSetArray, self.Y_trainingSet, self.Y_testSet = train_test_split (self.independentVariables.array, self.dependentVariable, test_size = 0.2, random_state = 0) 

        self.X_trainingSet = IndependentVariables (X_trainingSetArray)
        self.X_testSet = IndependentVariables (X_testSetArray)

        return self.X_testSet, self.Y_testSet, self.X_trainingSet, self.Y_trainingSet

    def classify (self):
        self.X_testSet.scale_Features ()
        self.X_trainingSet.scale_Features ()

    
    # def initialize (self):
    #     for i in range (self.numberOfInputs * self.numberOfNeurons):
    #         self.weights.append (random.random () / 100)

    # def get_Inputs (self):

    # def forward_Propagation (self):

    # def compare_PredictedAndActualResults (self):

    # def back_Propagation (self):

    # def ANN (self):
    #     while 
    #         self.initialize ()
    #         self.get_Inputs ()
    #         self.forward_Propagation ()
    #         self.compare_PredictedAndActualResults ()
    #         self.back_Propagation ()


