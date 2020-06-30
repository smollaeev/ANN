import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from independent_Variables import IndependentVariables
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix


class Solution:
    def __init__ (self, X, Y):
        self.independentVariables = X
        self.dependentVariable = Y
        self.classifier = Sequential ()
        self.weights = []
        self.X_testSet, self.Y_testSet, self.X_trainingSet, self.Y_trainingSet = self.determine_TestAndTrainingSets ()
        self.Y_Predicted = []
        self.decoded_Y_Test = []
        self.accuracy = 0

    def determine_TestAndTrainingSets (self):
        self.independentVariables.encode_CategoricalVariables ()
        X_trainingSetArray, X_testSetArray, self.Y_trainingSet, self.Y_testSet = train_test_split (self.independentVariables.array, self.dependentVariable, test_size = 0.2, random_state = 0) 

        self.X_trainingSet = IndependentVariables (X_trainingSetArray)
        self.X_testSet = IndependentVariables (X_testSetArray)

        return self.X_testSet, self.Y_testSet, self.X_trainingSet, self.Y_trainingSet

    def preprocess_Data (self):
        self.X_testSet.scale_Features ()
        self.X_trainingSet.scale_Features ()
        self.Y_testSet = to_categorical (self.Y_testSet)
        self.Y_trainingSet = to_categorical (self.Y_trainingSet)
    
    def ANN (self):
        self.initialize ()
        self.classifier.compile (optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        self.classifier.fit (self.X_trainingSet.array, self.Y_trainingSet, batch_size= 5, epochs= 1000)
        self.predict ()
        self.calculate_Accuracy ()
    
    def initialize (self):
        self.classifier.add (Dense (activation="relu", input_dim=8, units=4, kernel_initializer="uniform"))
        self.classifier.add (Dense (activation="relu", units=4, kernel_initializer="uniform"))
        self.classifier.add (Dense (activation="relu", units=4, kernel_initializer="uniform"))
        self.classifier.add (Dense (activation="softmax", input_dim=8, units=7, kernel_initializer="uniform"))

    def predict (self):
        self.Y_predictedProbabilities = self.classifier.predict (self.X_testSet.array)
        l = len (self.Y_testSet)
        predictedMatrix = np.zeros ((l , 7))
        for i in range (len (self.Y_predictedProbabilities)):
            maximumIndex = np.argmax (self.Y_predictedProbabilities [i,:])
            predictedMatrix[i, maximumIndex] = 1

        self.decode_Prediction (predictedMatrix)

    def decode_Prediction (self, predictedMatrix):

        for i in range (len (self.Y_testSet)):
            for j in range (len (self.Y_testSet [0])):
                if self.Y_testSet [i,j] == 1:
                    self.decoded_Y_Test.append (j)
                if predictedMatrix [i,j] == 1:
                    self.Y_Predicted.append (j)

    def calculate_Accuracy (self):
        confusionMatrix = confusion_matrix (self.decoded_Y_Test , self.Y_Predicted)
        numberOfCorrectAnswers = 0
        for i in range (len (confusionMatrix)):
            for j in range (len (confusionMatrix [0])):
                if (i == j) :
                    numberOfCorrectAnswers += confusionMatrix [i,j]
        self.accuracy = numberOfCorrectAnswers / len (self.Y_testSet)


