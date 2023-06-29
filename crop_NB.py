#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 07:56:33 2023

@author: GayatriBhamburkar
"""
#importing libraries
#We are importing pandas library as pd
import pandas as pd

#we are importing Gaussian Naive Bayes Classifier 
from sklearn.naive_bayes import GaussianNB

#To import train_test_split
from sklearn.model_selection import train_test_split

#metrics can be imported by using the following line
from sklearn import metrics

#we are importing numpy as np
import numpy as np

#making NaiveBayes as an object of GaussianNB()
NaiveBayes = GaussianNB()

#Importing the dataset from the local storage
df = pd.read_csv("C:/Users/GAYATRI/Downloads/Crop_Recommdation_Project/Crop_recommendation.csv")

'''Dividing the dataset into dependant and independent variables by using the following lines 
features is the independant variable target is the dependant variable'''
features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']

#Spliting the dataset into training and testing datasets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)

#Fitting the Model with the training datasets
NaiveBayes.fit(Xtrain.values,Ytrain.values)

#We are predicting the output on the basis of the value which is passed in the predict function
predicted_values = NaiveBayes.predict(Xtest.values)

'''Accuracy of the model can be checked by comparing the output dataframe with the actual dataframe result
it can be done by using the following lines'''
x = metrics.accuracy_score(Ytest.values, predicted_values)

#printing the accuracy of the model
print("Naive Bayes's Accuracy is: ", x*100)

#[83, 45, 60, 28, 70.3, 7.0, 150.9]
#[104,18, 30, 23.603016, 60.3, 6.7, 140.91]

#creating the numpy array of the desired values
data = np.array([[104,18, 30, 23.603016, 60.3, 6.7, 140.91]])

##Predicting the output on the basis of the numpy array which is passed to the predict function
prediction = NaiveBayes.predict(data)

#printing the output
print(prediction)