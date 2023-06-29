#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 07:56:33 2023

@author: GayatriBhamburkar
"""

from sklearn.naive_bayes import GaussianNB


from sklearn import metrics
import pandas as pd
import numpy as np

NaiveBayes = GaussianNB()


df = pd.read_csv("C:/Users/GAYATRI/Downloads/Crop_Recommdation_Project/Crop_recommendation.csv")


features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']




from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)


NaiveBayes.fit(Xtrain.values,Ytrain.values)

predicted_values = NaiveBayes.predict(Xtest.values)
x = metrics.accuracy_score(Ytest.values, predicted_values)

print("Naive Bayes's Accuracy is: ", x*100)



#[83, 45, 60, 28, 70.3, 7.0, 150.9]
#[104,18, 30, 23.603016, 60.3, 6.7, 140.91]

data = np.array([[104,18, 30, 23.603016, 60.3, 6.7, 140.91]])
prediction = NaiveBayes.predict(data)
print(prediction)