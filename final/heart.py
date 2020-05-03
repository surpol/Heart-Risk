#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 17:18:14 2020

@author: suryapolina ramkammari
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

train_file = open("train.csv")
train_file = list(train_file)
train_file.remove(train_file[0])
train_dic = {}
count = 0
for data in train_file:
    train_file[count] = data.split(',')
    for i in range(len(train_file[count])):
        if train_file[count][i] == "NA":
            train_file[count][i] = -1.0 #if data is NA
        else:
            train_file[count][i] = float(train_file[count][i])
    count+=1

#train_file into dictionary where value is label and keys are health factors
def arrToDict(file):
    temp_dictionary = {}
    for data in file:
        temp_arr = []
        for j in range(len(data)-1):
            temp_arr.append(data[j])
        temp_dictionary[tuple(temp_arr)] = data[len(data)-1]
    return temp_dictionary

train_dic = arrToDict(train_file)

df = pd.DataFrame(train_dic.items(),columns=['Factors', 'Label'])

X = df['Factors']
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

