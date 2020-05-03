#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 17:18:14 2020

@author: suryapolina
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

train_file = open("train.csv")
train_file = list(train_file)
train_file.remove(train_file[0])
count = 0
for data in train_file:
    train_file[count] = data.split(',')
    for i in range(len(train_file[count])):
        if train_file[count][i] == "NA":
            train_file[count][i] = -1.0 #if data is NA
        else:
            train_file[count][i] = float(train_file[count][i])
    count+=1

#df = pd.DataFrame(train_dic.items(),columns=['Drug', 'Active'])
#
#X = df['Factors']
#y = df['Positive']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

