#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 17:18:14 2020

@authors: suryapolina and ramkammari
"""
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from numpy import dot
from numpy.linalg import norm
from sklearn import svm
#make train_file into dictionary where key is health params and value is label 
def arrToDict(file):
    temp_dictionary = {}
    for data in file:
        temp_arr = []
        for j in range(len(data)-1):
            temp_arr.append(data[j])
        temp_dictionary[tuple(temp_arr)] = data[len(data)-1]
    return temp_dictionary

train_file = open("train.csv")
train_file = list(train_file)
train_file.remove(train_file[0])
train_dic = {}
count = 0
for data in train_file:
    train_file[count] = data.split(',')
    for i in range(len(train_file[count])):
        if train_file[count][i] == "NA":
            print(str(i) + "\n")
            train_file[count][i] = -1.0 #if data is NA
        else:
            train_file[count][i] = float(train_file[count][i])
    train_file[count].remove(train_file[count][14]) #removed glucose parameter
    count+=1


train_dic = arrToDict(train_file)

df = pd.DataFrame(train_dic.items(),columns=['Factors', 'Label'])

X = df['Factors']
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

#KNN
knn = KNeighborsClassifier(n_neighbors = 100)
knn.fit(list(X_train), y_train)
prediction = knn.predict(list(X_test))
score_knn = knn.score(list(X_test), y_test)

#decision tree #consistently 73%-78%
dt = DecisionTreeClassifier(criterion="gini", splitter='best')
dt.fit(list(X_train), y_train)
prediction_decision = dt.predict(list(X_test))
score_decision = dt.score(list(X_test), y_test)

#random forest
rf = RandomForestClassifier()
rf.fit(list(X_train), y_train)
prediction_random_forest = rf.predict(list(X_test))
score_random_forest = rf.score(list(X_test), y_test)

#Percepton
perc = Perceptron(random_state=3) #inconsistent ranges 20%-85%
perc.fit(list(X_train), y_train)
prediction_perceptron = perc.predict(list(X_test))
score_perceptron = perc.score(list(X_test), y_test)

#BernoulliNB
bnb = BernoulliNB()
bnb.fit(list(X_train), y_train)
prediction_bernoulli = bnb.predict(list(X_test))
score_bernoulli = bnb.score(list(X_test), y_test)

#GaussianNB
gnb = GaussianNB()
gnb.fit(list(X_train), y_train)
prediction_gaussian = gnb.predict(list(X_test))
score_gaussian = gnb.score(list(X_test), y_test)

#SVM
vector = svm.SVC()
vector.fit(list(X_train), y_train)
prediction_svm = vector.predict(list(X_test))
score_svm = vector.score(list(X_test), y_test)


