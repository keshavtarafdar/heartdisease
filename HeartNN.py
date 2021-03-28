# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 18:43:37 2021

@author: Keshav Tarafdar
"""

import keras
from keras import models
from keras.models import Sequential
from keras.layers import Dense
from keras import layers
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy as np

split = 181

def printFile(dic):
    
    for data in dic:
        print('%22s: %8.5f' % (data, dic.get(data)))

def normalize(x):

    stand_x = (x - np.mean(x, axis = 0))
    stand_x /= (np.max(x, axis = 0) - np.min(x, axis = 0))
    
    return stand_x

def standardize(x):
    
    stand_x = (x - np.nanmean(x, axis = 0))
    stand_x /= np.nanstd(x)
    
    return stand_x

fileName = 'heart.csv'
print("fileName: ", fileName)
raw_data = open(fileName, 'rt')
data = np.loadtxt(raw_data, usecols = (0,1,2,3,4,5,6,7,8,9,10,11,12,13), skiprows = 1, delimiter=",", dtype=np.float)

def createConfusion(yPred, yActual):
    
    truePos = 0
    trueNeg = 0
    falsePos = 0
    falseNeg = 0
    precision = 0
    accuracy = 0
    error_rate = 0
    recall = 0
    total = len(yPred)
    
    for pred, actual in zip(yPred, yActual):
        if pred == actual:
            if pred == 0:
                trueNeg += 1
            else:
                truePos += 1
        else:
            if pred == 0:
                falseNeg += 1
            else:
                falsePos += 1
    
    accuracy = (trueNeg + truePos) / total
    precision = (truePos) / (truePos + falsePos)
    recall = (truePos) / (truePos + falseNeg)
    error_rate = (falseNeg + falsePos) / total                
                
    return truePos, trueNeg, falsePos, falseNeg, accuracy, precision, recall, error_rate

def readTrain():
     
    fileName = 'heart.csv'
    print("fileName: ", fileName)
    raw_data = open(fileName, 'rt')
    data = np.loadtxt(raw_data, usecols = (0,1,2,3,4,5,6,7,8,9,10,11,12,13), skiprows = 1, delimiter=",", dtype=np.float)
    
    x = data[:, :13]
    y = data[:, 13:]
    
    x[:,0] = standardize(x[:,0])
    x[:,2] = normalize(x[:,2])
    x[:,3] = normalize(x[:,3])
    x[:,4] = standardize(x[:,4])
    x[:,6] = normalize(x[:,6])
    x[:,7] = standardize(x[:,7])   
    x[:,9] = normalize(x[:,9])
    x[:,10] = normalize(x[:,10])
    x[:,11] = normalize(x[:,11])
    x[:,12] = normalize(x[:,12])
    
    # bias
    bias = np.ones((len(x),1))
    x = np.concatenate((bias, x), axis = 1)
    
    # split data into test & train, 75/25 split
    x = x[:split, :]
    y = y[:split, :]
    
    return x, y

def readTest():
    
    fileName = 'heart.csv'
    print("fileName: ", fileName)
    raw_data = open(fileName, 'rt')
    data = np.loadtxt(raw_data, usecols = (0,1,2,3,4,5,6,7,8,9,10,11,12,13), skiprows = 1, delimiter=",", dtype=np.float)
    
    x = data[:, :13]
    y = data[:, 13:]

    x[:, 0] = standardize(x[:,0])
    x[:, 2] = normalize(x[:,2])
    x[:, 3] = normalize(x[:,3])
    x[:, 4] = standardize(x[:,4])
    x[:, 6] = normalize(x[:,6])
    x[:, 7] = standardize(x[:,7])   
    x[:, 9] = normalize(x[:,9])
    x[:, 10] = normalize(x[:,10])
    x[:, 11] = normalize(x[:,11])
    x[:, 12] = normalize(x[:,12])
    
    # bias
    bias = np.ones((len(x),1))
    x = np.concatenate((bias, x), axis = 1)
    
    # split data into test & train, 75/25 split
    x = x[split:, :]
    y = y[split:, :]
    
    return x, y

# calculate binary cross entropy
def binary_cross_entropy(actual, predicted):
    sum_score = 0.0
    for i in range(len(actual)):
        sum_score += actual[i] * np.log(1e-15 + predicted[i])
        mean_sum_score = 1.0 / len(actual) * sum_score
    return -mean_sum_score

def createCostList(actual, pred):
    cost_arr = []
    for i in range(len(actual)):
        cost_arr.append(binary_cross_entropy(actual[i], pred[i]))
    return cost_arr



xTrain, yTrain = readTrain()
xTest, yTest = readTest()

trainData = np.concatenate((xTrain,yTrain),axis=1)
testData = np.concatenate((xTest,yTest),axis=1)

model = Sequential()
model.add(Dense(16, input_shape=(split,14), activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(xTrain, yTrain, epochs = 135, batch_size = 128)

predProb = model.predict(xTest)
predClass = model.predict_classes(xTest)

tp, tn, fp, fn, acc, pr, rec, err = createConfusion(predClass,yTest)

cost_arr = createCostList(yTest, predClass)

fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8]) # [left, bottom, width, height]
    
ax.plot(range(120), cost_arr)
ax.set_xlabel("Iteration")
ax.set_ylabel("Cost")
ax.legend()

model.summary()

analysis = {'Accuracy': acc,
            'Error Rate': err,
            'Precision': pr,
            'Recall' : rec}

print("Will they have heart disease?") 
print()
print("n = " + str(len(predClass)))
print()
print('%35s %15s' % ("Predicted No", "Predicted Yes"))
print()
print('%20s %15s %15s' % ("Actual Yes", fn, tp))
print()
print('%20s %15s %15s' % ("Actual No", tn, fp))
print()
print()
printFile(analysis)