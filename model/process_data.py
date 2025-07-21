import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def trainValSplit(data, ratio):
    index = np.arange(data.shape[0])
    np.random.shuffle(index)
    trainIndex = index[:int(ratio * data.shape[0])]
    valIndex = index[int(ratio * data.shape[0]):]
    trainData = data[trainIndex]
    validationData = data[valIndex]
    return trainData, validationData

def normalize(data):
    return data/255

def patchShift(data):
    x = data[:, 0:data.shape[1]-1, ...]
    y = data[:, 1:data.shape[1], ...]
    return x, y

def processData(path, ratio=1, norm=True):
    data = np.load(path)
    
    print('Dimensions: ', data.ndim)
    print('Shape: ', data.shape)
    print('Type: ', data.dtype)

    if ratio < 1:
        data = np.expand_dims(data, axis=-1)
        trainData, valData = trainValSplit(data, ratio)
        trainData = normalize(trainData)
        valData = normalize(valData)

        xTrain, yTrain = patchShift(trainData)
        xVal, yVal = patchShift(valData)

        return xTrain, yTrain, xVal, yVal
    else:
        data = np.expand_dims(data, axis=-1)
        if norm:
            data = normalize(data)
        return data