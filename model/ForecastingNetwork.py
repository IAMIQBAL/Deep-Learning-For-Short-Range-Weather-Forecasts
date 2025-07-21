import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.eager.context import monitoring
import plot

class StackedNetwork(keras.models.Model):
    def __init__(self, nLayers, nFilters, kernelSize):
        super(StackedNetwork, self).__init__()
        # Change static args of layers to parameterized

        self.nLayers = nLayers
        self.nFilters = nFilters
        self.kernelSize = kernelSize
        self.convLSTMs = []
        self.batchNorms = []

        for i in range(self.nLayers):
            convLSTM = layers.ConvLSTM2D(filters = self.nFilters[i],
                                        kernel_size = self.kernelSize[i],
                                        padding = 'same',
                                        return_sequences = True,
                                        activation = 'relu')
            
            if i != self.nLayers - 1:
                bn = layers.BatchNormalization()
                self.batchNorms.append(bn)
            self.convLSTMs.append(convLSTM)

        self.convNN = layers.Conv3D(filters=1,
                                   kernel_size=(3,3,3),
                                   activation='sigmoid',
                                   padding='same')
    
    def call(self, inputTensor):
        x = self.convLSTMs[0](inputTensor)
        x = self.batchNorms[0](x)
        for i in range(1, self.nLayers):
            x = self.convLSTMs[i](x)
            if i != self.nLayers - 1:
                x = self.batchNorms[i](x)
        x = self.convNN(x)
        return x
    
def getModel(dims, nLayers, nFilters, kernelSize):
    inputLayer = tf.keras.Input(shape=(None, *dims.shape[2:]))
    x = StackedNetwork(nLayers, nFilters, kernelSize)(inputLayer)
    model = tf.keras.Model(inputs=inputLayer, outputs=x)
    print(model.summary(expand_nested=True))
    return model

def compileModel(model, loss, optimizer, metrics=['accuracy', 'mae', 'mse']):
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=[metrics]
    )
    return model

def train(model, data, batchSize, epochs, earlyStopping=10, reduceLR=5, weightsDir='ForecastingNW.h5'):
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=earlyStopping)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=reduceLR)
    x_train, y_train, x_val, y_val = data
    history = model.fit(
        x_train,
        y_train,
        batch_size=batchSize,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, reduce_lr]
    )
    model.save_weights(weightsDir)

    return history

def loadWeights(model, path):
    model.load_weights(path)
    return model

def predict(model, testData, recursive=True, cmap='p'):
    if cmap == 'p':
        cmap = plot.getPrecCMap()
    else:
        cmap = plot.getTempCMap()
    initCond = testData[:12, ...]
    actualPatches = testData[12:, ...]
    nonRec = np.zeros(shape=(12, *initCond[0].shape))
    if recursive:
        for _ in range(12):
            newPred = model.predict(np.expand_dims(initCond, axis=0))
            newPred = np.squeeze(newPred, axis=0)
            pred = np.expand_dims(newPred[-1, ...], axis=0)
            initCond = np.concatenate((initCond, pred), axis=0)

        fig, ax = plt.subplots(2, 12, figsize=(24, 6))
        plot.plotImageStackAxes(actualPatches, 2, 6, fig, ax, 0, 'Actual', cmap)
        predictions = initCond[12:, ...]
        plot.plotImageStackAxes(predictions, 2, 6, fig, ax, 1, 'Pred', cmap)
        # plt.show()
        return actualPatches, predictions
    else:
        for i in range(12):
            initCond = testData[:10+i+1, ...]
            newPred = model.predict(np.expand_dims(initCond, axis=0))
            newPred = np.squeeze(newPred, axis=0)
            pred = np.expand_dims(newPred[-1, ...], axis=0)
            nonRec = np.append(nonRec, pred, axis=0)
        fig, ax = plt.subplots(2, 12, figsize=(24, 6))
        plot.plotImageStackAxes(actualPatches, 2, 6, fig, ax, 0, 'Actual ', cmap)
        plot.plotImageStackAxes(nonRec[12:, ...], 2, 6, fig, ax, 1, 'Pred ', cmap)
        # plt.show()
        return actualPatches, nonRec[12:, ...]