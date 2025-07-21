import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2

def getPrecCMap():
    cmapPrecp = ListedColormap(['#DA00FF', '#FFA400', '#FF0000',
                            '#0080FF', '#00FFFF', '#FFFFFF'])
    return cmapPrecp

def getTempCMap():
    cmapTmp = ListedColormap(['#000080', '#0000D9', '#4000FF', '#8000FF', '#0080FF', '#00FFFF', '#00FF80', 
                            '#80FF00', '#DAFF00', '#FFFF00', '#FFF500', '#FFDA00', '#FFB000', '#FFA400',
                            '#FF4F00', '#FF2500', '#FF0A00', '#FF00FF'])
    return cmapTmp

def readImage(path):
    image = cv2.imread(path)
    return image

def readGrayImage(path):
    image = cv2.imread(path, cv2.COLOR_BGR2BGRA)
    return image

def plotImage(image, cmap=None):
    plt.imshow(image, cmap=cmap)
    plt.show()

def plotImageStack(index, data, w=4, h=6, cmap=None):
    fig, axes = plt.subplots(w, h, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.squeeze(data[index, i]), cmap=cmap)
        ax.set_title(f'Index {i}')
        ax.axis('off')
    plt.show()

def plotImageStackAxes(data, w, h, fig, ax, axIndex, title, cmap=None):
    for i, ax in enumerate(ax[axIndex]):
        ax.imshow(np.squeeze(data[i]), cmap=cmap)
        ax.set_title(f'{title} Index {i + 12}')
        ax.axis('off')