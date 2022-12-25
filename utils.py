import cv2
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import random

from keras.models import Sequential
from keras.layers import Convolution2D,Flatten,Dense
from keras.optimizers import Adam

def getName(filePath):
    return filePath.split('\\')[-1]

def importDataInfo(path):
    columns = ['Center','Left','Right','Steering','Throttle','Brake','Speed']
    data = pd.read_csv(os.path.join(path,'driving_log.csv'),names=columns)
    #print(data.head())   
    #print(data['Center'][0])  
    #print(getName(data['Center'][0]))
    data['Center'] = data['Center'].apply(getName)
    print(data.head())
    print('Total Images Imported:', data.shape[0])
    return data

def balanceData(data,display=True):
    nBins = 31
    samplesPerBin = 1000
    hist, bins=np.histogram(data['Steering'],nBins)
    #print(bins)
    if display:
        center = (bins[:-1]+bins[1:])*0.5
        #print(center)
        plt.bar(center,hist,width=0.06)
        plt.plot((-1,1),(1000,1000))
        plt.show()

    removeIndexList = []
    for j in range(nBins):
        binDataList = []
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j+1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeIndexList.extend(binDataList)
    print('Removed Images: ',len(removeIndexList))
    data.drop(data.index[removeIndexList],inplace = True)
    print('Remaining Images: ',len(data))

    if display:
        hist, _ = np.histogram(data['Steering'], nBins)
        plt.bar(center,hist,width = 0.06)
        plt.plot((-1,1),(samplesPerBin,samplesPerBin))
        plt.show()

    return data

def loadData(path,data):
    imagesPath=[]
    steering = []

    for i in range (len(data)):
        indexedData = data.iloc[i]
        #print(indexedData)
        imagesPath.append(os.path.join(path,'IMG',indexedData[0]))
        steering.append(float(indexedData[3]))
    imagesPath = np.asarray(imagesPath)
    steering = np.asarray(steering)
    return imagesPath,steering

def augmentImage(imgPath,steering):
    img = mpimg.imread(imgPath)
    ##PAN
    if np.random.rand():
        pan = iaa.Affine(translate_percent={'x':(-0.1,0.1),'y':(-0.1,0.1)})
        img = pan.augment_image(img)

    ##ZOOM
    if np.random.rand():
        zoom = iaa.Affine(scale=(1,1.2))
        img = zoom.augment_image(img)

    ##BRIGHTNESS
    if np.random.rand():
        brightness = iaa.Multiply((0.4,1.2))
        img = brightness.augment_image(img)

    ##FLIP
    if np.random.rand():
        img = cv2.flip(img,1)
        steering = -steering


    return img, steering

#imgRe, st = augmentImage('test.jpg',0)
#plt.imshow(imgRe)
#plt.show()

def preProcessing(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.resize(img,(200,66))
    img = img/255
    return img


imgRe = preProcessing(mpimg.imread('test.jpg'))
plt.imshow(imgRe)
plt.show()

def batchGen(imagesPath, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []
        for i in range(batchSize):
            index = random.randint(0,len(imagesPath)-1)
            if trainFlag:
                img, steering = augmentImage(imagesPath[index],steeringList[index])
            else:
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]
            img = preProcessing(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield (np.asarray(imgBatch),np.asarray(steeringBatch))

def creatModel():
    model = Sequential()

    model.add(Convolution2D(24,(5,5),(2,2),input_shape=(66,200,3),activation='elu'))
    model.add(Convolution2D(36,(5,5),(2,2),activation='elu'))
    model.add(Convolution2D(48,(5,5),(2,2),activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))

    model.add(Flatten())
    model.add(Dense(100,activation='elu'))
    model.add(Dense(50,activation='elu'))
    model.add(Dense(10,activation='elu'))
    model.add(Dense(1))

    model.compile(Adam(lr=0.0001),loss='mse')

    return model








