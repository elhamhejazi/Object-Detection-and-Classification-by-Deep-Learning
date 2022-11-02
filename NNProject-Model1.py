import os

import tf as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, AveragePooling2D
from keras.models import Sequential
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from matplotlib import pyplot, pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import PIL.Image
from tensorflow.python import tf2
import scipy


trainData = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)
testData = ImageDataGenerator(rescale=1. / 255)

trainDir = './Dataset/train'
testDir = './Dataset/test'

trainSet = trainData.flow_from_directory(trainDir, target_size=(128, 128), batch_size=32, class_mode="categorical", subset='training')
validationSet = trainData.flow_from_directory(trainDir, target_size=(128, 128), batch_size=32, class_mode="categorical", subset='validation')
testSet = testData.flow_from_directory(testDir, target_size=(128, 128), batch_size=32, class_mode="categorical")

cnnModel = Sequential()

cnnModel.add(Conv2D(filters=32, kernel_size=3, input_shape=[128, 128, 3], activation="relu", padding="SAME"))
cnnModel.add(MaxPool2D())
cnnModel.add(Conv2D(filters=64, kernel_size=5, activation="relu", padding="SAME"))
cnnModel.add(MaxPool2D())
cnnModel.add(Conv2D(filters=64, kernel_size=5, activation="relu", padding="SAME"))
cnnModel.add(MaxPool2D())
cnnModel.add(Conv2D(filters=64, kernel_size=3, activation="relu", padding="SAME"))
cnnModel.add(MaxPool2D())

cnnModel.add(Flatten())

cnnModel.add(Dense(units=6, activation="relu"))
cnnModel.add(Dense(units=3, activation="softmax"))

cnnModel.summary()
es = EarlyStopping(monitor="val_accuracy", patience=8, verbose=1)
cnnModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history= cnnModel.fit(trainSet, validation_data=validationSet, epochs=100, callbacks=[es], batch_size=64, verbose=1)
accuracyV = cnnModel.evaluate(validationSet)
accuracy = cnnModel.evaluate(testSet)
print("Validation", accuracyV)
print("test loss, test acc:", accuracy)



