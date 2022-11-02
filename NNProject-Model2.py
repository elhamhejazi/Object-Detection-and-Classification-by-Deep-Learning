from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, AveragePooling2D
from keras.models import Sequential
from keras.utils import plot_model
from matplotlib import pyplot, pyplot as plt
from keras.callbacks import EarlyStopping

trainData = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)
testData = ImageDataGenerator(rescale=1. / 255)

trainDir = './Dataset/train'
testDir = './Dataset/test'

trainSet = trainData.flow_from_directory(trainDir, target_size=(128, 128), batch_size=32, class_mode="categorical", subset='training')
validationSet = trainData.flow_from_directory(trainDir, target_size=(128, 128), batch_size=32, class_mode="categorical", subset='validation')
testSet = testData.flow_from_directory(testDir, target_size=(128, 128), batch_size=32, class_mode="categorical")

cnnModel = Sequential()

cnnModel.add(Conv2D(filters=32, kernel_size=5, input_shape=[128, 128, 3], activation="relu", padding="SAME"))
cnnModel.add(MaxPool2D())
# cnnModel.add(AveragePooling2D())

cnnModel.add(Conv2D(filters=64, kernel_size=5, activation="relu", padding="SAME"))
cnnModel.add(MaxPool2D())
# cnnModel.add(AveragePooling2D())

cnnModel.add(Conv2D(filters=128, kernel_size=3, activation="relu", padding="SAME"))
cnnModel.add(MaxPool2D())
# cnnModel.add(AveragePooling2D())

cnnModel.add(Flatten())

cnnModel.add(Dense(units=6, activation="relu"))
cnnModel.add(Dense(units=3, activation="softmax"))

cnnModel.summary()
plot_model(cnnModel, to_file='CNNModel.png', show_shapes=True, show_layer_names=True)
# plot_model(cnnModel, "CNN_Model.png", show_shapes=True)
# es = EarlyStopping(monitor="val_accuracy", patience=8, verbose=1)
# miniBatches = [16, 32, 64, 128, 256]
# miniBatches = [256]
# for m in miniBatches:
# print('\n\nFor Batch Size=', m)
cnnModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history= cnnModel.fit(trainSet, validation_data=validationSet, epochs=100, callbacks=[es], batch_size=128, verbose=1)
accuracyV = cnnModel.evaluate(validationSet)
accuracyTest = cnnModel.evaluate(testSet)
# print("Validation", accuracyV)
# print("test loss, test acc:", accuracyTest)
#
# plt.title('CNN Model')
#
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
#
# plt.ylabel('Accuracy')
# plt.xlabel('Epochs')
#
# plt.legend(['training', 'validation', 'test'], loc='best')
# plt.show()
# acc= history.history['accuracy'][-1]
# print(f'Accuracy for Training: {acc:.3}')
# acc= history.history['val_accuracy'][-1]
# print(f'Accuracy for test: {acc:.3}')


