# Object-Detection-and-Classification-by-Deep-Learning

Performance evaluation of the CNNs as feature extractors, as well as different classifiers for an object classification system\
Presented to Prof. Alioune Ngom\
Elhamsadat Hejazi(https://www.linkedin.com/in/elham-hejazi)

## Overall Context

## Introduction 
Deep learning has become a major machine learning technology in computer vision and image analysis and has emerged as a powerful method. 

In this work:
I present an overview of image classification approaches, focusing on the categorization of three types of vehicles (car, airplane, and ship). 
I provide a quick introduction to convolutional neural networks and outline their applications in object categorization and feature extraction. 
In addition, I examine the challenges and potential future applications of deep learning in picture classification.
Image Classification Algorithms:

## Dataset
This Dataset is taken from Kaggle and contains 3 classes namely Airplane, Cars, and Ships with 1000 samples for each class.
The total number of training data is 3000, which is split into training data (80%) and validation data (20%).
The total number of test data is 582.
The shape of the images is 128 * 128.

## Platform, Toolkit, Software
Language programing: Python (Numpy, MatplotLib, Keras)
TensorFlow 2.0
Windows11
Pycharm/Jupyter Notebook

## Preprocessing and Spiliting Data
normalizing the values of the pixels fall between 0 and 255. Thus, in order to scale each value between 0 and 1, it is imperative to divide each by 255.
Applying cross-validation on the dataset by splitting to training, validation, and test.

## Proposed Models
Image classification is used to take the features from images and extract them in order to detect patterns in the dataset. As the trainable parameters of ANN are very vast and would require expensive computation, Convolutional Neural Network (CNN) is utilized in our model because it decreases the parameters without harming the quality of the models.
A CNN architecture consists of two fundamental components. Feature extraction is a procedure used by a convolution tool to separate and identify the distinct characteristics of a picture for analysis. a fully connected layer that makes use of the convolutional process's output and determines the class of the image using the features that were previously extracted.
The proposed models included two fully linked layers for classification along with 3 or 4 convolutional layers, respectively.

## CNN Model 1
It consists of 5 layers. The first layer consists of an input image with dimensions of 128×128. It is convolved with 32 filters of size 5×5. The Pooling operation which a filter size of 2×2 and a stride of 2. Hence the resulting image dimension will be 64×64×32.
Similarly, the second layer is involved in a convolution operation with 64 filters of size 5×5 followed by a pooling layer with a similar filter size of 2×2 and stride of 2. Thus, the resulting image dimension will be reduced to 32×32×64.
The third layer is also involved in a convolution operation with 128 filters of size 3×3 followed by a pooling layer with a similar filter size of 2×2 and stride of 2. Thus, the resulting image dimension will be reduced to 16×16×128.
Once the image dimension is reduced, the fourth layer is a fully connected convolutional layer with 6 filters. The fifth layer is also a fully connected layer with 3 units.

## CNN Model 2
It consists of 6 layers. The first layer consists of an input image with dimensions of 128×128. It is convolved with 32 filters of size 5×5. The Pooling operation which a filter size of 2×2 and a stride of 2. Hence the resulting image dimension will be 64×64×32.
Similarly, the second layer is involved in a convolution operation with 64 filters of size 3×5 followed by a pooling layer with a similar filter size of 2×2 and stride of 2. Thus, the resulting image dimension will be reduced to 32×32×64.
The third layer is also involved in a convolution operation with 64 filters of size 3×3 followed by a pooling layer with a similar filter size of 2×2 and stride of 2. Thus, the resulting image dimension will be reduced to 16×16×64.
The fourth layer is also involved in a convolution operation with 64 filters of size 3×3 followed by a pooling layer with a similar filter size of 2×2 and stride of 2. Thus, the resulting image dimension will be reduced to 8×8×64.
Once the image dimension is reduced, the fifth layer is a fully connected convolutional layer with 6 filters. The sixth layer is also a fully connected layer with 3 units.

## Implementation Details-Model 1
The first model includes 3 layers of convolution and max-pooling along with 2 fully connected. 
Generating data from the dataset and spilling it into training, validation, and test.
Applying convolutional neural networks by selecting the different activation functions for different layers (optimal: Relu and softmax). 
Examining the model with various parameters such as different filter sizes, kernel sizes, and pooling to optimize the model.
Evaluating the model with four different optimizers (AdaDelta, Adam, SGD, and RMSprop) and different batch sizes. 
Calculating accuracy and loss metrics for each optimizer.
The Adam optimizer is the most efficient with a batch size of 128.

## Implementation Details-Model 2
The first model includes 4 layers of convolution and max-pooling along with 2 fully connected. 
Generating data from the dataset and spilling it into training, validation, and test.
Applying convolutional neural networks by selecting the different activation functions for different layers (optimal: Relu and softmax). 
Examine the model with various parameters such as different filter sizes, kernel sizes, and pooling to optimize the model. (Filter: 32 (the best accuracy), 64, 128 (the worst accuracy)) 
Arranging layers by filters 32 and 64 and obtaining the best accuracy by filters 32,64,64,64.
Evaluating the model with different kernel sizes. The most efficient accuracy by kernel sizes 3,5,5,3.
Applying different batch sizes that play a vital role in the model. The batch size 64 performs the best among different batch sizes and I chose it for our model.
replacing average pooling instead of max-pooling and different combination. Regarding the picture, max-pooling in all layers performs better.

## Results and Conclusion
The validation accuracy and loss for both models are shown below:
Model 1:
Validation Loss: 0.4118146300315857,
Validation Accuracy: 0.9089999856948853

Model 2:
Validation Loss: 0.184452623128891,
Validation Accuracy: 0.9383333325386047

The test loss and accuracy after fitting the models are as follow:
Model 1 Test Loss: 0.6389545202255249,
Model 1 Test Accuracy: 0.896288652420044,
Model 2 Test Loss: 0.2377959042787552,
Model 2 Test Accuracy: 0.9295532703399658

Additionally, VGG16, and Resnet50 are used to compare the complexity of our models. Resnet50 has 23888771, model 1 has 324187 parameters, model 2 has 216155 parameters, and VGG16 has 14789955 total parameters. Thus, Model 2 has the least amount of complexity.
As a result, I used the fundamental CNN structure, their architecture, and the various layers that comprise various CNN models in this project. Additionally, I evaluate the accuracy metrics for two additional models, VGG16 and Resnet50, to determine that both of our models outperform Resnet50, but VGG16 performs the best. Changing different parameters that affect the CNN should represent much more than optimality.

