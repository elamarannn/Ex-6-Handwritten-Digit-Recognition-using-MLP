# Skill Assisessment-Handwritten Digit Recognition using MLP
## Aim:
To Recognize the Handwritten Digits using Multilayer perceptron.
##  EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook
## Theory:
Handwriting recognition is the ability of a computer to receive and interpret intelligible handwritten input from sources such as paper documents, photographs, touch-screens, and other devices. There are many techniques to that have been developed to recognize the handwriting. One of them is Handwritten Digit Recognition. In this project, we would be using Machine Learning classifier Multi-layer Perceptron Neural Network.

An MLP is a supervised machine learning (ML) algorithm that belongs in the class of feedforward artificial neural networks. The algorithm essentially is trained on the data in order to learn a function. Given a set of features and a target variable (e.g. labels) it learns a non-linear function for either classification or regression.

MLP classifier is a very powerful neural network model that enables the learning of non-linear functions for complex data. The method uses forward propagation to build the weights and then it computes the loss. Next, back propagation is used to update the weights so that the loss is reduced. This is done in an iterative way and the number of iterations is an input hyperparameter. Other important hyperparameters are the number of neurons in each hidden layer and the number of hidden layers in total. These need to be fine-tuned.

It can be seen that the machine learning model can recognize the hand written digits. Though the accuracy is about 83% but still it can be increased by using Convolution Neural Network or Support Vector Machine classifier of machine learning with proper tuning.

## Algorithm :
Step 1:Import the necessary libraries of python.

Step 2:In the end_to_end function, first calculate the similarity between the inputs and the peaks. Then, to find w used the equation Aw = Y in matrix form. Each row of A (shape: (4, 2)) consists of index [0]:

Step 3:Similarity of point with peak 1 index [1]: similarity of point with peak 2 index[2]: Bias input (1) Y: Output associated with the input (shape: (4, )) W is calculated using the same equation we use to solve linear regression using a closed solution (normal equation).

Step 4:This part is the same as using a neural network architecture of 2-2-1, 2 node input (x1, x2) (input layer) 2 node (each for one peak) (hidden layer) 1 node output (output layer).

Step 5:To find the weights for the edges to the 1-output unit. Weights associated would be: edge joining 1st node (peak1 output) to the output node edge joining 2nd node (peak2 output) to the output node bias edge.

Step 6:End the program.

## Program:
```
Developed By : Elamaran S E
Register no. : 212222230036

Program to Recognize the Handwritten Digits using Multilayer perceptron (MLP).

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape
Y_train

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2
    
def ReLU(Z):
    return np.maximum(Z, 0)
    
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2
    
def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
    
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2
    
def get_predictions(A2):
    return np.argmax(A2, 0)    
    
def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size
    
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)
    
def make_predictions(X, W1, b1, W2, b2):
   _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
   predictions = get_predictions(A2)
   return predictions
   
def test_prediction(index, W1, b1, W2, b2):
   current_image = X_train[:, index, None]
   prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
   label = Y_train[index]
   print("Prediction: ", prediction)
   print("Label: ", label)
   current_image = current_image.reshape((28, 28)) * 255
   plt.gray()
   plt.imshow(current_image, interpolation='nearest')
   plt.show()
   test_prediction(0, W1, b1, W2, b2)
   test_prediction(1, W1, b1, W2, b2)
   test_prediction(2, W1, b1, W2, b2)
   test_prediction(3, W1, b1, W2, b2)
   dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
   get_accuracy(dev_predictions, Y_dev)
```

## Output :
Y_TRAIN:

![280643855-c0baa896-f86e-4997-bbf7-febef35e3d30](https://github.com/elamarannn/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/113497531/0f7885d2-e020-4d38-9547-b72024d1cebd)

ITERATIONS FROM 1-100:

![280644148-dc8a1d4a-21ee-4cd8-bc43-bd3598eb683a](https://github.com/elamarannn/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/113497531/14aedf1e-f296-4f14-a517-c5f01a5f5e82)

ITERATIONS FROM 110-230:

![280644288-257e2346-7a7c-447d-a696-c72c58b0f4be](https://github.com/elamarannn/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/113497531/c899a0f3-b066-409e-86bd-b84f66325a3b)

ITERATIONS FROM 240-340:

![280644455-281f86ae-c7bb-47e5-9f5f-79a56fb1647f](https://github.com/elamarannn/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/113497531/cdd500c8-23ef-413a-9941-ab4e3e92101d)

ITERATIONS FROM 350-450:

![280644617-fc5fc009-5bac-47dc-921d-2355a16658c5](https://github.com/elamarannn/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/113497531/1c9e3b8c-035f-4b0c-aba3-7818ce2a10e2)

ITERATIONS FROM 460-490:

![280644799-0c4ffb5c-8255-4c53-a8cd-70d4b7b3fec2](https://github.com/elamarannn/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/113497531/1c079dab-8342-4f23-8ba0-c1fc490eb07e)

PREDICTING 5:

![280644927-52f9ce41-d557-44d4-8fca-d19686d41a52](https://github.com/elamarannn/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/113497531/033093e5-d9f9-4858-95d3-61c84c95a6d3)

PREDICTING 2:

![280645121-3f763a67-b265-45d0-b392-81a9079765a6](https://github.com/elamarannn/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/113497531/40907419-5873-4587-8efd-c477013d7043)

DEV_PREDICTIONS:

![280645333-5d8a704d-0375-474a-bb5b-d87b7beabb2a](https://github.com/elamarannn/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/113497531/ee7701b8-d640-4f28-ac2a-05aa019e7a11)

ACCURACY:

![280645506-8c8f0d3f-5e71-4f49-ab6f-e0333221d1a5](https://github.com/elamarannn/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/113497531/95efea42-f9e0-42f9-8347-acc04da80980)

RESULT:
Thus, the program to recognize the handwritten digits using the multi-layer perceptron (MLP) is developed and executted successfully.
















## Result:
