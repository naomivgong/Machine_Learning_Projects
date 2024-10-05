import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

#deep learning library
#can easily create nerual networks with tensorflow
import tensorflow as tf
from tensorflow import keras

breastCancerData = pd.read_csv('/Users/naomigong/Coding/ML Projects/breastCancerScreening/data.csv')
#create a dataframe from this data set

#preprocess the data like you did earlier
breastCancerData.drop(columns = 'Unnamed: 32', axis = 1, inplace = True)
breastCancerData.drop(columns = 'id', axis = 1, inplace = True)
breastCancerData.dropna(inplace = True)

labelencoder = LabelEncoder()
#The fit_transform method is a convenience method that combines two steps:
#Fit: Compute the unique values in the input data and map each unique value to a numerical value.
#Transform: Replace each value in the input data with its corresponding numerical value.
labels = labelencoder.fit_transform(breastCancerData['diagnosis'])
breastCancerData['target'] = labels
breastCancerData.drop(columns = 'diagnosis', axis = 1, inplace = True)


print(breastCancerData.columns)
print(breastCancerData.head())
#groups the target varaibles and finds the mean of each
print(breastCancerData.groupby('target').mean())

breastCancerDataInput = breastCancerData.drop(columns = 'target', axis = 1)
output = breastCancerData['target']

#splitting
xtrain, xtest, ytrain, ytest = train_test_split(breastCancerDataInput   , output , test_size= 0.2, random_state = 2)
print(xtrain.shape, xtrain.shape, xtest.shape)


#standardize data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
""""
fit_transform does two things:
Fit: Calculates the mean and standard deviation for each feature in the training data (xtrain).
Transform: Uses these statistics to standardize the training data.
"""
XtrainStandard = scaler.fit_transform(xtrain) #fitting our data
#transform uses the mena and std from xtrain to standardize
xtestStandard = scaler.transform(xtest)

'''''How building neural networks work:
- You have an input layer
- Hidden layer
    - You can have several input layer
-Output Layer

We will use tensorflow/keras to build our neural networks
'''''

'''''set seed for tenosr flow, whenever you are training a neural network in keras, every time you run neural network it may change
you will get same score everytime you run this. Makes our code replicable
'''''
tf.random.set_seed(3)

#setting up layers
#Sequential is where we will stack our neural networks
'''''Explain dense layers:
- each neuron in the layer takes the weighted sum of the inputs in previous layer
    - if their are n inputs and m neurons in dense layer, the layer will have m * n weights

- each neuron has a bias

- This weight + bias is passed through activation function, which introduces non-linearity

Activation functions:
- relu is the activation function used in hidden layers
- The sigmoid activation function maps any input value to a value between 0 and 1. 
Sigmoid is often used in the output layer of a binary classification network. (LIKE THIS ONE 0 or 1)
- The softmax activation function is used for multi-class classification problems. It converts the raw output scores (logits) into probabilities.
Softmax is typically used in the output layer of a neural network for multi-class classification.


'''
model = keras.Sequential([keras.layers.Flatten(input_shape = (30,)),   #first layer (input), basically just reshapes it to 1D array. The neurons in the input layer should equal number of features
                         keras.layers.Dense(20, activation = 'relu'),     #hidden layer <- each neuron in the previous input gives it to each in the next. 20 specifies num of neurons
                         
                         keras.layers.Dense(2, activation = 'sigmoid')
                         ])      #output layer. you want 2 neurons in the output layer. !!! the number of neurons in output layer = number of classifications

#keras.layers.Dense(30, activation = 'sigmoid'), you can add another layer

#compiling neural network
''''adam optimizer is an algorithm to update weights of neural networks
sparase_categorical_crossentropy -- used for multiclass classification with integer outputs
'''''
model.compile(optimizer = 'adam', loss = "sparse_categorical_crossentropy", metrics = ['accuracy'])

#training neural network and save it
history = model.fit(XtrainStandard,ytrain, validation_split = 0.1, epochs = 10) #validation_split means 10% of xtrain and ytrain
''''
notice how at each epoch, the loss decreases
the accuracy also increases 

- !!!!you can imporve the accuracy through standardization function after splitting OR adding more layers OR increase epochs!!!!
    - but make sure model does NOT overfit, this can occur when you add too many epochs or too many layers
    - standardization helps because the data values are so far apart so our model doesnt work well with it
'''''


                         
''''
#Visualizing Accuracy And Loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['training data', 'validation'], loc = 'lower right')
plt.show()
'''''

#printing out the accuracy and loss
loss, accuracy = model.evaluate(xtestStandard, ytest)
print("loss:  ", loss, "accuracy: ", accuracy)

print(xtestStandard.shape)
print(xtestStandard[0])

Y_pred = model.predict(xtestStandard)
print(Y_pred.shape)
print(Y_pred[0]) 
    #prints: [0.94405544 0.10783381]
'''' 
- index 0: probability of label 0
- index 1: probability of label 1
'''''

print(Y_pred)
#returns an array probabilities

#model.predict() gives prediction probability of each class for that datapoint

#converting prediction probability to class labels
#argmax function tells you the index of maximum value
#this iterates through the list picking the max between the first index or second index
y_pred_labels = [np.argmax(i) for i in Y_pred]
print(y_pred_labels)

#Building Predictive System
input_data = () #input the data
input_data_as_np = np.asarray(input_data)

#reshape the numpy array
input_data_reshaped = input_data_as_np.reshape(1, -1)

input_data_std = scaler.transform(input_data_reshaped)

prediction = model.prediction(input_data_std)
print(prediction)

prediction_label = [np.argmax(prediction)]
if (prediction_label[0] == 0):
    print("tumor is malignant")
else:
    print("the tumor is benign")
