# **Artificial Neural Networks Project: Street View Housing Number Digit Recognition**

# **Marks: 30**

Welcome to the project on classification using Artificial Neural Networks. We will work with the Street View Housing Numbers (SVHN) image dataset for this project.

--------------
## **Context** 
--------------

One of the most interesting tasks in deep learning is to recognize objects in natural scenes. The ability to process visual information using machine learning algorithms can be very useful as demonstrated in various applications.

The SVHN dataset contains over 600,000 labeled digits cropped from street-level photos. It is one of the most popular image recognition datasets. It has been used in neural networks created by Google to improve the map quality by automatically transcribing the address numbers from a patch of pixels. The transcribed number with a known street address helps pinpoint the location of the building it represents. 

----------------
## **Objective**
----------------

To build a feed-forward neural network model that can recognize the digits in the images. 

-------------
## **Dataset**
-------------
Here, we will use a subset of the original data to save some computation time. The dataset is provided as a .h5 file. The basic preprocessing steps have been applied on the dataset.

## **Mount the drive**

Let us start by mounting the Google drive. You can run the below cell to mount the Google drive.


```python
from google.colab import drive

drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).


## **Importing the necessary libraries**


```python
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization

from tensorflow.keras.losses import categorical_crossentropy

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import to_categorical
```

**Let us check the version of tensorflow.**


```python
print(tf.__version__)
```

    2.11.0


## **Load the dataset**

- Let us now load the dataset that is available as a .h5 file.
- Split the data into the train and the test dataset.


```python
import h5py

# Open the file as read only
# User can make changes in the path as required

h5f = h5py.File('/content/drive/MyDrive/SVHN_single_grey1.h5', 'r')

# Load the training and the test dataset

X_train = h5f['X_train'][:]

y_train = h5f['y_train'][:]

X_test = h5f['X_test'][:]

y_test = h5f['y_test'][:]


# Close this file

h5f.close()
```

Let's check the number of images in the training and the testing dataset.


```python
len(X_train), len(X_test)
```




    (42000, 18000)



**Observation:**
- There are 42,000 images in the training data and 18,000 images in the testing data. 

## **Visualizing images**

- Use X_train to visualize the first 10 images.
- Use Y_train to print the first 10 labels.


```python
# Visualizing the first 10 images in the dataset and printing their labels

plt.figure(figsize = (10, 1))

for i in range(10):

    plt.subplot(1, 10, i+1)

    plt.imshow(X_train[i], cmap = "gray")

    plt.axis('off')

plt.show()

print('label for each of the above image: %s' % (y_train[0:10]))
```


    
![png](output_13_0.png)
    


    label for each of the above image: [2 6 7 4 4 0 3 0 7 3]


## **Data preparation**

- Print the shape and the array of pixels for the first image in the training dataset.
- Reshape the train and the test dataset because we always have to give a 4D array as input to CNNs.
- Normalize the train and the test dataset by dividing by 255.
- Print the new shapes of the train and the test dataset.
- One-hot encode the target variable.


```python
# Shape and the array of pixels for the first image

print("Shape:", X_train[0].shape)

print()

print("First image:\n", X_train[0])
```

    Shape: (32, 32)
    
    First image:
     [[ 33.0704  30.2601  26.852  ...  71.4471  58.2204  42.9939]
     [ 25.2283  25.5533  29.9765 ... 113.0209 103.3639  84.2949]
     [ 26.2775  22.6137  40.4763 ... 113.3028 121.775  115.4228]
     ...
     [ 28.5502  36.212   45.0801 ...  24.1359  25.0927  26.0603]
     [ 38.4352  26.4733  23.2717 ...  28.1094  29.4683  30.0661]
     [ 50.2984  26.0773  24.0389 ...  49.6682  50.853   53.0377]]



```python
# Reshaping the dataset to flatten them. We are reshaping the 2D image into 1D array

X_train = X_train.reshape(X_train.shape[0], 1024)

X_test = X_test.reshape(X_test.shape[0], 1024)
```

### **Question 1: Normalize the train and the test data. (2 Marks)**


```python
# Normalize inputs from 0-255 to 0-1
# As this an image dataset and pixel values go from 0-255, we divide all pixels
# by 255 to normalize and get a value between 0 - 1
X_train = X_train.astype('float32')/255.0

X_test = X_test.astype('float32')/255.0
```


```python
# New shape 

print('Training set:', X_train.shape, y_train.shape)

print('Test set:', X_test.shape, y_test.shape)
```

    Training set: (42000, 1024) (42000,)
    Test set: (18000, 1024) (18000,)



```python
# One-hot encode output

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)

# Test labels

y_test
```




    array([[0., 1., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 1., 0., 0.],
           [0., 0., 1., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 1., 0., 0.],
           [0., 0., 0., ..., 0., 0., 1.],
           [0., 0., 1., ..., 0., 0., 0.]], dtype=float32)



**Observation:**
- Notice that each entry of the target variable is a one-hot encoded vector instead of a single label.

## **Model Building**

Now that we have done the data preprocessing, let's build an ANN model.


```python
# Fixing the seed for random number generators

np.random.seed(42)

import random

random.seed(42)

tf.random.set_seed(42)
```

### **Model Architecture**
- Write a function that returns a sequential model with the following architecture:
 - First hidden layer with **64 nodes and the relu activation** and the **input shape = (1024, )**
 - Second hidden layer with **32 nodes and the relu activation**
 - Output layer with **activation as 'softmax' and number of nodes equal to the number of classes, i.e., 10**
 - Compile the model with the **loss equal to categorical_crossentropy, optimizer equal to Adam(learning_rate = 0.001), and metric equal to 'accuracy'**. Do not fit the model here, just return the compiled model.
- Call the nn_model_1 function and store the model in a new variable. 
- Print the summary of the model.
- Fit on the train data with a **validation split of 0.2, batch size = 128, verbose = 1, and epochs = 20**. Store the model building history to use later for visualization.

### **Question 2: Build and train an ANN model as per the above mentioned architecture. (10 Marks)**


```python
# Define the model

def nn_model_1():

    model = Sequential() 

    # Add layers as per the architecture mentioned above in the same sequence
    
    model.add(Dense(64, activation ='relu', input_shape = (1024, )))

    model.add(Dense(32, activation ='relu'))

    model.add(Dense(10, activation ='softmax')) #10 since we numbers from 0-9
  
    opt = Adam(learning_rate=0.001)
    # Compile the model

    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics=['accuracy'] )
    
    return model
```


```python
# Build the model

model_1 = nn_model_1()
```


```python
# Print the summary

model_1.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense_6 (Dense)             (None, 64)                65600     
                                                                     
     dense_7 (Dense)             (None, 32)                2080      
                                                                     
     dense_8 (Dense)             (None, 10)                330       
                                                                     
    =================================================================
    Total params: 68,010
    Trainable params: 68,010
    Non-trainable params: 0
    _________________________________________________________________



```python
# Fit the model

history_model_1 = model_1.fit(X_train, y_train, validation_split=0.2, batch_size=128, verbose=1, epochs=20)

```

    Epoch 1/20
    263/263 [==============================] - 3s 9ms/step - loss: 2.2934 - accuracy: 0.1233 - val_loss: 2.2324 - val_accuracy: 0.1744
    Epoch 2/20
    263/263 [==============================] - 3s 11ms/step - loss: 2.0774 - accuracy: 0.2509 - val_loss: 1.9318 - val_accuracy: 0.3089
    Epoch 3/20
    263/263 [==============================] - 3s 13ms/step - loss: 1.8679 - accuracy: 0.3405 - val_loss: 1.7779 - val_accuracy: 0.3869
    Epoch 4/20
    263/263 [==============================] - 3s 12ms/step - loss: 1.6573 - accuracy: 0.4391 - val_loss: 1.5288 - val_accuracy: 0.4907
    Epoch 5/20
    263/263 [==============================] - 2s 9ms/step - loss: 1.4635 - accuracy: 0.5180 - val_loss: 1.4202 - val_accuracy: 0.5255
    Epoch 6/20
    263/263 [==============================] - 2s 6ms/step - loss: 1.3588 - accuracy: 0.5568 - val_loss: 1.3240 - val_accuracy: 0.5768
    Epoch 7/20
    263/263 [==============================] - 2s 6ms/step - loss: 1.2895 - accuracy: 0.5839 - val_loss: 1.2558 - val_accuracy: 0.6013
    Epoch 8/20
    263/263 [==============================] - 2s 7ms/step - loss: 1.2375 - accuracy: 0.6044 - val_loss: 1.2217 - val_accuracy: 0.6221
    Epoch 9/20
    263/263 [==============================] - 2s 7ms/step - loss: 1.1990 - accuracy: 0.6206 - val_loss: 1.1674 - val_accuracy: 0.6330
    Epoch 10/20
    263/263 [==============================] - 3s 10ms/step - loss: 1.1666 - accuracy: 0.6334 - val_loss: 1.1484 - val_accuracy: 0.6420
    Epoch 11/20
    263/263 [==============================] - 3s 10ms/step - loss: 1.1465 - accuracy: 0.6399 - val_loss: 1.1256 - val_accuracy: 0.6462
    Epoch 12/20
    263/263 [==============================] - 2s 7ms/step - loss: 1.1239 - accuracy: 0.6485 - val_loss: 1.1110 - val_accuracy: 0.6533
    Epoch 13/20
    263/263 [==============================] - 2s 6ms/step - loss: 1.1017 - accuracy: 0.6589 - val_loss: 1.0951 - val_accuracy: 0.6612
    Epoch 14/20
    263/263 [==============================] - 2s 6ms/step - loss: 1.0981 - accuracy: 0.6596 - val_loss: 1.0940 - val_accuracy: 0.6620
    Epoch 15/20
    263/263 [==============================] - 2s 7ms/step - loss: 1.0854 - accuracy: 0.6636 - val_loss: 1.0948 - val_accuracy: 0.6650
    Epoch 16/20
    263/263 [==============================] - 2s 7ms/step - loss: 1.0715 - accuracy: 0.6682 - val_loss: 1.1295 - val_accuracy: 0.6475
    Epoch 17/20
    263/263 [==============================] - 2s 7ms/step - loss: 1.0626 - accuracy: 0.6713 - val_loss: 1.0618 - val_accuracy: 0.6744
    Epoch 18/20
    263/263 [==============================] - 2s 9ms/step - loss: 1.0535 - accuracy: 0.6746 - val_loss: 1.0626 - val_accuracy: 0.6702
    Epoch 19/20
    263/263 [==============================] - 2s 9ms/step - loss: 1.0486 - accuracy: 0.6745 - val_loss: 1.0610 - val_accuracy: 0.6736
    Epoch 20/20
    263/263 [==============================] - 3s 10ms/step - loss: 1.0420 - accuracy: 0.6770 - val_loss: 1.0535 - val_accuracy: 0.6745


### **Plotting the validation and training accuracies**

### **Question 3: Write your observations on the below plot. (2 Marks)**


```python
# Plotting the accuracies

dict_hist = history_model_1.history

list_ep = [i for i in range(1, 21)]

plt.figure(figsize = (8, 8))

plt.plot(list_ep, dict_hist['accuracy'], ls = '--', label = 'accuracy')

plt.plot(list_ep, dict_hist['val_accuracy'], ls = '--', label = 'val_accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.legend()

plt.show()
```


    
![png](output_32_0.png)
    


**Observations: We can see from the above plot that the accuracy only reaches up to ~68% which shows that this model is not performing that well on both the training and validation data. There is also a dip in validation accuracy at around 16 epochs. The validation accuracy is close to the training accuracy showing that the model is giving a generalized performance and is not over or underfitting.**

Let's build one more model with higher complexity and see if we can improve the performance of the model. 

First, we need to clear the previous model's history from the Keras backend. Also, let's fix the seed again after clearing the backend.


```python
# Clearing backend

from tensorflow.keras import backend

backend.clear_session()
```


```python
# Fixing the seed for random number generators

np.random.seed(42)

import random

random.seed(42)

tf.random.set_seed(42)
```

### **Second Model Architecture**
- Write a function that returns a sequential model with the following architecture:
 - First hidden layer with **256 nodes and the relu activation** and the **input shape = (1024, )**
 - Second hidden layer with **128 nodes and the relu activation**
 - Add the **Dropout layer with the rate equal to 0.2**
 - Third hidden layer with **64 nodes and the relu activation**
 - Fourth hidden layer with **64 nodes and the relu activation**
 - Fifth hidden layer with **32 nodes and the relu activation**
 - Add the **BatchNormalization layer**
 - Output layer with **activation as 'softmax' and number of nodes equal to the number of classes, i.e., 10**
 -Compile the model with the **loss equal to categorical_crossentropy, optimizer equal to Adam(learning_rate = 0.0005), and metric equal to 'accuracy'**. Do not fit the model here, just return the compiled model.
- Call the nn_model_2 function and store the model in a new variable.
- Print the summary of the model.
- Fit on the train data with a **validation split of 0.2, batch size = 128, verbose = 1, and epochs = 30**. Store the model building history to use later for visualization.

### **Question 4: Build and train the new ANN model as per the above mentioned architecture (10 Marks)**


```python
# Define the model

def nn_model_2():

    model = Sequential()  
    
    # Add layers as per the architecture mentioned above in the same sequence
    
    model.add(Dense(256, activation='relu', input_shape = (1024, )))

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(64, activation='relu'))

    model.add(Dense(64, activation='relu'))

    model.add(Dense(32, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dense(10, activation='softmax'))

    opt= Adam(learning_rate=0.0005)
    
    # Compile the model

    model.compile(loss = 'categorical_crossentropy', optimizer= opt, metrics=['accuracy'])
    
    return model
```


```python
# Build the model

model_2 = nn_model_2()
```


```python
# Print the model summary
model_2.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense (Dense)               (None, 256)               262400    
                                                                     
     dense_1 (Dense)             (None, 128)               32896     
                                                                     
     dropout (Dropout)           (None, 128)               0         
                                                                     
     dense_2 (Dense)             (None, 64)                8256      
                                                                     
     dense_3 (Dense)             (None, 64)                4160      
                                                                     
     dense_4 (Dense)             (None, 32)                2080      
                                                                     
     batch_normalization (BatchN  (None, 32)               128       
     ormalization)                                                   
                                                                     
     dense_5 (Dense)             (None, 10)                330       
                                                                     
    =================================================================
    Total params: 310,250
    Trainable params: 310,186
    Non-trainable params: 64
    _________________________________________________________________



```python
# Fit the model

history_model_2 = model_2.fit(
                  X_train,y_train,
                  validation_split=0.2,
                  batch_size=128,
                  verbose=1,
                  epochs=30
)
```

    Epoch 1/30
    263/263 [==============================] - 6s 14ms/step - loss: 2.3301 - accuracy: 0.1020 - val_loss: 2.3126 - val_accuracy: 0.0969
    Epoch 2/30
    263/263 [==============================] - 4s 14ms/step - loss: 2.3040 - accuracy: 0.1043 - val_loss: 2.3018 - val_accuracy: 0.1049
    Epoch 3/30
    263/263 [==============================] - 5s 21ms/step - loss: 2.1035 - accuracy: 0.2029 - val_loss: 1.9555 - val_accuracy: 0.2789
    Epoch 4/30
    263/263 [==============================] - 4s 15ms/step - loss: 1.6221 - accuracy: 0.4240 - val_loss: 1.5954 - val_accuracy: 0.4607
    Epoch 5/30
    263/263 [==============================] - 3s 12ms/step - loss: 1.3656 - accuracy: 0.5424 - val_loss: 1.2720 - val_accuracy: 0.5899
    Epoch 6/30
    263/263 [==============================] - 3s 13ms/step - loss: 1.1958 - accuracy: 0.6150 - val_loss: 1.1094 - val_accuracy: 0.6462
    Epoch 7/30
    263/263 [==============================] - 5s 19ms/step - loss: 1.1250 - accuracy: 0.6400 - val_loss: 1.0664 - val_accuracy: 0.6543
    Epoch 8/30
    263/263 [==============================] - 4s 17ms/step - loss: 1.0609 - accuracy: 0.6599 - val_loss: 1.0290 - val_accuracy: 0.6744
    Epoch 9/30
    263/263 [==============================] - 3s 13ms/step - loss: 1.0168 - accuracy: 0.6772 - val_loss: 0.9769 - val_accuracy: 0.6899
    Epoch 10/30
    263/263 [==============================] - 3s 13ms/step - loss: 0.9841 - accuracy: 0.6875 - val_loss: 0.9503 - val_accuracy: 0.7006
    Epoch 11/30
    263/263 [==============================] - 5s 17ms/step - loss: 0.9636 - accuracy: 0.6936 - val_loss: 0.9347 - val_accuracy: 0.7062
    Epoch 12/30
    263/263 [==============================] - 5s 18ms/step - loss: 0.9301 - accuracy: 0.7054 - val_loss: 0.9375 - val_accuracy: 0.7033
    Epoch 13/30
    263/263 [==============================] - 3s 13ms/step - loss: 0.9034 - accuracy: 0.7145 - val_loss: 0.9450 - val_accuracy: 0.6981
    Epoch 14/30
    263/263 [==============================] - 3s 12ms/step - loss: 0.8967 - accuracy: 0.7169 - val_loss: 0.8675 - val_accuracy: 0.7287
    Epoch 15/30
    263/263 [==============================] - 4s 14ms/step - loss: 0.8766 - accuracy: 0.7224 - val_loss: 0.8910 - val_accuracy: 0.7164
    Epoch 16/30
    263/263 [==============================] - 5s 21ms/step - loss: 0.8658 - accuracy: 0.7253 - val_loss: 0.8921 - val_accuracy: 0.7094
    Epoch 17/30
    263/263 [==============================] - 3s 12ms/step - loss: 0.8518 - accuracy: 0.7298 - val_loss: 0.8501 - val_accuracy: 0.7351
    Epoch 18/30
    263/263 [==============================] - 3s 12ms/step - loss: 0.8406 - accuracy: 0.7337 - val_loss: 0.8205 - val_accuracy: 0.7425
    Epoch 19/30
    263/263 [==============================] - 3s 13ms/step - loss: 0.8258 - accuracy: 0.7385 - val_loss: 0.8407 - val_accuracy: 0.7329
    Epoch 20/30
    263/263 [==============================] - 5s 20ms/step - loss: 0.8200 - accuracy: 0.7383 - val_loss: 0.8093 - val_accuracy: 0.7513
    Epoch 21/30
    263/263 [==============================] - 4s 15ms/step - loss: 0.8161 - accuracy: 0.7402 - val_loss: 0.8469 - val_accuracy: 0.7436
    Epoch 22/30
    263/263 [==============================] - 3s 12ms/step - loss: 0.8058 - accuracy: 0.7432 - val_loss: 0.8235 - val_accuracy: 0.7415
    Epoch 23/30
    263/263 [==============================] - 3s 13ms/step - loss: 0.8007 - accuracy: 0.7436 - val_loss: 0.8027 - val_accuracy: 0.7533
    Epoch 24/30
    263/263 [==============================] - 5s 18ms/step - loss: 0.7920 - accuracy: 0.7472 - val_loss: 0.7534 - val_accuracy: 0.7664
    Epoch 25/30
    263/263 [==============================] - 4s 17ms/step - loss: 0.7714 - accuracy: 0.7532 - val_loss: 0.7814 - val_accuracy: 0.7542
    Epoch 26/30
    263/263 [==============================] - 3s 13ms/step - loss: 0.7757 - accuracy: 0.7538 - val_loss: 0.7556 - val_accuracy: 0.7594
    Epoch 27/30
    263/263 [==============================] - 3s 13ms/step - loss: 0.7541 - accuracy: 0.7613 - val_loss: 0.8314 - val_accuracy: 0.7412
    Epoch 28/30
    263/263 [==============================] - 4s 15ms/step - loss: 0.7613 - accuracy: 0.7589 - val_loss: 0.7555 - val_accuracy: 0.7686
    Epoch 29/30
    263/263 [==============================] - 5s 20ms/step - loss: 0.7594 - accuracy: 0.7593 - val_loss: 0.7642 - val_accuracy: 0.7637
    Epoch 30/30
    263/263 [==============================] - 3s 13ms/step - loss: 0.7430 - accuracy: 0.7625 - val_loss: 0.7616 - val_accuracy: 0.7629


### **Plotting the validation and training accuracies**

### **Question 5: Write your observations on the below plot. (2 Marks)**


```python
# Plotting the accuracies

dict_hist = history_model_2.history

list_ep = [i for i in range(1, 31)]

plt.figure(figsize = (8, 8))

plt.plot(list_ep, dict_hist['accuracy'], ls = '--', label = 'accuracy')

plt.plot(list_ep, dict_hist['val_accuracy'], ls = '--', label = 'val_accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.legend()

plt.show()
```


    
![png](output_45_0.png)
    


**Observations: From the above plot, we can see that this model performed slightly better than Model 1. The accuracy reached around 85% for both training and validation data compared to the 68% in Model 1. As this model has more layers, it is able to identify more features from the images compared to Model 1. The validation accuracy is fluctuating more than the last model after ~11 epochs. The validation accuracy is close to the training accuracy showing that the model is giving a generalized performance and is not over or underfitting.**

## **Predictions on the test data**

- Make predictions on the test set using the second model.
- Print the obtained results using the classification report and the confusion matrix.
- Final observations on the obtained results.


```python
test_pred = model_2.predict(X_test) #predictions from the model using x_test

test_pred = np.argmax(test_pred, axis = -1) 
```

    563/563 [==============================] - 2s 3ms/step


**Note:** Earlier, we noticed that each entry of the target variable is a one-hot encoded vector but to print the classification report and confusion matrix, we must convert each entry of y_test to a single label.


```python
# Converting each entry to single label from one-hot encoded vector

y_test = np.argmax(y_test, axis = -1)
```

### **Question 6: Print the classification report and the confusion matrix for the test predictions. Write your observations on the final results. (4 Marks)**


```python
# Importing required functions

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

# Printing the classification report

print(classification_report(y_test, test_pred))

# Plotting the heatmap using confusion matrix

cm = confusion_matrix(y_test, test_pred) #The code for creating confusion matrix using actual labels (y_test) and predicted labels (test_pred)

plt.figure(figsize = (8, 5))

sns.heatmap(cm, annot = True,  fmt = '.0f')

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show()
```

                  precision    recall  f1-score   support
    
               0       0.84      0.75      0.79      1814
               1       0.74      0.79      0.77      1828
               2       0.77      0.80      0.79      1803
               3       0.62      0.77      0.69      1719
               4       0.79      0.83      0.81      1812
               5       0.76      0.68      0.72      1768
               6       0.80      0.74      0.77      1832
               7       0.83      0.77      0.80      1808
               8       0.72      0.72      0.72      1812
               9       0.75      0.74      0.75      1804
    
        accuracy                           0.76     18000
       macro avg       0.76      0.76      0.76     18000
    weighted avg       0.76      0.76      0.76     18000
    



    
![png](output_52_1.png)
    


**Final Observations: We can see from the classification report that the f1 score is lowest for the digit "3", and it also has the lowest precision out of all the digits. In the confusion matrix, we can see that digit 5 was misclassified as 3 the most frequently (the recall for digit 5 is also the lowest out of all digit recalls). We can also see that the digit 7 was misclassified as a 2 frequently. **
