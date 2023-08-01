# **Convolutional Neural Networks Project: Street View Housing Number Digit Recognition**

# **Marks: 30**

Welcome to the project on classification using Convolutional Neural Networks. We will continue to work with the Street View Housing Numbers (SVHN) image dataset for this project.

Note: This project was used to compare the performance of a CNN vs. ANN on image classification problems. The ANN Method is uploaded as part of the repo.

--------------
## **Context** 
--------------

One of the most interesting tasks in deep learning is to recognize objects in natural scenes. The ability to process visual information using machine learning algorithms can be very useful as demonstrated in various applications.

The SVHN dataset contains over 600,000 labeled digits cropped from street-level photos. It is one of the most popular image recognition datasets. It has been used in neural networks created by Google to improve the map quality by automatically transcribing the address numbers from a patch of pixels. The transcribed number with a known street address helps pinpoint the location of the building it represents. 

----------------
## **Objective**
----------------

To build a CNN model that can recognize the digits in the images.

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

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, LeakyReLU

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

# Load the the train and the test dataset

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

### **Question 1: Complete the below code to visualize the first 10 images in the dataset. (1 Mark)**


```python
# Visualizing the first 10 images in the dataset and printing their labels

%matplotlib inline

import matplotlib.pyplot as plt

plt.figure(figsize = (10, 1))

for i in range(10):

    plt.subplot(1, 10, i+1)
    
    plt.imshow(X_train[i], cmap = "gray")  # Write the function to visualize images

    plt.axis('off')

plt.show()

print('label for each of the above image: %s' % (y_train[0:10]))
```


    
![png](output_14_0.png)
    


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
# Reshaping the dataset to be able to pass them to CNNs. Remember that we always have to give a 4D array as input to CNNs

X_train = X_train.reshape(X_train.shape[0], 32, 32, 1)

X_test = X_test.reshape(X_test.shape[0], 32, 32, 1)
```


```python
# Normalize inputs from 0-255 to 0-1

X_train = X_train / 255.0

X_test = X_test / 255.0
```


```python
# New shape 

print('Training set:', X_train.shape, y_train.shape)

print('Test set:', X_test.shape, y_test.shape)
```

    Training set: (42000, 32, 32, 1) (42000,)
    Test set: (18000, 32, 32, 1) (18000,)


### **Question 2: One-hot encode the labels in the target variable y_train and y_test. (2 Marks)**


```python
# Write the function and appropriate variable name to one-hot encode the output

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)

# test labels

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

Now that we have done data preprocessing, let's build a CNN model.


```python
# Fixing the seed for random number generators

np.random.seed(42)

import random

random.seed(42)

tf.random.set_seed(42)
```

### **Model Architecture**
- **Write a function** that returns a sequential model with the following architecture:
 - First Convolutional layer with **16 filters and the kernel size of 3x3**. Use the **'same' padding** and provide the **input shape = (32, 32, 1)**
 - Add a **LeakyRelu layer** with the **slope equal to 0.1**
 - Second Convolutional layer with **32 filters and the kernel size of 3x3 with 'same' padding**
 - Another **LeakyRelu** with the **slope equal to 0.1**
 - A **max-pooling layer** with a **pool size of 2x2**
 - **Flatten** the output from the previous layer
 - Add a **dense layer with 32 nodes**
 - Add a **LeakyRelu layer with the slope equal to 0.1**
 - Add the final **output layer with nodes equal to the number of classes, i.e., 10** and **'softmax' as the activation function**
 - Compile the model with the **loss equal to categorical_crossentropy, optimizer equal to Adam(learning_rate = 0.001), and metric equal to 'accuracy'**. Do not fit the model here, just return the compiled model.
- Call the function cnn_model_1 and store the output in a new variable.
- Print the summary of the model.
- Fit the model on the training data with a **validation split of 0.2, batch size = 32, verbose = 1, and epochs = 20**. Store the model building history to use later for visualization.

### **Question 3: Build and train a CNN model as per the above mentioned architecture. (10 Marks)**


```python
# Define the model

def cnn_model_1():

    model = Sequential() 
    
    # Add layers as per the architecture mentioned above in the same sequence

    model.add(Conv2D(16, (3,3), padding='same',input_shape=(32,32,1)))

    model.add(LeakyReLU(0.1))

    model.add(Conv2D(32, (3,3), padding='same'))

    model.add(LeakyReLU(0.1))

    model.add(MaxPooling2D(2,2))

    model.add(Flatten())

    model.add(Dense(32))

    model.add(LeakyReLU(0.1))

    model.add(Dense(10, activation='softmax'))

    opt = Adam(learning_rate=0.001)
    
    # Compile the model

    model.compile(loss='categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
    
    return model
```


```python
# Build the model

model_1 = cnn_model_1()
```


```python
# Print the model summary

model_1.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d_4 (Conv2D)           (None, 32, 32, 16)        160       
                                                                     
     leaky_re_lu_5 (LeakyReLU)   (None, 32, 32, 16)        0         
                                                                     
     conv2d_5 (Conv2D)           (None, 32, 32, 32)        4640      
                                                                     
     leaky_re_lu_6 (LeakyReLU)   (None, 32, 32, 32)        0         
                                                                     
     max_pooling2d_2 (MaxPooling  (None, 16, 16, 32)       0         
     2D)                                                             
                                                                     
     flatten_1 (Flatten)         (None, 8192)              0         
                                                                     
     dense_2 (Dense)             (None, 32)                262176    
                                                                     
     leaky_re_lu_7 (LeakyReLU)   (None, 32)                0         
                                                                     
     dense_3 (Dense)             (None, 10)                330       
                                                                     
    =================================================================
    Total params: 267,306
    Trainable params: 267,306
    Non-trainable params: 0
    _________________________________________________________________



```python
# Fit the model

history_model_1 = model_1.fit(X_train, y_train,
                              validation_split=0.2,
                              batch_size=32,
                              verbose=1,
                              epochs=20)
```

    Epoch 1/20
    1050/1050 [==============================] - 8s 6ms/step - loss: 1.1113 - accuracy: 0.6379 - val_loss: 0.6518 - val_accuracy: 0.8110
    Epoch 2/20
    1050/1050 [==============================] - 6s 5ms/step - loss: 0.5365 - accuracy: 0.8476 - val_loss: 0.5270 - val_accuracy: 0.8481
    Epoch 3/20
    1050/1050 [==============================] - 5s 4ms/step - loss: 0.4424 - accuracy: 0.8686 - val_loss: 0.5002 - val_accuracy: 0.8551
    Epoch 4/20
    1050/1050 [==============================] - 5s 4ms/step - loss: 0.3836 - accuracy: 0.8865 - val_loss: 0.4426 - val_accuracy: 0.8761
    Epoch 5/20
    1050/1050 [==============================] - 6s 5ms/step - loss: 0.3392 - accuracy: 0.8981 - val_loss: 0.4404 - val_accuracy: 0.8770
    Epoch 6/20
    1050/1050 [==============================] - 5s 4ms/step - loss: 0.3000 - accuracy: 0.9096 - val_loss: 0.4628 - val_accuracy: 0.8727
    Epoch 7/20
    1050/1050 [==============================] - 6s 5ms/step - loss: 0.2688 - accuracy: 0.9197 - val_loss: 0.4558 - val_accuracy: 0.8669
    Epoch 8/20
    1050/1050 [==============================] - 5s 5ms/step - loss: 0.2389 - accuracy: 0.9270 - val_loss: 0.4706 - val_accuracy: 0.8719
    Epoch 9/20
    1050/1050 [==============================] - 5s 4ms/step - loss: 0.2152 - accuracy: 0.9340 - val_loss: 0.4835 - val_accuracy: 0.8723
    Epoch 10/20
    1050/1050 [==============================] - 6s 6ms/step - loss: 0.1955 - accuracy: 0.9391 - val_loss: 0.4965 - val_accuracy: 0.8739
    Epoch 11/20
    1050/1050 [==============================] - 5s 4ms/step - loss: 0.1732 - accuracy: 0.9464 - val_loss: 0.5213 - val_accuracy: 0.8730
    Epoch 12/20
    1050/1050 [==============================] - 5s 5ms/step - loss: 0.1570 - accuracy: 0.9511 - val_loss: 0.5726 - val_accuracy: 0.8695
    Epoch 13/20
    1050/1050 [==============================] - 5s 5ms/step - loss: 0.1409 - accuracy: 0.9554 - val_loss: 0.6092 - val_accuracy: 0.8644
    Epoch 14/20
    1050/1050 [==============================] - 5s 5ms/step - loss: 0.1282 - accuracy: 0.9601 - val_loss: 0.6057 - val_accuracy: 0.8652
    Epoch 15/20
    1050/1050 [==============================] - 6s 6ms/step - loss: 0.1153 - accuracy: 0.9628 - val_loss: 0.6340 - val_accuracy: 0.8724
    Epoch 16/20
    1050/1050 [==============================] - 5s 4ms/step - loss: 0.1007 - accuracy: 0.9680 - val_loss: 0.6675 - val_accuracy: 0.8682
    Epoch 17/20
    1050/1050 [==============================] - 5s 4ms/step - loss: 0.0915 - accuracy: 0.9710 - val_loss: 0.7182 - val_accuracy: 0.8695
    Epoch 18/20
    1050/1050 [==============================] - 6s 6ms/step - loss: 0.0885 - accuracy: 0.9710 - val_loss: 0.7447 - val_accuracy: 0.8656
    Epoch 19/20
    1050/1050 [==============================] - 5s 4ms/step - loss: 0.0817 - accuracy: 0.9736 - val_loss: 0.7968 - val_accuracy: 0.8607
    Epoch 20/20
    1050/1050 [==============================] - 5s 5ms/step - loss: 0.0708 - accuracy: 0.9764 - val_loss: 0.8076 - val_accuracy: 0.8654


### **Plotting the validation and training accuracies**

### **Question 4: Write your observations on the below plot. (2 Marks)**


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


    
![png](output_33_0.png)
    


**Observations: The above plot shows that the training accuracy is lot higher at almost 98% compared to the validation accuracy which was around 87%. This shows that the model is overfitting the training data. The model needs to be more generalized. **

Let's build another model and see if we can get a better model with generalized performance.

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
 - First Convolutional layer with **16 filters and the kernel size of 3x3**. Use the **'same' padding** and provide the **input shape = (32, 32, 1)**
 - Add a **LeakyRelu layer** with the **slope equal to 0.1**
 - Second Convolutional layer with **32 filters and the kernel size of 3x3 with 'same' padding**
 - Add **LeakyRelu** with the **slope equal to 0.1**
 - Add a **max-pooling layer** with a **pool size of 2x2**
 - Add a **BatchNormalization layer**
 - Third Convolutional layer with **32 filters and the kernel size of 3x3 with 'same' padding**
 - Add a **LeakyRelu layer with the slope equal to 0.1**
 - Fourth Convolutional layer **64 filters and the kernel size of 3x3 with 'same' padding** 
 - Add a **LeakyRelu layer with the slope equal to 0.1**
 - Add a **max-pooling layer** with a **pool size of 2x2**
 - Add a **BatchNormalization layer**
 - **Flatten** the output from the previous layer
 - Add a **dense layer with 32 nodes**
 - Add a **LeakyRelu layer with the slope equal to 0.1**
 - Add a **dropout layer with the rate equal to 0.5**
 - Add the final **output layer with nodes equal to the number of classes, i.e., 10** and **'softmax' as the activation function**
 - Compile the model with the **categorical_crossentropy loss, adam optimizers (learning_rate = 0.001), and metric equal to 'accuracy'**. Do not fit the model here, just return the compiled model.
- Call the function cnn_model_2 and store the model in a new variable.
- Print the summary of the model.
- Fit the model on the train data with a **validation split of 0.2, batch size = 128, verbose = 1, and epochs = 30**. Store the model building history to use later for visualization.

### **Question 5: Build and train the second CNN model as per the above mentioned architecture. (10 Marks)**


```python
# Define the model

def cnn_model_2():
    
    model = Sequential()
    
    # Add layers as per the architecture mentioned above in the same sequence

    model.add(Conv2D(16, (3,3), padding='same', input_shape=(32,32,1)))

    model.add(LeakyReLU(0.1))

    model.add(Conv2D(32, (3,3), padding='same'))

    model.add(LeakyReLU(0.1))

    model.add(MaxPooling2D(2,2))

    model.add(BatchNormalization())

    model.add(Conv2D(32, (3,3), padding='same'))

    model.add(LeakyReLU(0.1))

    model.add(Conv2D(64, (3,3), padding='same'))

    model.add(LeakyReLU(0.1))

    model.add(MaxPooling2D(2,2))

    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(32))

    model.add(LeakyReLU(0.1))

    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    opt = Adam(learning_rate=0.001)

    # Compile the model

    model.compile(loss='categorical_crossentropy', optimizer = opt, metrics =['accuracy'])
    
    return model
```


```python
# Build the model

model_2 = cnn_model_2()
```


```python
# Print the summary

model_2.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d (Conv2D)             (None, 32, 32, 16)        160       
                                                                     
     leaky_re_lu (LeakyReLU)     (None, 32, 32, 16)        0         
                                                                     
     conv2d_1 (Conv2D)           (None, 32, 32, 32)        4640      
                                                                     
     leaky_re_lu_1 (LeakyReLU)   (None, 32, 32, 32)        0         
                                                                     
     max_pooling2d (MaxPooling2D  (None, 16, 16, 32)       0         
     )                                                               
                                                                     
     batch_normalization (BatchN  (None, 16, 16, 32)       128       
     ormalization)                                                   
                                                                     
     conv2d_2 (Conv2D)           (None, 16, 16, 32)        9248      
                                                                     
     leaky_re_lu_2 (LeakyReLU)   (None, 16, 16, 32)        0         
                                                                     
     conv2d_3 (Conv2D)           (None, 16, 16, 64)        18496     
                                                                     
     leaky_re_lu_3 (LeakyReLU)   (None, 16, 16, 64)        0         
                                                                     
     max_pooling2d_1 (MaxPooling  (None, 8, 8, 64)         0         
     2D)                                                             
                                                                     
     batch_normalization_1 (Batc  (None, 8, 8, 64)         256       
     hNormalization)                                                 
                                                                     
     flatten (Flatten)           (None, 4096)              0         
                                                                     
     dense (Dense)               (None, 32)                131104    
                                                                     
     leaky_re_lu_4 (LeakyReLU)   (None, 32)                0         
                                                                     
     dropout (Dropout)           (None, 32)                0         
                                                                     
     dense_1 (Dense)             (None, 10)                330       
                                                                     
    =================================================================
    Total params: 164,362
    Trainable params: 164,170
    Non-trainable params: 192
    _________________________________________________________________



```python
# Fit the model

history_model_2 = model_2.fit(X_train, y_train,
                              validation_split=0.2,
                              batch_size=128,
                              verbose=1,
                              epochs=30)
```

    Epoch 1/30
    263/263 [==============================] - 6s 11ms/step - loss: 1.3153 - accuracy: 0.5535 - val_loss: 1.8483 - val_accuracy: 0.3005
    Epoch 2/30
    263/263 [==============================] - 2s 9ms/step - loss: 0.6658 - accuracy: 0.7952 - val_loss: 0.5816 - val_accuracy: 0.8351
    Epoch 3/30
    263/263 [==============================] - 3s 10ms/step - loss: 0.5475 - accuracy: 0.8314 - val_loss: 0.4538 - val_accuracy: 0.8661
    Epoch 4/30
    263/263 [==============================] - 3s 11ms/step - loss: 0.4826 - accuracy: 0.8528 - val_loss: 0.4384 - val_accuracy: 0.8692
    Epoch 5/30
    263/263 [==============================] - 3s 11ms/step - loss: 0.4383 - accuracy: 0.8649 - val_loss: 0.3773 - val_accuracy: 0.8985
    Epoch 6/30
    263/263 [==============================] - 3s 10ms/step - loss: 0.3942 - accuracy: 0.8783 - val_loss: 0.3842 - val_accuracy: 0.8888
    Epoch 7/30
    263/263 [==============================] - 2s 9ms/step - loss: 0.3757 - accuracy: 0.8848 - val_loss: 0.3485 - val_accuracy: 0.9058
    Epoch 8/30
    263/263 [==============================] - 3s 10ms/step - loss: 0.3451 - accuracy: 0.8918 - val_loss: 0.3998 - val_accuracy: 0.8911
    Epoch 9/30
    263/263 [==============================] - 3s 10ms/step - loss: 0.3217 - accuracy: 0.9001 - val_loss: 0.3535 - val_accuracy: 0.9108
    Epoch 10/30
    263/263 [==============================] - 3s 11ms/step - loss: 0.3013 - accuracy: 0.9071 - val_loss: 0.3491 - val_accuracy: 0.9085
    Epoch 11/30
    263/263 [==============================] - 3s 10ms/step - loss: 0.2837 - accuracy: 0.9108 - val_loss: 0.3424 - val_accuracy: 0.9138
    Epoch 12/30
    263/263 [==============================] - 3s 10ms/step - loss: 0.2731 - accuracy: 0.9139 - val_loss: 0.3453 - val_accuracy: 0.9101
    Epoch 13/30
    263/263 [==============================] - 3s 10ms/step - loss: 0.2619 - accuracy: 0.9176 - val_loss: 0.3361 - val_accuracy: 0.9073
    Epoch 14/30
    263/263 [==============================] - 3s 12ms/step - loss: 0.2539 - accuracy: 0.9196 - val_loss: 0.3746 - val_accuracy: 0.9077
    Epoch 15/30
    263/263 [==============================] - 3s 11ms/step - loss: 0.2447 - accuracy: 0.9237 - val_loss: 0.4583 - val_accuracy: 0.8918
    Epoch 16/30
    263/263 [==============================] - 3s 10ms/step - loss: 0.2349 - accuracy: 0.9240 - val_loss: 0.4153 - val_accuracy: 0.8986
    Epoch 17/30
    263/263 [==============================] - 3s 10ms/step - loss: 0.2263 - accuracy: 0.9268 - val_loss: 0.4079 - val_accuracy: 0.9057
    Epoch 18/30
    263/263 [==============================] - 3s 10ms/step - loss: 0.2145 - accuracy: 0.9295 - val_loss: 0.3353 - val_accuracy: 0.9188
    Epoch 19/30
    263/263 [==============================] - 3s 12ms/step - loss: 0.2041 - accuracy: 0.9341 - val_loss: 0.3856 - val_accuracy: 0.9132
    Epoch 20/30
    263/263 [==============================] - 3s 11ms/step - loss: 0.1993 - accuracy: 0.9357 - val_loss: 0.4142 - val_accuracy: 0.9120
    Epoch 21/30
    263/263 [==============================] - 3s 10ms/step - loss: 0.1905 - accuracy: 0.9365 - val_loss: 0.4418 - val_accuracy: 0.9038
    Epoch 22/30
    263/263 [==============================] - 3s 10ms/step - loss: 0.1837 - accuracy: 0.9388 - val_loss: 0.4061 - val_accuracy: 0.9054
    Epoch 23/30
    263/263 [==============================] - 3s 10ms/step - loss: 0.1806 - accuracy: 0.9426 - val_loss: 0.4207 - val_accuracy: 0.9133
    Epoch 24/30
    263/263 [==============================] - 3s 12ms/step - loss: 0.1746 - accuracy: 0.9427 - val_loss: 0.4414 - val_accuracy: 0.9026
    Epoch 25/30
    263/263 [==============================] - 3s 11ms/step - loss: 0.1686 - accuracy: 0.9444 - val_loss: 0.4233 - val_accuracy: 0.9154
    Epoch 26/30
    263/263 [==============================] - 3s 10ms/step - loss: 0.1634 - accuracy: 0.9465 - val_loss: 0.4282 - val_accuracy: 0.9135
    Epoch 27/30
    263/263 [==============================] - 3s 10ms/step - loss: 0.1645 - accuracy: 0.9452 - val_loss: 0.4571 - val_accuracy: 0.9121
    Epoch 28/30
    263/263 [==============================] - 3s 10ms/step - loss: 0.1602 - accuracy: 0.9481 - val_loss: 0.4222 - val_accuracy: 0.9127
    Epoch 29/30
    263/263 [==============================] - 3s 11ms/step - loss: 0.1561 - accuracy: 0.9463 - val_loss: 0.4281 - val_accuracy: 0.9119
    Epoch 30/30
    263/263 [==============================] - 3s 11ms/step - loss: 0.1506 - accuracy: 0.9493 - val_loss: 0.4203 - val_accuracy: 0.9173


### **Plotting the validation and training accuracies**

### **Question 6: Write your observations on the below plot. (2 Marks)**


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


    
![png](output_46_0.png)
    


**Observations: This model is much more generalized model than Model 1. The training and validation accuracies are close to each other. The training accuracy is around 95% while the validation accuracy is around 91%. This shows that the model is peforming well.**

## **Predictions on the test data**

- Make predictions on the test set using the second model.
- Print the obtained results using the classification report and the confusion matrix.
- Final observations on the obtained results.

### **Question 7: Make predictions on the test data using the second model. (1 Mark)** 


```python
# Make prediction on the test data using model_2 

test_pred = model_2.predict(X_test)

test_pred = np.argmax(test_pred, axis = -1)
```

    563/563 [==============================] - 2s 3ms/step


**Note:** Earlier, we noticed that each entry of the target variable is a one-hot encoded vector, but to print the classification report and confusion matrix, we must convert each entry of y_test to a single label.


```python
# Converting each entry to single label from one-hot encoded vector

y_test = np.argmax(y_test, axis = -1)
```

### **Question 8: Write your final observations on the performance of the model on the test data. (2 Marks)**


```python
# Importing required functions

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

# Printing the classification report

print(classification_report(y_test, test_pred))

# Plotting the heatmap using confusion matrix

cm = confusion_matrix(y_test, test_pred)

plt.figure(figsize = (8, 5))

sns.heatmap(cm, annot = True,  fmt = '.0f')

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show()
```

                  precision    recall  f1-score   support
    
               0       0.93      0.94      0.94      1814
               1       0.91      0.91      0.91      1828
               2       0.94      0.92      0.93      1803
               3       0.90      0.89      0.89      1719
               4       0.92      0.93      0.92      1812
               5       0.89      0.93      0.91      1768
               6       0.90      0.90      0.90      1832
               7       0.95      0.92      0.93      1808
               8       0.92      0.90      0.91      1812
               9       0.90      0.91      0.90      1804
    
        accuracy                           0.91     18000
       macro avg       0.91      0.91      0.91     18000
    weighted avg       0.91      0.91      0.91     18000
    



    
![png](output_54_1.png)
    


**Final Observations: Overall, the model is performing well, with each digit having relatively high f1 scores. Digit 3 was  misclassified as a 5 the most frequently (Digit 3 also has the lowest f1 score out of all digits, while Digits 6 and 9 have the second lowest f1 score). Digit 8 was also misclassified as a 6 frequently. Overall, the model was able to accurately classify the digits as seen by the diagonal of the confusion matrix. **
