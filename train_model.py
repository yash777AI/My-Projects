#import libraries

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook
from sklearn.utils import shuffle

#Load the data using pandas
print("[INFO] loading datasets... ")
data = pd.read_csv("./dataset/A_Z Handwritten Data.csv").astype('float32')
print(data.head())

#split the given data into data and label
X = data.drop('0',axis = 1)
y = data['0']
print(X.shape)
print(y.shape)
print(y.head())

#split the data into train data set and test data set

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.3)

#reshape data so that it can treat as an image
train_x = np.reshape(train_x.values, (train_x.shape[0], 28,28))
test_x = np.reshape(test_x.values, (test_x.shape[0], 28,28))

print("Train data shape: ", train_x.shape)
print("Test data shape: ", test_x.shape)


#Reshape data for model

train_X = train_x.reshape(train_x.shape[0],train_x.shape[1],train_x.shape[2],1)
print("New shape of train data: ", train_X.shape)

test_X = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2],1)
print("New shape of train data: ", test_X.shape)


#model CNN
print("[INFO] compiling model...")
model = Sequential()

model.add(Conv2D(32, (3, 3), padding = "same", activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(26, activation='softmax'))

model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())


# Converting the labels to categorical values...

train_y_cat = to_categorical(train_y, num_classes = 26, dtype='int')
print("New shape of train labels: ", train_y_cat.shape)

test_y_cat = to_categorical(test_y, num_classes = 26, dtype='int')
print("New shape of test labels: ", test_y_cat.shape)


print("[INFO] training Model...")
history = model.fit(train_X, train_y_cat, epochs=3,  validation_data = (test_X,test_y_cat))


print("[INFO] evaluating Model...")
print("validation accuracy:", history.history['val_accuracy'])
print("Training accuracy:", history.history['accuracy'])
print("Validation loss:", history.history['val_loss'])
print("Training loss:", history.history['loss'])

#save model
print("[INFO] save Model...")
model.save(r'./model/model_handwritting_recognition.h5')





