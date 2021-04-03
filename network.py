import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, AveragePooling2D, Conv2D, Dense
from tensorflow.keras.layers import Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras import regularizers
from tensorflow.keras import metrics
from visualize import confusion

def check_some_data(x_train):
    check1 = np.argwhere(np.isnan(x_train))
    print("-------------------")

def run_network(x_train, x_test, y_train, y_test):
    #check_some_data(x_train)
    ### START NETWORK ######
    cnn = Sequential()
    #Layer2
    cnn.add(Conv2D(filters=64, kernel_size=2))
    cnn.add(Activation('relu'))
    cnn.add(AveragePooling2D(2))

     #Layer3
    cnn.add(Conv2D(filters=128, kernel_size=2))
    cnn.add(Activation('relu'))
    cnn.add(AveragePooling2D(2))

    #Layer4
    cnn.add(Conv2D(filters=256, kernel_size=2))
    cnn.add(Activation('relu'))
    cnn.add(AveragePooling2D(2))
    #Layer 5
    cnn.add(Conv2D(filters=512, kernel_size=2))
    cnn.add(Activation('relu'))
    cnn.add(AveragePooling2D(2))

    #Dense
    cnn.add(Dense(1024, activation='relu'))
    cnn.add(Dense(2048, activation='relu'))
    cnn.add(Dense(4096, activation='relu'))
    cnn.add(Flatten())
    cnn.add(Dense(10, activation='sigmoid'))
    #compile model, Adam or rmsprop
    cnn.compile(optimizer="Adam", loss='categorical_crossentropy')
    cnn.build((12860, 100,100,7))
    print(cnn.summary())

    cnn.fit(x=x_train, y=y_train, epochs=10)

    #Output
    out = cnn.predict(x_test)
    classes = out.argmax(axis=-1)
    for i in range(10):
        print("Out:", out[i], "classes", classes[i], "test", y_test[i], "idx", i)
        print("----")

    confusion(classes, y_test)

