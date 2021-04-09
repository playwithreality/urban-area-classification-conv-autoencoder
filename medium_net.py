import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, AveragePooling2D, Conv2D, Dense
from tensorflow.keras.layers import Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import metrics
from tensorflow import keras
from visualize import confusion



def medium_net(x_train, x_test, y_train, y_test):
    print("**********")
    activation = "sigmoid"
    #check_some_data(x_train)
    ### START NETWORK ######
    cnn = Sequential()
    #Scale transformation
    cnn.add(AveragePooling2D(3, padding="same"))
    #Layer1
    cnn.add(Conv2D(filters=64, kernel_size=3, 
                    padding="same"))
    cnn.add(Activation(activation))
    #Layer2
    cnn.add(Conv2D(filters=128, kernel_size=3, padding="same"))
    cnn.add(Activation("sigmoid"))
    #Layer3
    cnn.add(Conv2D(filters=128, kernel_size=3, padding="same"))
    cnn.add(Activation("sigmoid"))
    cnn.add(Dropout(0.2))
    #Layer4
    cnn.add(Conv2D(filters=256, kernel_size=3, padding="same"))
    cnn.add(Activation("sigmoid"))
    #Layer5
    cnn.add(Conv2D(filters=256, kernel_size=3, padding="same"))
    cnn.add(Activation("sigmoid"))
    #Layer13
    cnn.add(Conv2D(filters=512, kernel_size=3, padding="same"))
    cnn.add(Activation("sigmoid"))
    #Layer14
    cnn.add(Conv2D(filters=512, kernel_size=3, padding="same"))
    cnn.add(Activation("sigmoid"))
    #Layer15
    cnn.add(Conv2D(filters=512, kernel_size=3, padding="same"))
    cnn.add(Activation("sigmoid"))
    cnn.add(Dropout(0.2))
    #Layer16
    cnn.add(Conv2D(filters=512, kernel_size=3, padding="same"))
    cnn.add(Activation("sigmoid"))


    #Dense
    cnn.add(Dense(1024, activation=activation))
    #cnn.add(Dense(2048, activation='tanh'))
    #cnn.add(Dense(4096, activation='tanh'))
    cnn.add(Flatten())
    cnn.add(Dense(10, activation="softmax"))
    #compile model, Adam or rmsprop
    cnn.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.1), loss='categorical_crossentropy')
    cnn.build((2400, 100,100,2))
    print(cnn.summary())
    #exit()
    cnn.fit(x=x_train, y=y_train, epochs=6, validation_data=(x_test, y_test))

    #Output
    out = cnn.predict(x_test)


    classes = out.argmax(axis=-1)
    correct = 0
    for i in range(len(y_test)):
        #if i < 20:
           # print(classes[i], np.where(y_test[i] == 1)[0][0])
        #print("----")
       if classes[i] == np.where(y_test[i] == 1)[0][0]:
            correct = correct + 1
    print(correct, "accuracy", correct/len(y_test))
    confusion(classes, y_test)
