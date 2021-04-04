import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, AveragePooling2D, Conv2D, Dense
from tensorflow.keras.layers import Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import metrics
from visualize import confusion



def large_net(x_train, x_test, y_train, y_test):
    print("**********")
    activation = "sigmoid"
    norm_val = 1.0
    #check_some_data(x_train)
    ### START NETWORK ######
    cnn = Sequential()
    #Layer1
    cnn.add(Conv2D(filters=64, kernel_size=3, kernel_constraint=max_norm(norm_val), 
                    padding="same", activity_regularizer=regularizers.l1(10e-5)))
    cnn.add(Activation(activation))

    #Layer2
    cnn.add(Conv2D(filters=128, kernel_size=3, kernel_constraint=max_norm(norm_val), padding="same"))
    cnn.add(Activation("relu"))

    #Layer3
    cnn.add(Conv2D(filters=258, kernel_size=3, kernel_constraint=max_norm(norm_val), padding="same"))
    cnn.add(Activation("relu"))
    #Layer4
    cnn.add(Conv2D(filters=512, kernel_size=3, kernel_constraint=max_norm(norm_val), padding="same"))
    cnn.add(Activation("relu"))
    #Layer5
    cnn.add(Conv2D(filters=512, kernel_size=3, kernel_constraint=max_norm(norm_val), padding="same"))
    cnn.add(Activation("relu"))
    #Layer6
    cnn.add(Conv2D(filters=512, kernel_size=3, kernel_constraint=max_norm(norm_val), padding="same"))
    cnn.add(Activation("relu"))
    #Layer7
    cnn.add(Conv2D(filters=512, kernel_size=3, kernel_constraint=max_norm(norm_val), padding="same"))
    cnn.add(Activation("relu"))
    #Layer8
    cnn.add(Conv2D(filters=512, kernel_size=3, kernel_constraint=max_norm(norm_val), padding="same"))
    cnn.add(Activation("relu"))
    #Layer9
    cnn.add(Conv2D(filters=512, kernel_size=3, kernel_constraint=max_norm(norm_val), padding="same"))
    cnn.add(Activation("relu"))


    #Dense
    cnn.add(Dense(1024, activation=activation, activity_regularizer=regularizers.l1(10e-5)))
    #cnn.add(Dense(2048, activation='tanh'))
    #cnn.add(Dense(4096, activation='tanh'))
    cnn.add(Flatten())
    cnn.add(Dense(10, activation=activation))
    #compile model, Adam or rmsprop
    cnn.compile(optimizer="rmsprop", loss='categorical_crossentropy')
    cnn.build((1310, 100,100,7))
    #print(cnn.summary())
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
