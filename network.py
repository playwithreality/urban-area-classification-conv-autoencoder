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


from tensorflow.config.experimental import list_physical_devices, set_memory_growth
physical_devices = list_physical_devices('GPU')
set_memory_growth(physical_devices[0], True)

def run_network(x_train, x_test, y_train, y_test, norm_val, activation, opti):
    print("**********")
    applypool = MaxPooling2D(2)
    #check_some_data(x_train)
    ### START NETWORK ######
    cnn = Sequential()
    #Layer2
    cnn.add(Conv2D(filters=16, kernel_size=3, kernel_constraint=max_norm(norm_val), padding="same"))
    cnn.add(Activation(activation))
    cnn.add(applypool)

     #Layer3
    cnn.add(Conv2D(filters=32, kernel_size=3, kernel_constraint=max_norm(norm_val), padding="same"))
    cnn.add(Activation(activation))
    cnn.add(applypool)

    #Layer4
    cnn.add(Conv2D(filters=64, kernel_size=3, kernel_constraint=max_norm(norm_val), padding="same"))
    cnn.add(Activation(activation))
    cnn.add(applypool)
    #Layer 5
    cnn.add(Conv2D(filters=32, kernel_size=2, kernel_constraint=max_norm(norm_val), padding="same"))
    cnn.add(Activation(activation))
    cnn.add(applypool)
    #Layer 5
    cnn.add(Conv2D(filters=32, kernel_size=2, kernel_constraint=max_norm(norm_val), padding="same"))
    cnn.add(Activation(activation))
    cnn.add(applypool)
    #Layer 5
    cnn.add(Conv2D(filters=32, kernel_size=2, kernel_constraint=max_norm(norm_val), padding="same"))
    cnn.add(Activation(activation))
    cnn.add(applypool)

    #Dense
    cnn.add(Dense(32, activation=activation))
    cnn.add(Dense(1, activation=activation))
    #cnn.add(Dense(2048, activation='tanh'))
    #cnn.add(Dense(4096, activation='tanh'))
    cnn.add(Flatten())
    cnn.add(Dense(10, activation=activation))
    #compile model, Adam or rmsprop
    cnn.compile(optimizer=opti, loss='categorical_crossentropy')
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
    print("current norm", norm_val, activation, opti)
    print("-----\n\n")
    return correct/len(y_test)
    #confusion(classes, y_test)

