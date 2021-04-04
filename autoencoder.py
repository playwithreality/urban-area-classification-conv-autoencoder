
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, AveragePooling2D, Conv2D, Dense
from tensorflow.keras.layers import Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras import regularizers
from tensorflow.keras import metrics
from visualize import confusion

def autoencoder(x_train, y_train, x_test, y_test):
    ### START NETWORK ######
    inputs = Input(shape=(12860, 100,100,7))
    size = 32
    #convolutional layer
    conv = Conv2D(size, kernel_size=3, activation='relu')(inputs)
    #pool size was not specified we can probably use 2x2 or 3x3 pooling
    pooling = AveragePooling2D(pool_size=(3,3))(conv)

    #first auto encoder
    encoded1 = Dense(size * 2, activation='relu', #non-linear activation, hidden units higher than size
                    activity_regularizer=regularizers.l1(10e-5))(pooling)#more sparse than 2nd autoencoder
    decoded1 = Dense(size, activation='softmax')(encoded1)#softmax applied
    dropout1 = Dropout((0.2))(decoded1)#dropout with 0.2 rate

    #second auto encoder
    encoded2 = Dense(size / 2, activation='relu') (dropout1)#non-linear activation, hidden units lower than size
    decoded2 = Dense(size, activation='softmax')(encoded2)#softmax applied
    dropout2 = Dropout((0.2))(decoded2)
    #classification
    flatten = Flatten()(dropout2)
    #changed from 1 to 10 due to one-hot
    output = Dense(10, activation='sigmoid')(flatten)
    model = Model(inputs=inputs, outputs=output)

    print(model.summary())
    ##end of summary should be (None, 10) in order to make model work,
    ##one-hot changed the output requirement
    use_metrics = [metrics.Accuracy()]
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=use_metrics)

    #one-hot encoded labels, required by categorical_crossentropy
    model.fit(x_train,y_train, validation_data=(x_test, y_test), epochs=1)

    #currently output is only 1 values?
    out = model.predict(x_test)
    classes = out.argmax(axis=-1)
    for i in range(3):
        print("Out:", out[i], "classes", classes[i], "test", y_test[i], "idx", i)
        print("----")

    confusion(classes, y_test)