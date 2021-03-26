import dataloader as d
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import AveragePooling2D, Conv2D, Dense
from tensorflow.keras.layers import Dropout, Flatten, Input
from tensorflow.keras import regularizers
from tensorflow.keras import metrics

#We expect tensorflow >2.3.2, preferably 2.4 or greater
print(tf.__version__)

#train_ds, validation_ds = d.rgb_loader()
#train_ds, val_ds = d.calib_loader()
train_x, train_y, test_x, test_y = d.get_manual_calib_data()

input_shape = (100,100,2)
inputs = Input(shape=input_shape)

#convolutional layer
conv = Conv2D(32, kernel_size=3, activation='relu')(inputs)
#pool size was not specified we can probably use 2x2 or 3x3 pooling
pooling = AveragePooling2D(pool_size=(3,3))(conv)
#first auto encoder
encoded1 = Dense(16, activation='relu', #non-linear activation
                activity_regularizer=regularizers.l1(10e-5))(pooling)#more sparse than 2nd autoencoder
decoded1 = Dense(32, activation='softmax')(encoded1)#softmax applied
dropout1 = Dropout((0.2))(decoded1)#dropout with 0.2 rate
#second auto encoder
encoded2 = Dense(8, activation='relu') (dropout1)#non-linear activation
decoded2 = Dense(16, activation='softmax')(encoded2)#softmax applied
dropout2 = Dropout((0.2))(decoded2)
#classification
flatten = Flatten()(dropout2)
#changed from 1 to 10 due to one-hot
output = Dense(10, activation='sigmoid')(flatten)
model = Model(inputs=inputs, outputs=output)

print(model.summary())
##end of summary should be (None, 10) in order to make model work,
##one-hot changed the output requirement
metrics = [metrics.Accuracy()]
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=metrics)

#one-hot encoded labels, required by categorical_crossentropy
fitting = model.fit(train_x,train_y, validation_data=(test_x,test_y)
epochs=1)

#currently output is only 1 values?
out = model.predict(test_x)
classes = out.argmax(axis=-1)
for i in range(50):
    print(out[i], classes[i], test_y[i], i)

