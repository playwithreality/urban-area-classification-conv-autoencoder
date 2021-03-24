import dataloader as d
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import AveragePooling2D

#We expect tensorflow >2.3.2, preferably 2.4 or greater
print(tf.__version__)

#train_ds, validation_ds = d.rgb_loader()
#train_ds, val_ds = d.calib_loader()
train_x, train_y, test_x, test_y = d.get_manual_calib_data()

input_shape = (100,100,2)
inputs = Input(shape=input_shape)

#convolutional layer
conv = Conv2D(32, kernel_size=3, activation='relu')(inputs)
#pool size was not specified we an probably use 2x2 or 3x3 pooling
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
output = Dense(1, activation='sigmoid')(flatten)
model = Model(inputs=inputs, outputs=output)

print(model.summary())
##end of summary should be (None, 1) in order to make model work

#model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
#model.fit(x=train_x, y=train_y, epochs=1)


#model = models.Sequential()
#model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
#model.complie()