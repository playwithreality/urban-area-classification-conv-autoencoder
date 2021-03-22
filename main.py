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
#missing transformations
conv = Conv2D(filters=32, kernel_size=3, activation='relu')(inputs)
#pool size was not specified we an probably use 2x2 or 3x3 pooling
pooling = AveragePooling2D(pool_size=(3,3))(conv)

model = Model(inputs=inputs, outputs=pooling)

print(model.summary())
##end of summary should be (None, 1) in order to make model work

#model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
#model.fit(x=train_x, y=train_y, epochs=1)


#model = models.Sequential()
#model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
#model.complie()