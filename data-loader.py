import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.preprocessing as preprocessing

#We expect tensorflow >2.3.2, preferably 2.4 or greater
print(tf.__version__)

data_dir = "./openSAR/patch_RGB"
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(100, 100),
  batch_size=100
)
validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(100, 100),
  batch_size=100
)

print("train classes", train_ds.class_names)
print("validation classes", validation_ds.class_names)

#Should probably use patch_calib instead?---
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(15):
        ax = plt.subplot(3,5, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(train_ds.class_names[labels[i]])
        plt.axis("off")
plt.savefig("dataviz.png")

#

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense

inputs = Input(shape=(100,100,3))
#missing transformations
conv = Conv2D(32, kernel_size=3, activation='relu')(inputs)#convolution
model = Model(inputs=inputs, outputs=conv)

print(model.summary())
##end of summary should be (None, 1) in order to make model work

#model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
#model.fit(train_ds, epochs=1, validation_data=validation_ds)
