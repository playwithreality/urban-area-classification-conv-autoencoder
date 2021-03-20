import dataloader as d
import tensorflow as tf

#We expect tensorflow >2.3.2, preferably 2.4 or greater
print(tf.__version__)

#train_ds, validation_ds = d.rgb_loader()
#train_ds, val_ds = d.calib_loader()
train_ds, validation_ds = d.get_manual_calib_data()
for x, y in train_ds.take(1):
    print("Image: ", x.numpy().shape)
    print("Class: ", y.numpy())