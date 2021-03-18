import dataloader as d
import tensorflow as tf

#We expect tensorflow >2.3.2, preferably 2.4 or greater
print(tf.__version__)

train_ds, validation_ds = d.rgb_loader()
#train_ds, val_ds = d.calib_loader()
print("train classes", train_ds.class_names)
print("validation classes", validation_ds.class_names)