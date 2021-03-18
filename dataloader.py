import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.preprocessing as preprocessing
from os import listdir
import pathlib


def rgb_loader():
  data_dir = "./openSAR/patch_RGB"
  cal_dir = "./openSAR/patch_Calib"
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

  return train_ds, validation_ds

def visualize_data(data):
  plt.figure(figsize=(10, 10))
  for images, labels in data.take(1):
      for i in range(15):
          ax = plt.subplot(3,5, i+1)
          plt.imshow(images[i].numpy().astype("uint8"))
          plt.title(train_ds.class_names[labels[i]])
          plt.axis("off")
  plt.savefig("dataplot.png")


def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  one_hot = parts[-2] == class_names
  # Integer encode the label
  return tf.argmax(one_hot)

def get_images(path):
  filecount = len(listdir(path))
  list_ds = tf.data.Dataset.list_files(path, shuffle=False)
  list_ds = list_ds.shuffle(filecount, reshuffle_each_iteration=False)
  val_size = int(filecount * 0.2)
  train_ds = list_ds.skip(val_size)
  val_ds = list_ds.take(val_size)

  print("Sets", filecount, tf.data.experimental.cardinality(train_ds).numpy(), tf.data.experimental.cardinality(val_ds).numpy())

  return train_ds, val_ds

def calib_loader():
  path = "openSar/patch_Calib"
  train_ds, val_ds = get_images(path)
  data_dir = pathlib.Path(path)
  class_names = np.array(sorted([item.name for item in data_dir.glob('openSar/patch_Calib') if item.name != "LICENSE.txt"]))
  print("röh", class_names)
  labels = get_label(path)
  print("LABELS", labels)
  return train_ds, val_ds
