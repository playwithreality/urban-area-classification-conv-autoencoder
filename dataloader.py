import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.preprocessing as preprocessing
import tensorflow_io as tfio
from os import listdir
import pathlib
import tifffile as tiffer
from sklearn.model_selection import StratifiedShuffleSplit

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
  labels = get_label(path)
  print("LABELS", labels)
  return train_ds, val_ds


def manual_calib_importer(band):
  path = "openSar/patch_Calib"
  classes = listdir(path)
  images = []
  image_classes = []
  for c in classes:
    files = listdir(path+"/"+c)
    for file in files:
      img = tiffer.imread(path+"/"+c+"/"+file, key=0)
      images.append(img[:,:,band])
      image_classes.append(c)
  return np.stack(images)/np.max(images), np.stack(image_classes), len(image_classes)

class_coversions = {
  "airport": 0,
  "denselow": 1,
  "GeneralResidential": 2,
  "highbuildings": 3,
  "highway": 4,
  "railway": 5,
  "SingleBuilding": 6,
  "Skyscraper": 7,
  "StorageArea": 8,
  "vegetation": 9
}

#encodes in one-hot format
def convert_labels(y):
  new_y = []
  for i in y:
    array = [0,0,0,0,0,0,0,0,0,0]
    array[class_coversions.get(i)] = 1
    new_y.append(array)
  return np.stack(new_y)

def stratified_sampling(test_percentage, x, y):
    splits = StratifiedShuffleSplit(test_size=test_percentage, random_state=2)
    for train_index, test_index in splits.split(x,y):
      x_train, x_test = x[train_index], x[test_index]
      y_train, y_test = y[train_index], y[test_index]
      return x_train, y_train, x_test, y_test


def get_manual_calib_data(test_percentage, band):
  x, y, length = manual_calib_importer(band)
  x_train, y_train, x_test, y_test = stratified_sampling(test_percentage, x, y)
  y_train = convert_labels(y_train)
  y_test = convert_labels(y_test)
  print("Shapes: x_train: ", x_train.shape, "y_train", y_train.shape, "x_test", x_test.shape, "y_test", y_test.shape)
  #Laziness is a virtue so lets store train values to speed up loading :D
  np.save("x_train", x_train)
  np.save("y_train", y_train)
  np.save("x_test", x_test)
  np.save("y_test", y_test)

  return

