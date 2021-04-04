import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.preprocessing as preprocessing
import tensorflow_io as tfio
from os import listdir
import os
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
  return np.stack(images), np.stack(image_classes)

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
  x, y = manual_calib_importer(band)
  x_train, y_train, x_test, y_test = stratified_sampling(test_percentage, x, y)
  y_train = convert_labels(y_train)
  y_test = convert_labels(y_test)
  print("Shapes: x_train: ", x_train.shape, "y_train", y_train.shape, "x_test", x_test.shape, "y_test", y_test.shape)
  #split for saving to github

  batch1 = int(12860/6)
  batch2 = int(batch1*2)
  batch3 = int(batch1*3)
  batch4 = int(batch1*4)
  batch5 = int(batch1*5)
  x_train1 = x_train[:batch1]
  x_train2 = x_train[batch1:batch2]
  x_train3 = x_train[batch2:batch3]
  x_train4 = x_train[batch3:batch4]
  x_train5 = x_train[batch4:batch5]
  x_train6 = x_train[batch5:]
  x_test1 = x_test[:int(3215/2)]
  x_test2 = x_test[int(3215/2):]
  #Laziness is a virtue so lets store train values to speed up loading :D
  #print("size", x_train.size)
  np.save("x_train1", x_train1)
  np.save("x_train2", x_train2)
  np.save("x_train3", x_train3)
  np.save("x_train4", x_train4)
  np.save("x_train5", x_train5)
  np.save("x_train6", x_train6)
  np.save("y_train", y_train)
  np.save("x_test1", x_test1)
  np.save("x_test2", x_test2)
  np.save("y_test", y_test)

  return

def get_prepared_data():
  x_train1 = np.load(open('data/x_train1.npy', 'rb'))
  x_train2 = np.load(open('data/x_train2.npy', 'rb'))
  x_train3 = np.load(open('data/x_train3.npy', 'rb'))
  x_train4 = np.load(open('data/x_train4.npy', 'rb'))
  x_train5 = np.load(open('data/x_train5.npy', 'rb'))
  x_train6 = np.load(open('data/x_train6.npy', 'rb'))
  y_train = np.load(open('data/y_train.npy', 'rb'))
  x_test1 = np.load(open('data/x_test1.npy', 'rb'))
  x_test2 = np.load(open('data/x_test2.npy', 'rb'))
  y_test = np.load(open('data/y_test.npy', 'rb'))

  x_train = np.concatenate((x_train1, x_train2, x_train3, x_train4, x_train5, x_train6))
  x_test = np.concatenate((x_test1, x_test2))
  return x_train, y_train, x_test, y_test

def get_prepared_means():
  mean_train1 = np.load(open('glcm/mean_train1.npy', 'rb'))
  mean_train2 = np.load(open('glcm/mean_train2.npy', 'rb'))
  mean_train3 = np.load(open('glcm/mean_train3.npy', 'rb'))
  mean_train4 = np.load(open('glcm/mean_train4.npy', 'rb'))
  mean_train5 = np.load(open('glcm/mean_train5.npy', 'rb'))
  mean_train6 = np.load(open('glcm/mean_train6.npy', 'rb'))
  mean_test1 = np.load(open('glcm/mean_test1.npy', 'rb'))
  mean_test2 = np.load(open('glcm/mean_test2.npy', 'rb'))
  mean_train = np.concatenate((mean_train1, mean_train2, mean_train3, mean_train4, mean_train5, mean_train6))
  mean_test = np.concatenate((mean_test1, mean_test2))
  return mean_train, mean_test

def get_prepared_glcm():
  mean_train1 = np.stack(np.load(open('glcm/mean_train1.npy', 'rb')))
  mean_train2 = np.stack(np.load(open('glcm/mean_train2.npy', 'rb')))
  mean_train3 = np.stack(np.load(open('glcm/mean_train3.npy', 'rb')))
  mean_train4 = np.stack(np.load(open('glcm/mean_train4.npy', 'rb')))
  mean_train5 = np.stack(np.load(open('glcm/mean_train5.npy', 'rb')))
  mean_train6 = np.stack(np.load(open('glcm/mean_train6.npy', 'rb')))
  mean_train7 = np.stack(np.load(open('glcm/mean_train7.npy', 'rb')))
  mean_train8 = np.stack(np.load(open('glcm/mean_train8.npy', 'rb')))
  mean_train9 = np.stack(np.load(open('glcm/mean_train9.npy', 'rb')))
  mean_train10 = np.stack(np.load(open('glcm/mean_train10.npy', 'rb')))
  mean_test1 = np.stack(np.load(open('glcm/mean_test1.npy', 'rb')))
  mean_test2 = np.stack(np.load(open('glcm/mean_test2.npy', 'rb')))
  mean_test3 = np.stack(np.load(open('glcm/mean_test3.npy', 'rb')))

  var_train1 = np.stack(np.load(open('glcm/var_train1.npy', 'rb')))
  var_train2 = np.stack(np.load(open('glcm/var_train2.npy', 'rb')))
  var_train3 = np.stack(np.load(open('glcm/var_train3.npy', 'rb')))
  var_train4 = np.stack(np.load(open('glcm/var_train4.npy', 'rb')))
  var_train5 = np.stack(np.load(open('glcm/var_train5.npy', 'rb')))
  var_train6 = np.stack(np.load(open('glcm/var_train6.npy', 'rb')))
  var_train7 = np.stack(np.load(open('glcm/var_train7.npy', 'rb')))
  var_train8 = np.stack(np.load(open('glcm/var_train8.npy', 'rb')))
  var_train9 = np.stack(np.load(open('glcm/var_train9.npy', 'rb')))
  var_train10 = np.stack(np.load(open('glcm/var_train10.npy', 'rb')))
  var_test1 = np.stack(np.load(open('glcm/var_test1.npy', 'rb')))
  var_test2 = np.stack(np.load(open('glcm/var_test2.npy', 'rb')))
  var_test3 = np.stack(np.load(open('glcm/var_test3.npy', 'rb')))

  mean_train = np.concatenate((mean_train1, mean_train2, mean_train3, mean_train4, mean_train5, 
                              mean_train6, mean_train7, mean_train8, mean_train9, mean_train10))
  mean_test = np.concatenate((mean_test1, mean_test2, mean_test3))
  var_train = np.concatenate((var_train1, var_train2, var_train3, var_train4, var_train5, 
                            var_train6, var_train7, var_train8, var_train9, var_train10))
  var_test = np.concatenate((var_test1, var_test2, var_test3))

  return mean_train[..., np.newaxis], var_train[..., np.newaxis], mean_test[..., np.newaxis], var_test[..., np.newaxis]