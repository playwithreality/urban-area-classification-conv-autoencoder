import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import  AveragePooling2D, MaxPooling2D

import dataloader as d
from visualize import confusion, plot_images
from filters import compute_glcm_results, gabor, glcm_no_save
from network import run_network
from autoencoder import autoencoder
from large_net import large_net
from medium_net import medium_net
from scaling import scaler
#from resampler import resampler
#We expect tensorflow >2.3.2, preferably 2.4 or greater
print(tf.__version__)

##any config values
test_percentage = 0.2
#input_shape = (100,100,7)
band = 1

x_train, y_train, x_test, y_test = d.get_manual_calib_data(test_percentage, band)
x_train2, y_train2, x_test2, y_test2 = d.get_manual_calib_data(test_percentage, 0)
#I heard that laziness is a virtue and programmers are lazy people
#x_train, y_train, x_test, y_test = d.get_prepared_data()
#x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
#x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))
print("min-max", np.min(x_train), np.max(x_train), np.min(x_test), np.max(x_test))
#x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
#x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))
x_train = scaler(x_train)
x_test = scaler(x_test)
x_train2 = scaler(x_train2)
x_test2 = scaler(x_test2)
print("min-max", np.min(x_train), np.max(x_train), np.min(x_test), np.max(x_test))
print("SAMPLS", x_train.shape)
print("SAMPLS", x_test.shape)
#np.savetxt("train_onehot.csv", y_train, delimiter=",", fmt = '% s')
#np.savetxt("test_onehot.csv", y_train, delimiter=",", fmt = '% s')


#scaler1 = preprocessing.StandardScaler().fit(x_train)
#scaler2 = preprocessing.StandardScaler().fit(x_test)
#x_train = scaler1.transform(x_train)
#x_test = scaler2.transform(x_test)

##this section will include glcm+gabor filter computation / loading ##
#compute glcm mea+varn and store in file for later use, laziness level 2.0
#mean_train, var_train, mean_test, var_test = compute_glcm_results(x_train, x_test)

#mean_train, var_train, mean_test, var_test = d.get_prepared_glcm()
#mean_train, var_train, mean_test, var_test = glcm_no_save(x_train, x_test)

#gabor_train = np.stack(gabor(x_train, "x_train"))
#gabor_test = np.stack(gabor(x_test, "x_test"))
#x_train = x_train[..., np.newaxis]
#x_train2 = x_train2[..., np.newaxis]
#x_test = x_test[..., np.newaxis]
#x_test2 = x_test2[..., np.newaxis]
#
print("shapes", x_train.shape, x_train2.shape, x_test.shape, x_test2.shape)
train = np.stack((x_train, x_train2), axis=3)
test = np.stack((x_test, x_test2), axis=3)
print("shapes", train.shape, test.shape)

#print("CHECK SHAPES", original_train.shape, mean_train.shape, var_train.shape, gabor_train.shape)
#Layer 1 result
#train = np.concatenate((original_train, mean_train, var_train, gabor_train), axis=3)
#test = np.concatenate((original_test, mean_test, var_test, gabor_test), axis=3)
#print("layer 1 shapes", layer_1_test.shape, layer_1_train.shape)

#train = np.concatenate((original_train,gabor_train), axis=3)
#test = np.concatenate((original_test, gabor_test), axis=3)

##improve sampling for train, more convenient to do when everything is stacked already albeit less efficient
#resampler(layer_1_train, y_train)


#autoencoder(layer_1_train, y_train, layer_1_test, y_test)
#results = ["accuracy", "optimizer", "activation", "norms"]

#optimizer = ["rmsprop"]
#activations = ["sigmoid"]

#for norms in np.arange(0.2, 2, 0.2):
#    for activation in activations:
#            for opt in optimizer:
#                row = [run_network(train, test, y_train, y_test, norms, activation, opt), opt, activation, norms]
#                results.append(row)
#np.savetxt("results.txt", np.stack(results), delimiter=",", fmt = '% s')

#large_net(original_train, original_test, y_train, y_test)
medium_net(train, test, y_train, y_test)

#model = tf.keras.applications.VGG16(
#    weights=None,
#    input_tensor=original_train,
#    input_shape=(100,100,1)
#)
#model.build()
#print(model.summary())

#X, y = datasets.make_classification(n_samples=1389, n_features=1)
