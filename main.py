import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from tensorflow.keras.layers import  AveragePooling2D, MaxPooling2D

import dataloader as d
from visualize import confusion, plot_images
from filters import compute_glcm_results, gabor, glcm_no_save
from network import run_network
from autoencoder import autoencoder
from large_net import large_net
#from resampler import resampler
#We expect tensorflow >2.3.2, preferably 2.4 or greater
print(tf.__version__)

##any config values
test_percentage = 0.2
#input_shape = (100,100,7)
band = 0

#x_train, y_train, x_test, y_test = d.get_manual_calib_data(test_percentage, band)

#I heard that laziness is a virtue and programmers are lazy people
x_train, y_train, x_test, y_test = d.get_prepared_data()



#np.savetxt("train_onehot.csv", y_train, delimiter=",", fmt = '% s')
#np.savetxt("test_onehot.csv", y_train, delimiter=",", fmt = '% s')

x_train = x_train * 6000
x_test = x_test * 6000

#scaler1 = preprocessing.StandardScaler().fit(x_train)
#scaler2 = preprocessing.StandardScaler().fit(x_test)
#x_train = scaler1.transform(x_train)
#x_test = scaler2.transform(x_test)

##this section will include glcm+gabor filter computation / loading ##
#compute glcm mea+varn and store in file for later use, laziness level 2.0
mean_train, var_train, mean_test, var_test = compute_glcm_results(x_train, x_test)
#mean_train, var_train, mean_test, var_test = d.get_prepared_glcm()
#mean_train, var_train, mean_test, var_test = glcm_no_save(x_train, x_test)
gabor_train = np.stack(gabor(x_train, "x_train"))
gabor_test = np.stack(gabor(x_test, "x_test"))
original_train = x_train[..., np.newaxis]
original_test = x_test[..., np.newaxis]


#print("CHECK SHAPES", original_train.shape, mean_train.shape, var_train.shape, gabor_train.shape)
#Layer 1 result
train = np.concatenate((original_train, mean_train, var_train, gabor_train), axis=3)
test = np.concatenate((original_test, mean_test, var_test, gabor_test), axis=3)
#print("layer 1 shapes", layer_1_test.shape, layer_1_train.shape)

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

large_net(train, test, y_train, y_test)