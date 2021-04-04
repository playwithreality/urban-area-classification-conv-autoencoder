import tensorflow as tf
import numpy as np

import dataloader as d
from visualize import confusion, plot_images
from filters import compute_glcm_results, gabor, glcm_no_save
from network import run_network
from autoencoder import autoencoder
#We expect tensorflow >2.3.2, preferably 2.4 or greater
print(tf.__version__)

##any config values
test_percentage = 0.2
#input_shape = (100,100,7)
band = 0

#train_ds, validation_ds = d.rgb_loader()
#train_ds, val_ds = d.calib_loader()
#d.get_manual_calib_data(test_percentage, band)

#I heard that laziness is a virtue and programmers are lazy people
x_train, y_train, x_test, y_test = d.get_prepared_data()

##this section will include glcm+gabor filter computation / loading ##
#compute glcm mea+varn and store in file for later use, laziness level 2.0
#compute_glcm_results(x_train, x_test)
mean_train, var_train, mean_test, var_test = d.get_prepared_glcm()
#mean_train, var_train, mean_test, var_test = glcm_no_save(x_train, x_test)
gabor_train = np.stack(gabor(x_train, "x_train"))
gabor_test = np.stack(gabor(x_test, "x_test"))
original_train = x_train[..., np.newaxis]
original_test = x_test[..., np.newaxis]

#Layer 1 result
layer_1_train = np.concatenate((original_train, mean_train, var_train, gabor_train), axis=3)
layer_1_test = np.concatenate((original_test, mean_test, var_test, gabor_test), axis=3)
print("layer 1 shapes", layer_1_test.shape, layer_1_train.shape)


#autoencoder(layer_1_train, y_train, layer_1_test, y_test)
run_network(layer_1_train, layer_1_test, y_train, y_test)
