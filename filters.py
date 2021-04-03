import math
import numpy as np
from dataloader import get_prepared_means
import multiprocessing as mp
from gabor_filter_parts import gaborize_image
from glcm_filter_parts import glcm_apply_mean, glcm_apply_variance


#Generic instructions for custom keras filter
#https://stackoverflow.com/questions/51930312/how-to-include-a-custom-filter-in-a-keras-based-cnn
#We probably want to just return a numpy array and stack the GLCM and Gabor outputs.


def glcm_mean(images):
    #images with mean values
    cpu_pool = mp.Pool(mp.cpu_count())
    filtered_images = cpu_pool.map(glcm_apply_mean,[image for image in images])
    cpu_pool.close()
    filtered_stack = np.stack(filtered_images)
    return filtered_stack[..., np.newaxis]


def glcm_variance(images, means):
    cpu_pool = mp.Pool(mp.cpu_count())
    filtered_images = cpu_pool.map(glcm_apply_variance(images[id], means[id]), [int(id) for id in range(images.shape[0])])
    cpu_pool.close()
    filtered_stack = np.stack(filtered_images)
    return filtered_stack[..., np.newaxis]

def glcm_no_save(x_train, x_test):
    print("mean train")
    mean_train = glcm_mean(x_train)
    print("mean test")
    mean_test = glcm_mean(x_test)
    print("var train")
    var_train = glcm_variance(x_train, mean_train)
    print("var test")
    var_test = glcm_variance(x_test, mean_test)

    return mean_train, var_train, mean_test, var_test

def compute_glcm_results(x_train, x_test):
    mean_train = glcm_mean(x_train)
    mean_test = glcm_mean(x_test)

    batch1 = int(12860/10)
    batch2 = int(batch1*2)
    batch3 = int(batch1*3)
    batch4 = int(batch1*4)
    batch5 = int(batch1*5)
    batch6 = int(batch1*6)
    batch7 = int(batch1*7)
    batch8 = int(batch1*8)
    batch9 = int(batch1*9)
    mean_train1 = mean_train[:batch1]
    mean_train2 = mean_train[batch1:batch2]
    mean_train3 = mean_train[batch2:batch3]
    mean_train4 = mean_train[batch3:batch4]
    mean_train5 = mean_train[batch4:batch5]
    mean_train6 = mean_train[batch5:batch6]
    mean_train7 = mean_train[batch6:batch7]
    mean_train8 = mean_train[batch7:batch8]
    mean_train9 = mean_train[batch8:batch9]
    mean_train10 = mean_train[batch9:]
    triplet1 = int(3215/3)
    triplet2 = int(triplet1*2)
    mean_test1 = mean_test[:triplet1]
    mean_test2 = mean_test[triplet1:triplet2]
    mean_test3 = mean_test[triplet2:]

    np.save("glcm/mean_train1", mean_train1)
    np.save("glcm/mean_train2", mean_train2)
    np.save("glcm/mean_train3", mean_train3)
    np.save("glcm/mean_train4", mean_train4)
    np.save("glcm/mean_train5", mean_train5)
    np.save("glcm/mean_train6", mean_train6)
    np.save("glcm/mean_train7", mean_train7)
    np.save("glcm/mean_train8", mean_train8)
    np.save("glcm/mean_train9", mean_train9)
    np.save("glcm/mean_train10", mean_train10)
    np.save("glcm/mean_test1", mean_test1)
    np.save("glcm/mean_test2", mean_test2)
    np.save("glcm/mean_test3", mean_test3)
    
    #uncomment below if you have means already but need to recompute variances
    #mean_train, mean_test = get_prepared_means()
    print("var_train")
    var_train = glcm_variance(x_train, mean_train)
    print("var_test")
    var_test = glcm_variance(x_test, mean_test)
    var_train1 = var_train[:batch1]
    var_train2 = var_train[batch1:batch2]
    var_train3 = var_train[batch2:batch3]
    var_train4 = var_train[batch3:batch4]
    var_train5 = var_train[batch4:batch5]
    var_train6 = var_train[batch5:batch6]
    var_train7 = var_train[batch6:batch7]
    var_train8 = var_train[batch7:batch8]
    var_train9 = var_train[batch8:batch9]
    var_train10 = var_train[batch9:]
    var_test1 = var_test[:triplet1]
    var_test2 = var_test[triplet1:triplet2]
    var_test3 = var_test[triplet2:]

    np.save("glcm/var_train1", var_train1)
    np.save("glcm/var_train2", var_train2)
    np.save("glcm/var_train3", var_train3)
    np.save("glcm/var_train4", var_train4)
    np.save("glcm/var_train5", var_train5)
    np.save("glcm/var_train6", var_train6)
    np.save("glcm/var_train7", var_train7)
    np.save("glcm/var_train8", var_train8)
    np.save("glcm/var_train9", var_train9)
    np.save("glcm/var_train10", var_train10)
    np.save("glcm/var_test1", var_test1)
    np.save("glcm/var_test2", var_test2)
    np.save("glcm/var_test3", var_test3)
    return


def gabor(images, name_type):
    cpu_pool = mp.Pool(mp.cpu_count())
    gabor_images = cpu_pool.map(gaborize_image,[image for image in images])
    cpu_pool.close()
    return gabor_images
