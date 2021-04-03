import math
import numpy as np
import cv2
from dataloader import get_prepared_means
import multiprocessing as mp

window = 3
window_ct = math.pow(2*window+1, 2)
mean_weight = 1/window_ct
var_weight = 1/pow(window_ct,2)

mp.Pool(mp.cpu_count())

def update_point(current):
    x = 0
    y = 0
    if current[0] == 99:
        y = current[1] + 1
        x = 0
    else:
        y = current[1]
        x = current[0] + 1
    return (x,y)

#Generic instructions for custom keras filter
#https://stackoverflow.com/questions/51930312/how-to-include-a-custom-filter-in-a-keras-based-cnn
#We probably want to just return a numpy array and stack the GLCM and Gabor outputs.

def glcm_apply_mean(image):
    #x, y both <= 99
    current = [0,0]
    #init array for mean result
    mean_array = np.zeros((100,100))
    while not current[1] == 100:
        values = []
        start_x = current[0] - 1
        start_y = current[1] - 1
        for x in range(window):
            for y in range(window):
                check_x = start_x + x
                check_y = start_y + y
                #append padded value if beyond borders
                if check_x < 0 or check_y < 0 or check_x > 99 or check_y > 99:
                    values.append(0)
                else:
                    values.append(image[check_x, check_y])
        #compute GLCM mean for coordinate x,y with window size w
        mean_result = mean_weight * np.sum(values)
        #append value to the point for original image center point
        mean_array[current[0], current[1]] = mean_result
        #refresh to next index and repeat
        current = update_point(current)
    return mean_array

def glcm_mean(images):
    #images with mean values
    filtered_images = []
    id = 0
    for image in images:
        if (id % 100) == 0:
            print("current glcm mean id", id)
        filtered_images.append(glcm_apply_mean(image))
        id = id + 1
    
    return np.stack(filtered_images)

def glcm_apply_variance(image, mean):
     #x, y both <= 99
    current = [0,0]
    #init array for mean result
    variance_array = np.zeros((100,100))
    while not current[1] == 100:
        values = []
        start_x = current[0] - 1
        start_y = current[1] - 1
        for x in range(window):
            for y in range(window):
                check_x = start_x + x
                check_y = start_y + y
                #append padded value if beyond borders
                if check_x < 0 or check_y < 0 or check_x > 99 or check_y > 99:
                    values.append(0)
                else:
                    val = image[check_x, check_y]
                    m = mean[check_x, check_y]
                    values.append(math.pow(val-m,2) )
        variance_array[current[0], current[1]] = np.sum(values) * var_weight
        current = update_point(current)
    return variance_array

def glcm_variance(images, means):
    filtered_images = []
    for i in range(int(images.shape[0])):
        if (i % 100) == 0:
            print("current glcm var id", i)
        filtered_images.append(glcm_apply_variance(images[i], means[i]))

    return np.stack(filtered_images)

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

def gabor_filters():
    filters = []
    glambda = np.pi / 2
    #rotate over half-circle at 12.5 deg increments
    for theta in np.arange(0, np.pi, np.pi/16):
        kernel = cv2.getGaborKernel((window, window), 1.0, theta, glambda, 0, ktype=cv2.CV_32F)
        kernel /= 1.5*kernel.sum()
        filters.append(kernel)
    return filters

def apply_filter(image, kernel):
    result = np.zeros(image)
    #we always apply same kernel
    fimg = cv2.filter2D(image, cv2.CV8UC3, kernel)
    np.maximum(result, fimg, result)
    return result

def gabor(images):
    images_len = images.shape[0]
    filters = gabor_filters()
    filt_len = len(filters)
    arr = np.zeros((images_len, 100, 100, filt_len))
    for id in range(images.shape[0]):
        filter_arr = arr[id]
        for fid in range(filt_len):
            res = apply_filter(images[id], filters[fid])
            filter_arr[:][:][fid] = res
        arr[id] = filter_arr
        if (id % 100) == 0:
            print("id:", id)
    np.save("gabor/gabor_res", arr)
