import math
import numpy as np

window = 3
window_ct = math.pow(2*window+1, 2)
mean_weight = 1/window_ct
var_weight = 1/pow(window_ct,2)

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

def glcm_variance(images, means):
    filtered_images = []
    id = 0
    for i in range(len(images)):
        print("current glcm var id", i)
        filtered_images.append(glcm_variance(images[i], means[i]))

    return np.stack(filtered_images)

def gabor():
    arr = []
