import numpy as np
import math

window = 3
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
