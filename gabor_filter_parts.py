import cv2
import numpy as np

window = 3
current_id = 0

def gabor_filters():
    filters = []
    glambda = np.pi / 2
    #rotate over half-circle at 12.5 deg increments
    for theta in np.arange(0, np.pi, np.pi/4):
        kernel = cv2.getGaborKernel((window, window), 1.0, theta, glambda, 0, ktype=cv2.CV_32F)
        #kernel /= 1.5*kernel.sum()ernel)
        filters.append(kernel)
    return filters

def apply_filter(image, kernel):
    result = np.zeros((100,100))
    #we always apply same kernel
    #print("kernel shape", np.array(kernel).shape)
    fimg = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    np.maximum(result, fimg, result)
    return result


filters = gabor_filters()
filt_len = len(filters)

def gaborize_image(image):
    global current_id
    gabor_arr = []
    for fid in range(filt_len):
        res = apply_filter(image, filters[fid])
        gabor_arr.append(res)
    #gabor_arr.append(gabor_arr)
    #if(current_id % 100) == 0:
    #    print("current_id", current_id)
    #current_id = current_id + 1
    return np.stack(gabor_arr, axis=2)