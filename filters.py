import math

window = 3
window_ct = math.pow(2*window+1, 2)
mean_weight = 1/window_ct

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
    while not current[1] == 100:
        center_point = image[current[0], current[1]]
        print("current", current)
        current = update_point(current)

def glcm_filters(images):
    filtered_images = []
    for image in images:
        filtered_images.append(glcm_apply_mean(image))

def glcm_apply_variance(images):
    current = [0,0]

def glcm_variance(images):
    filtered_images = []
    for image in images:
        filtered_images.append(glcm_variance(image))

def gabor():
    arr = []

glcm_mean()