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

def glcm_mean():
    #x, y both <= 99
    current = [0,0]
    while not current[1] == 100:
        print("current", current)
        current = update_point(current)


def glcm_variance():
    arr = []


def gabor():
    arr = []

glcm_mean()