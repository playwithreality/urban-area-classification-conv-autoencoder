from sklearn.preprocessing import StandardScaler
import numpy as np

def scaler(x):
    scaler = StandardScaler()
    new_x = []
    for image in x:
        new_x.append(scaler.fit_transform(image))
    return np.stack(new_x)