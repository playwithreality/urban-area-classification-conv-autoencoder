from sklearn.metrics import confusion_matrix
import numpy as np


labels = ["airport",
  "denselow",
  "GeneralResidential",
  "highbuildings",
  "highway",
  "railway",
  "SingleBuilding",
  "Skyscraper",
  "StorageArea",
  "vegetation"]

#still need to implement visualization with proper str labels
#probably using seaborn library
def confusion(predict, actual):
    actual_classes = []
    for a in actual:
        maxit = np.argmax(a)
        actual_classes.append(np.argmax(a))
    actual_classes = np.stack(actual_classes)
    res = confusion_matrix(predict, actual_classes)
    print(res)

