from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


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
        actual_classes.append(np.argmax(a))
    actual_classes = np.stack(actual_classes)
    res = confusion_matrix(predict, actual_classes)
    print(res)

def plot_images(original, gabor, mean, var):
  plt.imsave("visualizations/original0.png", original[0])
  plt.imsave("visualizations/original1.png", original[1])
  plt.imsave("visualizations/original2.png", original[2])

  #mean0 = mean[0,:,:]
  #plt.imsave("visualizations/mean0.png")
  plt.plot(mean[1])
  plt.savefig("visualizations/mean1.png")
  plt.plot(mean[2])
  plt.savefig("visualizations/mean2.png")

  plt.plot(var[0])
  plt.savefig("visualizations/var0.png")
  plt.plot(var[1])
  plt.savefig("visualizations/var1.png")
  plt.plot(var[2])
  plt.savefig("visualizations/var2.png")

  gabor0 = gabor[0]
  gabor1 = gabor[0]
  gabor2 = gabor[0]

  gabor0_0 = gabor0[:,:,0]
  gabor0_1 = gabor0[:,:,1]
  gabor0_2 = gabor0[:,:,2]
  gabor0_3 = gabor0[:,:,3]

  gabor1_0 = gabor1[:,:,0]
  gabor1_1 = gabor1[:,:,1]
  gabor1_2 = gabor1[:,:,2]
  gabor1_3 = gabor1[:,:,3]

  gabor2_0 = gabor2[:,:,0]
  gabor2_1 = gabor2[:,:,1]
  gabor2_2 = gabor2[:,:,2]
  gabor2_3 = gabor2[:,:,3]

  plt.plot(gabor0_0)
  plt.savefig("visualizations/gabor0_0")
  plt.plot(gabor0_1)
  plt.savefig("visualizations/gabor0_1")
  plt.plot(gabor0_2)
  plt.savefig("visualizations/gabor0_2")
  plt.plot(gabor0_3)
  plt.savefig("visualizations/gabor0_3")

  plt.plot(gabor1_0)
  plt.savefig("visualizations/gabor1_0")
  plt.plot(gabor1_1)
  plt.savefig("visualizations/gabor1_1")
  plt.plot(gabor1_2)
  plt.savefig("visualizations/gabor1_2")
  plt.plot(gabor1_3)
  plt.savefig("visualizations/gabor1_3")

  plt.plot(gabor2_0)
  plt.savefig("visualizations/gabor2_0")
  plt.plot(gabor2_1)
  plt.savefig("visualizations/gabor2_1")
  plt.plot(gabor2_2)
  plt.savefig("visualizations/gabor2_2")
  plt.plot(gabor2_3)
  plt.savefig("visualizations/gabor2_3")