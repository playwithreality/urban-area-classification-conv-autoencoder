from sklearn import preprocessing, ensemble, datasets
import numpy as np
from visualize import confusion

def random_forest(x_train, x_test, y_train, y_test):
    sk_train = x_train.reshape((1389, 10000))
    sk_test = x_test.reshape((348, 10000))
    clf = ensemble.RandomForestClassifier(max_depth=20, random_state=0, n_estimators=200)
    clf.fit(sk_train, y_train)
    pred = clf.predict(sk_test)

    pred = pred.argmax(axis=-1)
    correct = 0
    for i in range(len(y_test)):
        #if i < 20:
            # print(classes[i], np.where(y_test[i] == 1)[0][0])
        #print("----")
        if pred[i] == np.where(y_test[i] == 1)[0][0]:
            correct = correct + 1
    print(correct, "accuracy", correct/len(y_test))
    confusion(pred, y_test)