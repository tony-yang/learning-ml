#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

def pretty_picture(clf, x_test, y_test):
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0

    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.seismic)

    grade_sig = [x_test[i][0] for i in range(len(x_test)) if y_test[i] == 0]
    bumpy_sig = [x_test[i][1] for i in range(len(x_test)) if y_test[i] == 0]
    grade_bkg = [x_test[i][0] for i in range(len(x_test)) if y_test[i] == 1]
    bumpy_bkg = [x_test[i][1] for i in range(len(x_test)) if y_test[i] == 1]

    plt.scatter(grade_sig, bumpy_sig, color='b', label='fast')
    plt.scatter(grade_bkg, bumpy_bkg, color='r', label='slow')
    plt.legend()
    plt.xlabel('grade')
    plt.ylabel('bumpiness')

    plt.show()
    plt.draw()