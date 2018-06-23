#!/usr/bin/python3
import matplotlib.pyplot as plt
from prep_terrain_data import make_terrain_data
from prep_terrain_data import make_terrain_data2
from class_vis import pretty_picture
import time

from knn import knn

from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

features_train, labels_train, features_test, labels_test = make_terrain_data()
#features_train, labels_train, features_test, labels_test = make_terrain_data2()

grade_fast = [features_train[i][0] for i in range(len(features_train)) if labels_train[i] == 0]
bumpy_fast = [features_train[i][1] for i in range(len(features_train)) if labels_train[i] == 0]
grade_slow = [features_train[i][0] for i in range(len(features_train)) if labels_train[i] == 1]
bumpy_slow = [features_train[i][1] for i in range(len(features_train)) if labels_train[i] == 1]

x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.scatter(grade_fast, bumpy_fast, color='b', label='fast')
plt.scatter(grade_slow, bumpy_slow, color='r', label='slow')
plt.legend()
plt.xlabel('grade')
plt.ylabel('bumpiness')
plt.show()

gaussianNB_begin = time.time()
clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(labels_test, pred)
gaussianNB_end = time.time()
print('====== Gaussian Naive Bayes ======')
print('The Gaussian naive bayes classification accuracy = {}'.format(accuracy))
print('Training and prediction time = {}'.format(gaussianNB_end - gaussianNB_begin))

# KNN custom implementation
customKNN_begin = time.time()
pred = knn(features_train, labels_train, features_test, k=5)
accuracy = accuracy_score(labels_test, pred)
customKNN_end = time.time()
print('====== Custom KNN ======')
print('The custom KNN classification accuracy = {}'.format(accuracy))
print('Training and prediction time = {}'.format(customKNN_end - customKNN_begin))

# KNN scikit-learn
knn_begin = time.time()
clf2 = KNeighborsClassifier(n_neighbors=5)
clf2.fit(features_train, labels_train)
pred = clf2.predict(features_test)
accuracy = accuracy_score(labels_test, pred)
knn_end = time.time()
print('====== sklearn KNN ======')
print('The sklearn KNN classification accuracy = {}'.format(accuracy))
print('Training and prediction time = {}'.format(knn_end - knn_begin))

# AdaBoost scikit-learn
adaboost_begin = time.time()
clf3 = AdaBoostClassifier(n_estimators=100)
clf3.fit(features_train, labels_train)
pred = clf3.predict(features_test)
accuracy = accuracy_score(labels_test, pred)
adaboost_end = time.time()
print('====== sklearn AdaBoost ======')
print('The sklearn AdaBoost classification accuracy = {}'.format(accuracy))
print('Training and prediction time = {}'.format(adaboost_end - adaboost_begin))


try:
    pretty_picture(clf, features_test, labels_test)
    pretty_picture(clf2, features_test, labels_test)
    pretty_picture(clf3, features_test, labels_test)
except NameError as e:
    print('There is a name error: {}'.format(e))