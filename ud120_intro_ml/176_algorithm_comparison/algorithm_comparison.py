#!/usr/bin/python3
import matplotlib.pyplot as plt
from prep_terrain_data import make_terrain_data
from prep_terrain_data import make_terrain_data2
from class_vis import pretty_picture
import time

from adaboost import AdaBoost
from knn import knn
from random_forest import RandomForest

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
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

gaussian_nb_begin = time.time()
clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(labels_test, pred)
gaussian_nb_end = time.time()
print('====== Gaussian Naive Bayes ======')
print('The Gaussian naive bayes classification accuracy = {}'.format(accuracy))
print('Training and prediction time = {}'.format(gaussian_nb_end - gaussian_nb_begin))

# KNN custom implementation
custom_knn_begin = time.time()
pred = knn(features_train, labels_train, features_test, k=5)
accuracy = accuracy_score(labels_test, pred)
custom_knn_end = time.time()
print('====== Custom KNN ======')
print('The custom KNN classification accuracy = {}'.format(accuracy))
print('Training and prediction time = {}'.format(custom_knn_end - custom_knn_begin))

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

# AdaBoost custom implementation
custom_adaboost_begin = time.time()
custom_adaboost = AdaBoost(num_of_hypotheses=100)
custom_adaboost.fit(features_train, labels_train)
pred = custom_adaboost.predict(features_test)
accuracy = accuracy_score(labels_test, pred)
custom_adaboost_end = time.time()
print('====== Custom AdaBoost ======')
print('The custom AdaBoost classification accuracy = {}'.format(accuracy))
print('Training and prediction time = {}'.format(custom_adaboost_end - custom_adaboost_begin))

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

# RandomForest custom implementation
custom_rf_begin = time.time()
custom_rf = RandomForest(n_estimators=100, max_depth=10, random_state=42)
custom_rf.fit(features_train, labels_train)
pred = custom_rf.predict(features_test)
accuracy = accuracy_score(labels_test, pred)
custom_rf_end = time.time()
print('====== Custom Random Forest ======')
print('The custom Random Forest classification accuracy = {}'.format(accuracy))
print('Training and prediction time = {}'.format(custom_rf_end - custom_rf_begin))

# RandomForest scikit-learn
rf_begin = time.time()
clf4 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf4.fit(features_train, labels_train)
pred = clf4.predict(features_test)
accuracy = accuracy_score(labels_test, pred)
rf_end = time.time()
print('====== sklearn Random Forest ======')
print('The sklearn Random Forest classification accuracy = {}'.format(accuracy))
print('Training and prediction time = {}'.format(rf_end - rf_begin))

try:
    pretty_picture(clf, features_test, labels_test)
    pretty_picture(clf2, features_test, labels_test)
    pretty_picture(clf3, features_test, labels_test)
    pretty_picture(clf4, features_test, labels_test)
except NameError as e:
    print('There is a name error: {}'.format(e))