#!/usr/bin/python3
import matplotlib.pyplot as plt
from data.prep_terrain_data import make_terrain_data
from data.prep_terrain_data import make_terrain_data2
from data.class_vis import pretty_picture

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

features_train, labels_train, features_test, labels_test = make_terrain_data()
#features_train, labels_train, features_test, labels_test = make_terrain_data2()

grade_fast = [features_train[i][0] for i in range(len(features_train)) if labels_train[i] == 0]
bumpy_fast = [features_train[i][1] for i in range(len(features_train)) if labels_train[i] == 0]
grade_slow = [features_train[i][0] for i in range(len(features_train)) if labels_train[i] == 1]
bumpy_slow = [features_train[i][1] for i in range(len(features_train)) if labels_train[i] == 1]

print('features train =')
print(features_train)
print('labels_train')
print(labels_train)
print('grade fast')
print(grade_fast)
print('bumpy_bast')
print(bumpy_fast)
print('grade slow')
print(grade_slow)
print('bumpy slow')
print(bumpy_slow)

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

clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(labels_test, pred)

try:
    pretty_picture(clf, features_test, labels_test)
except NameError as e:
    print('There is a name error: {}'.format(e))