import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import tree
import pprint

pp = pprint.PrettyPrinter(indent=4)

def make_terrain_data(n_points=1000):
    random.seed(42)
    bumpy_sig = [random.uniform(0, 0.9) for i in range(n_points)]
    bumpy_bkg = [random.random() for i in range(n_points)]

    grade_sig = []
    grade_bkg = []

    for x in bumpy_sig:
        error = random.uniform(0, 0.1)
        grade_sig.append(max(0, random.uniform(0.0, (1.0 - x)) - error))

    for x in bumpy_bkg:
        error = random.uniform(0, 0.03)
        grade_bkg.append(min(1, random.uniform((1.0 - x), 1.0) - error))

    return bumpy_sig, grade_sig, bumpy_bkg, grade_bkg

def create_label(bumpy_sig, grade_sig, bumpy_bkg, grade_bkg):
    features_train_sig = np.c_[bumpy_sig, grade_sig]
    features_train_bkg = np.c_[bumpy_bkg, grade_bkg]

    features_train = np.concatenate((features_train_sig, features_train_bkg), axis=0)
    labels_train = [1]*len(bumpy_sig) + [2] * len(bumpy_bkg)

    return features_train, labels_train

def main():
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0
    bumpy_sig, grade_sig, bumpy_bkg, grade_bkg = make_terrain_data(200)

    # Plot the data
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend()
    plt.xlabel('bumpiness')
    plt.ylabel('grade')

    # Train the data
    features_train, labels_train = create_label(bumpy_sig, grade_sig, bumpy_bkg, grade_bkg)

    test_bumpy_sig, test_grade_sig, test_bumpy_bkg, test_grade_bkg = make_terrain_data(200)
    features_test, labels_test = create_label(test_bumpy_sig, test_grade_sig, test_bumpy_bkg, test_grade_bkg)

    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)

    print('The prediction for the features test using NB = ')
    pp.pprint(pred)

    score = accuracy_score(labels_test, pred)
    print('The prediction accuracy score = {}'.format(score))

    clfSVM = svm.SVC(kernel='linear', gamma=1.0, C=0.1)
    clfSVM.fit(features_train, labels_train)
    predSVM = clfSVM.predict(features_test)

    print('The prediction for the features test using SVM = ')
    pp.pprint(predSVM)

    scoreSVM = accuracy_score(labels_test, predSVM)
    print('The prediction accuracy score = {}'.format(scoreSVM))

    clfTree = tree.DecisionTreeClassifier(min_samples_split=2)
    clfTree.fit(features_train, labels_train)
    predTree = clfTree.predict(features_test)

    print('The prediction for the features test using DecisionTree =')
    print(predTree)

    scoreTree = accuracy_score(labels_test, predTree)
    print('The prediction accuracy score for Decision Tree = {}'.format(scoreTree))

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    features_test2, labels_test2 = create_label(XX.ravel(), YY.ravel(), [], [])
    Z = clfTree.predict(features_test2)
    Z = Z.reshape(XX.shape)

    plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)
    plt.scatter(bumpy_sig, grade_sig, color='b', label='fast')
    plt.scatter(bumpy_bkg, grade_bkg, color='r', label='slow')

    plt.show()

if __name__ == '__main__':
    main()
