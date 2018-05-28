import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pprint

pp = pprint.PrettyPrinter(indent=4)

def make_terrain_data(n_points=1000):
    random.seed()
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
    features_train = np.array([[0, 0]])
    labels_train = np.array([1])
    for i in range(len(bumpy_sig)):
        features_train = np.append(features_train, [[bumpy_sig[i], grade_sig[i]]], axis=0)

    for i in range(len(bumpy_bkg)):
        features_train = np.append(features_train, [[bumpy_bkg[i], grade_bkg[i]]], axis=0)

    for i in range(len(bumpy_sig)):
        labels_train = np.append(labels_train, [1])

    for i in range(len(bumpy_bkg)):
        labels_train = np.append(labels_train, [2])

    return features_train, labels_train

def main():
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0
    bumpy_sig, grade_sig, bumpy_bkg, grade_bkg = make_terrain_data(375)

    # Plot the data
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.scatter(bumpy_sig, grade_sig, color='b', label='fast')
    plt.scatter(bumpy_bkg, grade_bkg, color='r', label='slow')
    plt.legend()
    plt.xlabel('bumpiness')
    plt.ylabel('grade')
    #plt.show()

    # Train the data
    features_train, labels_train = create_label(bumpy_sig, grade_sig, bumpy_bkg, grade_bkg)

    test_bumpy_sig, test_grade_sig, test_bumpy_bkg, test_grade_bkg = make_terrain_data(200)
    features_test, labels_test = create_label(test_bumpy_sig, test_grade_sig, test_bumpy_bkg, test_grade_bkg)

    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)

    print('The prediction for the features test = ')
    pp.pprint(pred)

    score = accuracy_score(labels_test, pred)
    print('The prediction accuracy score = {}'.format(score))

if __name__ == '__main__':
    main()
