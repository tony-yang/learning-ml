import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
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
    features_train_sig = np.c_[bumpy_sig, grade_sig]
    features_train_bkg = np.c_[bumpy_bkg, grade_bkg]

    features_train = np.concatenate((features_train_sig, features_train_bkg), axis=0)
    labels_train = [1]*len(bumpy_sig) + [2] * len(bumpy_bkg)

    return features_train, labels_train



def main():
    n_points = 1000
    random.seed(42)
    grade = [random.random() for ii in range(0,n_points)]
    bumpy = [random.random() for ii in range(0,n_points)]
    error = [random.random() for ii in range(0,n_points)]
    y = [round(grade[ii]*bumpy[ii]+0.3+0.1*error[ii]) for ii in range(0,n_points)]
    for ii in range(0, len(y)):
        if grade[ii]>0.8 or bumpy[ii]>0.8:
            y[ii] = 1.0

### split into train/test sets
    X = [[gg, ss] for gg, ss in zip(grade, bumpy)]
    split = int(0.75*n_points)
    X_train = X[0:split]
    X_test  = X[split:]
    y_train = y[0:split]
    y_test  = y[split:]

    grade_sig = [X_train[ii][0] for ii in range(0, len(X_train)) if y_train[ii]==0]
    bumpy_sig = [X_train[ii][1] for ii in range(0, len(X_train)) if y_train[ii]==0]
    grade_bkg = [X_train[ii][0] for ii in range(0, len(X_train)) if y_train[ii]==1]
    bumpy_bkg = [X_train[ii][1] for ii in range(0, len(X_train)) if y_train[ii]==1]

    training_data = {"fast":{"grade":grade_sig, "bumpiness":bumpy_sig}
            , "slow":{"grade":grade_bkg, "bumpiness":bumpy_bkg}}

    features_train, labels_train = create_label(bumpy_sig, grade_sig, bumpy_bkg, grade_bkg)

    grade_sig = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]
    bumpy_sig = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==0]
    grade_bkg = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==1]
    bumpy_bkg = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==1]

    test_data = {"fast":{"grade":grade_sig, "bumpiness":bumpy_sig}
            , "slow":{"grade":grade_bkg, "bumpiness":bumpy_bkg}}

    features_test, labels_test = create_label(bumpy_sig, grade_sig, bumpy_bkg, grade_bkg)


    clf = svm.SVC(kernel='rbf', gamma=10)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)


    x_min = 0.0; x_max = 1.0
    y_min = 0.0; y_max = 1.0

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    h = .01  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the test points
    grade_sig = [features_test[ii][0] for ii in range(0, len(features_test)) if labels_test[ii]==0]
    bumpy_sig = [features_test[ii][1] for ii in range(0, len(features_test)) if labels_test[ii]==0]
    grade_bkg = [features_test[ii][0] for ii in range(0, len(features_test)) if labels_test[ii]==1]
    bumpy_bkg = [features_test[ii][1] for ii in range(0, len(features_test)) if labels_test[ii]==1]

    print('============')
    print(grade_sig)
    plt.scatter(grade_sig, bumpy_sig, color = "b", label="fast")
    plt.scatter(grade_bkg, bumpy_bkg, color = "r", label="slow")
    plt.legend()
    plt.xlabel("bumpiness")
    plt.ylabel("grade")



    plt.show()
    acc = accuracy_score(pred, labels_test)
    print('Accuracy = {}'.format(acc))


if __name__ == '__main__':
    main()
