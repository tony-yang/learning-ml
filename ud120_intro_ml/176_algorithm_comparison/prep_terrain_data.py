#!/usr/bin/python3

import numpy as np
import random

def make_terrain_data(n_points=10000):
    random.seed(42)
    grade = [random.random() for i in range(n_points)]
    bumpy = [random.random() for i in range(n_points)]
    error = [random.random() for i in range(n_points)]
    y = [round(grade[i] * bumpy[i] + 0.3 + 0.1 * error[i]) for i in range(n_points)]
    for i in range(len(y)):
        if grade[i] > 0.8 or bumpy[i] > 0.8:
            y[i] = 1.0
    
    x = [[grade_item, bumpy_item] for grade_item, bumpy_item in zip(grade, bumpy)]
    split = int(0.75 * n_points)
    x_train = x[0:split]
    x_test = x[split:]
    y_train = y[0:split]
    y_test = y[split:]

    return x_train, y_train, x_test, y_test

def create_label(bumpy_sig, grade_sig, bumpy_bkg, grade_bkg):
    # features_train_sig = np.c_[bumpy_sig, grade_sig]
    # features_train_bkg = np.c_[bumpy_bkg, grade_bkg]

    features_train_sig = [[grade_item, bumpy_item] for grade_item, bumpy_item in zip(grade_sig, bumpy_sig)]
    features_train_bkg = [[grade_item, bumpy_item] for grade_item, bumpy_item in zip(grade_bkg, bumpy_bkg)]

    # features_train = np.concatenate((features_train_sig, features_train_bkg), axis=0)
    features_train = features_train_sig + features_train_bkg
    labels_train = [0]*len(bumpy_sig) + [1] * len(bumpy_bkg)

    return features_train, labels_train

def create_random_data_points(n_points=50):
    n_points = int(n_points / 2)
    random.seed(42)
    bumpy_sig = [random.uniform(0.1, 0.7) for i in range(n_points)]
    bumpy_bkg = [random.uniform(0.2, 1.0) for i in range(n_points)]

    grade_sig = []
    grade_bkg = []

    for x in bumpy_sig:
        error = random.uniform(0, 0.1)
        grade_sig.append(max(0, random.uniform(0.0, (1.0 - x)) - error))
        #grade_sig.append(random.uniform(0.0, (1.0 - x)) - error)

    for x in bumpy_bkg:
        error = random.uniform(0, 0.05)
        grade_bkg.append(min(1, random.uniform((1.0 - x), 1.0) - error))
        #grade_bkg.append(random.uniform((1.0 - x), 1.0) - error)


    return bumpy_sig, grade_sig, bumpy_bkg, grade_bkg

def make_terrain_data2(n_points=50):
    bumpy_sig, grade_sig, bumpy_bkg, grade_bkg = create_random_data_points()
    features_train, labels_train = create_label(bumpy_sig, grade_sig, bumpy_bkg, grade_bkg)
    bumpy_sig, grade_sig, bumpy_bkg, grade_bkg = create_random_data_points()
    features_test, labels_test = create_label(bumpy_sig, grade_sig, bumpy_bkg, grade_bkg)
    return features_train, labels_train, features_test, labels_test