#!/usr/bin/python3

import numpy as np
from sklearn.tree import DecisionTreeClassifier

import pprint

def get_error_rate(prediction, actual):
    return sum(prediction != actual) / float(len(actual))

def adaboost_clf(train_x, train_y, m, base_clf):
    print('x = {}'.format(train_x))
    print('y = {}'.format(train_y))
    num_of_train = len(train_x)
    print('num of train = {}'.format(num_of_train))

    w = np.ones(num_of_train) / num_of_train
    print('init w = {}'.format(w))

    #pred_train = [np.zeros(num_of_train)]
    pred_train = [0 for i in range(num_of_train)]
    print('pred_train = {}'.format(pred_train))

    print('m = {}'.format(m))
    for i in range(m):
        print('=====================')
        print('w = {}'.format(w))
        base_clf.fit(train_x, train_y, sample_weight = w)
        pred_train_i = base_clf.predict(train_x)
        print('pred_train i = {}'.format(pred_train_i))
        miss = [int(x) for x in (pred_train_i != train_y)]
        miss2 = [x if x == 1 else -1 for x in miss]
        print('miss = {}'.format(miss))
        print('miss2 = {}'.format(miss2))

        err_m = np.dot(w, miss) / sum(w)
        print('error_m = {}'.format(err_m))

        alpha_m = 0.5 * np.log((1 - err_m) / float(err_m))
        print('alpha m = {}'.format(alpha_m))
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        print('The new w = {}'.format(w))

        #pred_train = [sum(x) for x in zip(pred_train, [x * alpha_m for x in pred_train_i])]
        new_alpha_m = []
        for x in pred_train_i:
            print('the item in pred_train_i = {}'.format(x))
            new_alpha_m.append(x * alpha_m)

        print('the new alpha m = {}'.format(new_alpha_m))
        
        new_pred_train = []
        for x in zip(pred_train, new_alpha_m):
            new_pred_train.append(sum(x))
            print('original x = {}'.format(x))
        pred_train = new_pred_train
        print('pred_train = {}'.format(pred_train))

    pred_train = np.sign(pred_train)
    print('##########################')
    print('Final pred train = {}'.format(pred_train))
    return get_error_rate(pred_train, train_y)

if __name__ == '__main__':

    train_x = [
        [1, 2],
        [1, 4],
        [2.5, 5.5],
        [2, 1],
        [5, 2]
    ]
    train_y = [
        1, 1, 1, -1, -1
    ]

    base_clf = DecisionTreeClassifier(max_depth = 1, random_state = 1)
    print('base clf')
    print(base_clf)

    error_in_train = []

    x_range = range(10, 30, 10)
    print('x range')
    print(x_range)

    for i in x_range:
        adaboost_result = adaboost_clf(train_x, train_y, i, base_clf)
        error_in_train.append(adaboost_result)
    
    pp = pprint.PrettyPrinter(indent=4)

    print('the er train result =')
    pp.pprint(error_in_train)