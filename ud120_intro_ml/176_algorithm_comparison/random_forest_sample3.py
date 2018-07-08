import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import urllib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def print_dx_percentage(dx, data_frame, column):
    print(data_frame[column])
    print(data_frame[column].value_counts())
    dx_vals = data_frame[column].value_counts()
    print('=========')
    dx_vals = dx_vals.reset_index()
    print(dx_vals)

    f = lambda x, y: 100 * (x / sum(y))
    for i in range(len(dx)):
        print(dx[i])
        print(dx_vals[column].iloc[i])
        print('==========')
        print(dx_vals[column][i])
        print('##########')
        print(dx_vals[column])
        print('{} accounts for {:.2f}% of the diagnosis class'.format(dx[i], f(dx_vals[column].iloc[i], dx_vals[column])))

if __name__ == '__main__':
    plt.style.use('ggplot')
    pd.set_option('display.max_columns', 500)

    names = ['id_number', 'diagnosis', 'radius_mean',
            'texture_mean', 'perimeter_mean', 'area_mean',
            'smoothness_mean', 'compactness_mean',
            'concavity_mean','concave_points_mean',
            'symmetry_mean', 'fractal_dimension_mean',
            'radius_se', 'texture_se', 'perimeter_se',
            'area_se', 'smoothness_se', 'compactness_se',
            'concavity_se', 'concave_points_se',
            'symmetry_se', 'fractal_dimension_se',
            'radius_worst', 'texture_worst',
            'perimeter_worst', 'area_worst',
            'smoothness_worst', 'compactness_worst',
            'concavity_worst', 'concave_points_worst',
            'symmetry_worst', 'fractal_dimension_worst']

    dx = ['Benign', 'Malignant']

    breast_cancer = pd.read_csv('wdbc.csv', names=names)
    breast_cancer.set_index(['id_number'], inplace=True)
    breast_cancer['diagnosis'] = breast_cancer['diagnosis'].map({'M':1, 'B':0})
    print(breast_cancer.describe())

    names_index = names[2:]
    print(breast_cancer.head())

    print('Here is the dimension: {}'.format(breast_cancer.shape))
    print('Here is the data type: {}'.format(breast_cancer.dtypes))

    print_dx_percentage(dx, breast_cancer, 'diagnosis')

    feature_space = breast_cancer.iloc[:, breast_cancer.columns != 'diagnosis']
    feature_class = breast_cancer.iloc[:, breast_cancer.columns == 'diagnosis']

    training_set, test_set, class_set, test_class_set = train_test_split(feature_space, feature_class, test_size=0.2, random_state=42)

    class_set = class_set.values.ravel()
    test_class_set = test_class_set.values.ravel()

    fit_rf = RandomForestClassifier(random_state=42)

    np.random.seed(42)
    start = time.time()

    param_dist = {
        'max_depth': [2, 3, 4],
        'bootstrap': [True, False],
        'max_features': ['auto', 'sqrt', 'log2', None],
        'criterion': ['gini', 'entropy']
    }

    # cv_rf = GridSearchCV(fit_rf, cv=10, param_grid=param_dist, n_jobs=3)
    # cv_rf.fit(training_set, class_set)
    # print('Best Param using grid search {}'.format(cv_rf.best_params_))
    end = time.time()
    print('Time taken in grid search {:.2f}'.format(end-start))

    fit_rf.set_params(max_features='log2', max_depth=3)
    print(fit_rf)

    fit_rf.set_params(warm_start=True, oob_score=True)
    min_estimators = 15
    max_estimators = 1000
    error_rate = {}
    for i in range(min_estimators, max_estimators + 1):
        fit_rf.set_params(n_estimators=i)
        fit_rf.fit(training_set, class_set)

        oob_error = 1 - fit_rf.oob_score_
        error_rate[i] = oob_error
    
    oob_series = pd.Series(error_rate)
    fig, ax = plt.subplots(figsize=(10, 10))
    oob_series.plot(kind='line', color='red')
    plt.axhline(0.055, color='#875fdb', linestyle='--')
    plt.axhline(0.05, color='#875fdb', linestyle='--')
    plt.xlabel('n_estimators')
    plt.ylabel('OOB Error Rate')
    plt.title('OOB Error Rate Across various Forest Size')
    plt.show()
