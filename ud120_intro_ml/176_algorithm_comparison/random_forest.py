#!/usr/bin/python3

import math
import numpy as np
import random

from sklearn.tree import DecisionTreeClassifier

class RandomForest:
    def __init__(self, n_estimators=11, criterion='gini', max_depth=None, max_features=None, random_state=None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth

        max_features_whitelist = ['auto', 'sqrt', 'log2', None]
        if max_features in max_features_whitelist or type(max_features) is int or type(max_features) is float:
            self.max_features = max_features
        else:
            self.max_features = 'log2'

        random.seed(random_state)
        self.trees = []
    
    def get_subset_training_data(self, training_data, training_label):
        data_sample = []
        label_sample = []
        sample_size = len(training_data)
        while len(data_sample) < sample_size:
            index = random.randrange(sample_size)
            data_sample.append(training_data[index])
            label_sample.append(training_label[index])
        return data_sample, label_sample

    def fit(self, training_data, training_label):
        for i in range(self.n_estimators):
            dt_classifier = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth, max_features=self.max_features)

            data_sample, label_sample = self.get_subset_training_data(training_data, training_label)
            dt_classifier.fit(data_sample, label_sample)
            self.trees.append(dt_classifier)
    
    def predict(self, test_x):
        aggregated_predictions = np.array([])
        for tree_classifier in self.trees:
            prediction = tree_classifier.predict(test_x)
            if aggregated_predictions.any():
                aggregated_predictions = np.column_stack((aggregated_predictions, prediction))
            else:
                aggregated_predictions = prediction
        
        final_prediction = []
        for item in aggregated_predictions:
            final_pred = max(set(item), key=list(item).count)
            final_prediction.append(final_pred)
        
        return final_prediction


if __name__ == '__main__':
    train_x = [
        [1, 2],
        [1, 4],
        [2.5, 5.5],
        [2, 1],
        [5, 2],
        [5,2.2]
    ]
    train_y = [
        1, 1, 1, -1, -1, -1
    ]

    rf = RandomForest(n_estimators=11, max_features=None, random_state=42)
    rf.fit(train_x, train_y)

    test_x = [
        [1.2, 2.1], # 1
        [1, 4.5],   # 1
        [2.5, 4.5], # 1
        [2.5, 1],   # -1
        [5.1, 2.2]  # -1
    ]
    final_prediction = rf.predict(test_x)
    print('final prediction = {}'.format(final_prediction))