# !/usr/bin/python3
import numpy as np
import math

from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    def __init__(self, num_of_hypotheses=10, base_classifier=DecisionTreeClassifier(max_depth=1, random_state=1)):
        self.alphas = []
        self.data_weights = []
        self.num_of_hypotheses = num_of_hypotheses
        self.base_classifier = base_classifier

    def error_rate(self, prediction, actual):
        return sum(prediction != actual) / float(len(actual))
    
    def generate_new_data_weights(self, data_weights, alpha, correct_classification_in_integer, num_of_data_points):
        new_data_weights = []
        wrong_classification_index = []
        correct_classification_index = []

        for i in range(num_of_data_points):
            if correct_classification_in_integer[i] < 0:
                wrong_classification_index.append(i)
            else:
                correct_classification_index.append(i)
        
        wrong_classification_scale = 0.5 / sum([data_weights[i] for i in wrong_classification_index])
        correct_classification_scale = 0.5 / sum([data_weights[i] for i in correct_classification_index])
        
        for i in range(num_of_data_points):
            if correct_classification_in_integer[i] < 0:
                new_data_weights.append(data_weights[i] * wrong_classification_scale)
            else:
                new_data_weights.append(data_weights[i] * correct_classification_scale)

        return new_data_weights

    def weighted_prediction_error(self, data_weights, incorrect_classification, num_of_data_points):
        errors = 0
        for i in range(num_of_data_points):
            errors += data_weights[i] * incorrect_classification[i]
        return errors

    def fit(self, train_x, train_y):
        self.x = train_x
        self.y = train_y
        num_of_data_points = len(train_x)
        data_weights = [1.0 / num_of_data_points for i in range(num_of_data_points)]
        prediction_from_train_x = [0.0 for i in range(num_of_data_points)]

        for i in range(self.num_of_hypotheses):
            self.data_weights.append(data_weights)
            self.base_classifier.fit(train_x, train_y, sample_weight=data_weights)
            hypothesis_i_prediction = self.base_classifier.predict(train_x)
            incorrect_classification = [result for result in (hypothesis_i_prediction != train_y)]
            correct_classification_in_integer = [-2 * item + 1 for item in incorrect_classification]
            error = self.weighted_prediction_error(data_weights, incorrect_classification, num_of_data_points)
            normalized_error = error / sum(data_weights)
            alpha = 0.5 * math.log((1 - normalized_error) / float(normalized_error))
            self.alphas.append(alpha)
            data_weights = self.generate_new_data_weights(data_weights, alpha, correct_classification_in_integer, num_of_data_points)
            # TODO: Seems like these last 2 lines are unneccessary? To improve performance maybe remove them?
            hypothesis_i_prediction = [item if item > 0 else -1 for item in hypothesis_i_prediction]
            prediction_from_train_x = [sum(aggregated_prediction) for aggregated_prediction in zip(prediction_from_train_x, [prediction_item * alpha for prediction_item in hypothesis_i_prediction])]
        
        # TODO: Seems like the final prediction and error rate are also unnecessary other than for reporting purpose
        final_prediction = np.sign(prediction_from_train_x)
        return self.error_rate(final_prediction, train_y)

    def normalize_final_prediction(self, final_prediction):
        return [1 if item > 0 else 0 for item in final_prediction]

    def predict(self, test_x):
        num_of_data_points = len(test_x)
        prediction_from_test_x = [0 for i in range(num_of_data_points)]
        for i in range(self.num_of_hypotheses):
            self.base_classifier.fit(self.x, self.y, self.data_weights[i])
            hypothesis_i_prediction = self.base_classifier.predict(test_x)
            hypothesis_i_prediction = [item if item > 0 else -1 for item in hypothesis_i_prediction]
            prediction_from_test_x = [sum(aggregated_prediction) for aggregated_prediction in zip(prediction_from_test_x, [prediction_item * self.alphas[i] for prediction_item in hypothesis_i_prediction])]
        
        final_prediction = self.normalize_final_prediction(prediction_from_test_x)
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

    adaboost = AdaBoost()
    final_error_rate = adaboost.fit(train_x, train_y)
    print(adaboost.alphas)
    print('final_error_rate = {}'.format(final_error_rate))

    test_x = [
        [1.2, 2.1],
        [1, 4.5],
        [2.5, 4.5],
        [2.5, 1],
        [5.1, 2.2]
    ]
    final_prediction = adaboost.predict(test_x)
    print('final prediction = {}'.format(final_prediction))