#!/usr/bin/python3

import math

def euclidean_distance(data1, data2):
    return math.sqrt((data2[0] - data1[0]) ** 2 + (data2[1] - data1[1]) ** 2)

def knn(training_data, training_label, test_data, k=3):
    test_data_predictions = []

    for test_item in test_data:
        test_item_predictions = []
        for i, training_item in enumerate(training_data):
            distance = euclidean_distance(test_item, training_item)
            result = {
                'training_item': training_item,
                'distance': distance,
                'neighbor_label': training_label[i]
            }
            test_item_predictions.append(result)

        sorted_predictions = sorted(test_item_predictions, key=lambda result: result['distance'])
        
        class_vote = {}
        for i, item in enumerate(sorted_predictions):
            if i >= k:
                break
            if item['neighbor_label'] in class_vote:
                class_vote[item['neighbor_label']] += 1
            else:
                class_vote[item['neighbor_label']] = 1
        
        predicted_class = max(class_vote.items(), key=lambda item: item[1])[0]
        test_data_predictions.append(int(predicted_class))

    return test_data_predictions



if __name__ == '__main__':
    training_data = [
        [0, 0],
        [0, 0.1],
        [0.2, 0.2],
        [0.4, 0.5],
        [1, 1],
        [0.9, 0.9],
        [0.6, 0.4],
        [0.2, 0.9]
    ]
    training_label = [0, 0, 0, 0, 1, 1, 1, 1]
    test_data = [
        [0.3, 0.3],
        [0.8, 0.8],
        [0.3, 0.4],
        [0.1, 0.15],
        [0.78, 0.5]
    ]
    predictions = knn(training_data, training_label, test_data)
    print('Test data = ')
    print(test_data)
    print('Predictions = ')
    print(predictions)
