import csv
import math
import random

import pprint
import time
from prep_terrain_data import make_terrain_data

def load_csv(filename):
    dataset = []
    with open(filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    
    return dataset

def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = {}
    for i, value in enumerate(unique):
        lookup[value] = i
    # Why not OHE?
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

def cross_validation_split(dataset, n_folds):
    # print('    cross validation split called from eval algo')
    dataset_split = []
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    # print('    fold size = {}'.format(fold_size))
    for i in range(n_folds):
        fold = []
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def accuracy_metric(actual, predicted):
    correct = 0
    for i, value in enumerate(actual):
        if value == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    # print('############ main method evaluate algorithm')
    folds = cross_validation_split(dataset, n_folds)
    scores = []
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = [list(row) for row in fold]
        for row in test_set:
            row[-1] = None
        # for row in fold:
        #     print('    row = {}'.format(row))
        #     row_copy = list(row)
        #     test_set.append(row_copy)
        #     row_copy[-1] = None

        # print('args = {}'.format(args))
        # print('########## It calls the algorithm which is random forest')
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

def test_split(index, value, train_set):
    left, right = [], []
    # print('######### test split called from get_split')
    # print('         index = {} and value = {}'.format(index, value))
    for row in train_set:
        # print('         row = {}'.format(row))
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def gini_index(groups, classes):
    # print('############ gini index called from get split')
    # print('classes = {}'.format(classes))
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        # print('         size = {} and n_instance = {}'.format(size, n_instances))
        # print('         group = {}'.format(group))
        for class_val in classes:
            # print('         cal val = {}'.format(class_val))
            p = [row[-1] for row in group].count(class_val) / size
            # print('         p = {}'.format(p))
            score += p * p
        # print('         score = {}'.format(score))
        gini += (1.0 - score) * (size / n_instances)
    # print('         final gini = {}'.format(gini))
    return gini

def get_split(train_set, n_features):
    # print('@@@@@@@@@@@ get split called by build tree')
    class_values = list({row[-1] for row in train_set})
    # print('         class values = {}'.format(class_values))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = []
    while len(features) < n_features:
        index = random.randrange(len(train_set[0])-1)
        if index not in features:
            features.append(index)
    # print('         features = {}'.format(features))
    for index in features:
        for row in train_set:
            # print('=======row in train set=========')
            # print('         row = {} and index = {}'.format(row, index))
            groups = test_split(index, row[index], train_set)
            # print('         result of groups = ')
            # pp.pprint(groups)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
                # print('index = {} value = {} score = {} groups = {}'.format(b_index, b_value, b_score, b_groups))
    # print('     !!!!!!!!!!!!!!!Final value = ')
    # print('     index = {} value = {} score = {} groups = {}'.format(b_index, b_value, b_score, b_groups))
    return {'index': b_index, 'value': b_value, 'score': b_score, 'groups': b_groups}

def to_terminal(group):
    # print('333333333 calling to terminal from split')
    outcomes = [row[-1] for row in group]
    # print('     outcomes = {}'.format(outcomes))
    result = max(set(outcomes), key=outcomes.count)
    # print('     result = {}'.format(result))
    return result

def split(node, max_depth, min_size, n_features, depth):
    # print('####### Final calling split from build tree')
    # print('split:     node = {} max depth = {} min size = {} n features = {} depth = {}'.format(node, max_depth, min_size, n_features, depth))
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        # print('split:       not left or not right')
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        # print('split:       depth > max depth')
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        # print('split:       Starting the get_split on left')
        # print('         left = {}'.format(left))
        node['left'] = get_split(left, n_features)
        # print('split:          the node left after get split score = {}'.format(node['left']['score']))
        split(node['left'], max_depth, min_size, n_features, depth+1)
        # print('split:           node left split done')

    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        # print('split:       Starting the get_split on right')
        # print('         right = {}'.format(right))
        node['right'] = get_split(right, n_features)
        # print('split:          the node right after get split score = {}'.format(node['right']['score']))
        split(node['right'], max_depth, min_size, n_features, depth+1)
        # print('split:           node right split done')

def build_tree(train_set, max_depth, min_size, n_features):
    # print('======== build tree called by random forest')
    # print('         max depth = {} min size = {} n_features ={}'.format(max_depth, min_size, n_features))
    root = get_split(train_set, n_features)
    # print('#### root =')
    # pp.pprint(root)
    split(root, max_depth, min_size, n_features, 1)
    return root

def subsample(dataset, sample_size):
    # print('====== subsample called by random forest')
    sample = []
    n_sample = round(len(dataset) * sample_size)
    # print('         sample size = {} and n sample = {}'.format(sample_size, n_sample))
    while len(sample) < n_sample:
        index = random.randrange(len(dataset))
        sample.append(dataset[index])
    return sample

def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

def bagging_predict(trees, row):
    # print('###+++### Calling bagging_predict from random forest')
    # print('     trees = ')
    # pp.pprint(trees)
    predictions = [predict(tree, row) for tree in trees]

    # print('#####+++++##### predictions = {}'.format(predictions))
    return max(set(predictions), key=predictions.count)

def random_forest(train_set, test_set, max_depth, min_size, sample_size, n_trees, n_features):
    # print('============ random forest called by main fn')
    # print('max depth = {} min size = {} sample size = {} n trees = {} n features = {}'.format(max_depth, min_size, sample_size, n_trees, n_features))
    trees = []
    for i in range(n_trees):
        sample = subsample(train_set, sample_size)
        tree = build_tree(sample, max_depth, min_size, n_features)
        # print('#######======== result of tree = ')
        # pp.pprint(tree)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test_set]
    # print('#####+++++#####+++++ Before the final prediction test set = ')
    # print(test_set)
    # print('+++++#####+++++ predictions = {}'.format(predictions))
    return predictions


if __name__ == '__main__':
    pp = pprint.PrettyPrinter(indent=4)
    random.seed(2)

    # Load data
    # filename = 'sonar.all_data.csv'
    # dataset = load_csv(filename)
    # pp.pprint('dataset = {}'.format(dataset))
    features_train, labels_train, features_test, labels_test = make_terrain_data()
    for i, item in enumerate(features_train):
        item.append(labels_train[i])
    dataset = features_train

    t1 = time.time()
    # Prepare data, normalize, clean
    # for i in range(len(dataset[0])-1):
    #     str_column_to_float(dataset, i)
    # str_column_to_int(dataset, len(dataset[0])-1)

    n_folds = 5
    max_depth = 10
    min_size = 1
    sample_size = 1.0
    n_features = int(math.sqrt(len(dataset[0])-1))
    n_trees = 1
    scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
    t2 = time.time()
    print('Trees = {}'.format(n_trees))
    print('Scores = {}'.format(scores))
    print('Mean Accuracy {}'.format(sum(scores)/float(len(scores))))
    print('Time = {}'.format(t2-t1))
