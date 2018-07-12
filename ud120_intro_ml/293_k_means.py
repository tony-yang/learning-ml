import matplotlib.pyplot as plt
import numpy as np
import pickle

from sklearn.cluster import KMeans

def feature_format(data_dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys=False):
    return_list = []

    if sort_keys:
        keys = sorted(data_dictionary.keys())
    else:
        keys = data_dictionary.keys()
    
    for key in keys:
        tmp_list = []
        for feature in features:
            try:
                value = data_dictionary[key][feature]
                if value == 'NaN' and remove_NaN:
                    value = 0
                tmp_list.append(float(value))
            except KeyError:
                print('error: key {} not presented'.format(feature))
                return
        
        append = True
        if features[0] == 'poi':
            test_list = tmp_list[1:]
        else:
            test_list = tmp_list
        
        if remove_all_zeroes:
            append = False
            for item in test_list:
                if item != 0 and item != 'NaN':
                    append = True
                    break
        
        if remove_any_zeroes:
            if 0 in test_list or 'NaN' in test_list:
                append = False
        
        if append:
            return_list.append(np.array(tmp_list))
    return return_list

def target_feature_split(data):
    target = []
    features = []
    for item in data:
        target.append(item[0])
        features.append(item[1:])
    return target, features

def draw(pred, features, poi, mark_poi=False, f1_name='feature1', f2_name='feature2'):
    colors = ['b', 'c', 'k', 'm', 'g', 'r', 'y', 'w']
    for i, prediction in enumerate(pred):
        plt.scatter(features[i][0], features[i][1], color=colors[pred[i]])
    
    if mark_poi:
        for i, prediction in enumerate(pred):
            if poi[i]:
                plt.scatter(features[i][0], featuers[i][1], color='r', marker='*')
    
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.show()

data_dictionary = pickle.load(open('data/final_project_dataset.pkl', 'rb'))
data_dictionary.pop('TOTAL', 0)

feature1 = 'salary'
feature2 = 'exercised_stock_options'
poi = 'poi'
features_list = [poi, feature1, feature2]
data = feature_format(data_dictionary, features_list)
poi, finance_features = target_feature_split(data)

print('poi =')
print(poi)
print('finance features =')
print(finance_features)

for f1, f2 in finance_features:
    plt.scatter(f1, f2)

plt.show()

kmeans = KMeans(random_state=42).fit(finance_features)
pred = kmeans.predict(finance_features)
print('prediction = ')
print(pred)

try:
    draw(pred, finance_features, poi, mark_poi=False, f1_name=feature1, f2_name=feature2)
except NameError:
    print('no predictions object named pred found, no clusters to plot')