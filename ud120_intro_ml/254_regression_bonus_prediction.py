import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model
from sklearn.cross_validation import train_test_split

dictionary = pickle.load(open('data/final_project_dataset_modified.pkl', 'rb'))

def feature_format(dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys=False):
    return_list = []

    if isinstance(sort_keys, str):
        import pickle
        keys = pickle.load(open(sort_keys, 'rb'))
    elif sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = dictionary.keys()
    
    for key in keys:
        tmp_list = []
        for feature in features:
            try:
                value = dictionary[key][feature]
                if value == 'NaN' and remove_NaN:
                    value = 0
                tmp_list.append(float(value))
            except KeyError:
                print('error: key {} not present'.format(feature))
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
        
    return np.array(return_list)

def target_feature_split(data):
    target = []
    features = []
    for item in data:
        target.append(item[0])
        features.append(item[1:])
    return target, features

features_list = ['bonus', 'salary']
data = feature_format(dictionary, features_list, remove_any_zeroes=True)
target, features = target_feature_split(data)

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = 'b'
test_color = 'r'

reg = linear_model.LinearRegression()
reg.fit(feature_train, target_train)

for feature, target in zip(feature_test, target_test):
    plt.scatter(feature, target, color=test_color)
for feature, target in zip(feature_train, target_train):
    plt.scatter(feature, target, color=train_color)

plt.scatter(feature_test[0], target_test[0], color=test_color, label='test')
plt.scatter(feature_test[0], target_test[0], color=train_color, label='train')

try:
    plt.plot(feature_test, reg.predict(feature_test))
except NameError:
    pass

print('slope = {}'.format(reg.coef_))
print('intercept = {}'.format(reg.intercept_))
print('######### stats on test dataset')
print('r-squred score = {}'.format(reg.score(feature_test, target_test)))

print('########## stats on training dataset')
print('r-squred score = {}'.format(reg.score(feature_train, target_train)))

plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()