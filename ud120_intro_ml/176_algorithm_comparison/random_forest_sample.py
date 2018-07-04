#!/usr/bin/python3

import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz

features = pd.read_csv('./temps.csv')
print(features.head(5))

print('The shape of our features is: {}'.format(features.shape))
print(features.describe())

features = pd.get_dummies(features)
print(features.iloc[:, 5:].head(5))

labels = np.array(features['actual'])
print('labels = before np array = {}'.format(features['actual']))
print('labels = {}'.format(labels))
features = features.drop('actual', axis = 1)
print('features after drop = {}'.format(features.head(5)))
feature_list = list(features.columns)
print('feature list = {}'.format(feature_list))
features = np.array(features)
print('features after np array = {}'.format(features))

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

print('Training features shape: {}'.format(train_features.shape))
print('Training labels shape: {}'.format(train_labels.shape))
print('Testing features shape: {}'.format(test_features.shape))
print('Tesitng labels shape: {}'.format(test_labels.shape))

baseline_preds = test_features[:, feature_list.index('average')]
print('#################')
print('baseline preds = {}'.format(baseline_preds))
print('test labels = {}'.format(test_labels))
baseline_errors = abs(baseline_preds - test_labels)
print('the baseline errors = {}'.format(baseline_errors))
print('Average baseline error: {}'.format(round(np.mean(baseline_errors), 2)))
rms = (baseline_preds - test_labels) ** 2
print('intermediate rms = {}'.format(rms))
rms = np.mean(rms) ** 0.5
print('intermediate rms 2 = {}'.format(rms))
print('rms of the baseline error = {} degrees'.format(round(rms, 2)))

random_forest = RandomForestRegressor(n_estimators = 1000, random_state = 42)
random_forest.fit(train_features, train_labels)

predictions = random_forest.predict(test_features)
errors = abs(predictions - test_labels)
print('Mean absolute error: {} degrees'.format(round(np.mean(errors), 2)))

rms = (predictions - test_labels) ** 2
rms = np.mean(rms) ** 0.5
print('rms of the error = {} degrees'.format(round(rms, 2)))

mape = 100 * (errors / test_labels)
accuracy = 100 - np.mean(mape)
print('Accuracy: {}%'.format(round(accuracy, 2)))

tree = random_forest.estimators_[5]

random_forest_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
random_forest_small.fit(train_features, train_labels)

tree_small = random_forest_small.estimators_[5]

importances = list(random_forest.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse = True)
[print('Var: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

plt.style.use('fivethirtyeight')
x_values = list(range(len(importances)))
plt.bar(x_values, importances, orientation='vertical')
plt.xticks(x_values, feature_list, rotation='vertical')
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.title('Variable Importances')
#plt.show()

months = features[:, feature_list.index('month')]
days = features[:, feature_list.index('day')]
years = features[:, feature_list.index('year')]

dates = ['{}-{}-{}'.format(str(int(year)), str(int(month)), str(int(day))) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

true_data = pd.DataFrame(data = {'date': dates, 'actual': labels})
true_data['temp_1'] = features[:, feature_list.index('temp_1')]
true_data['average'] = features[:, feature_list.index('average')]
true_data['friend'] = features[:, feature_list.index('friend')]

months = test_features[:, feature_list.index('month')]
days = test_features[:, feature_list.index('day')]
years = test_features[:, feature_list.index('year')]

test_dates = ['{}-{}-{}'.format(str(int(year)), str(int(month)), str(int(day))) for year, month, day in zip(years, months, days)]
test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]

predictions_data = pd.DataFrame(data={'date': test_dates, 'prediction': predictions})

plt.plot(true_data['date'], true_data['actual'], 'b-', label='actual', alpha=1.0)
plt.plot(true_data['date'], true_data['temp_1'], 'y-', label='temp_1', alpha=1.0)
plt.plot(true_data['date'], true_data['average'], 'k-', label='average', alpha=0.8)
plt.plot(true_data['date'], true_data['friend'], 'r-', label='friend', alpha=0.3)
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label='prediction')
plt.xticks(rotation = '60')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Maximum Temperature (F)')
plt.title('Actual and Predicted Values')
#plt.show()