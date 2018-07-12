import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

from sklearn import linear_model
from sklearn.model_selection import train_test_split

def outlier_cleaner(predictions, ages_train, net_worths_train):
    # This approach, even though we are doing sort, the overall performance is O(nlogn), dominating by the sort
    cleaned_data = []
    errors = []

    for i, prediction in enumerate(predictions):
        errors.append((net_worths_train[i] - prediction) ** 2)
    errors = np.sort(errors, axis=None)[::-1]
    smallest_outlier = errors[int(len(errors) * 0.1)]
    
    for i, prediction in enumerate(predictions):
        error = (net_worths_train[i] - prediction) ** 2
        if error < smallest_outlier:
            cleaned_data.append((ages_train[i], net_worths_train[i], error))

    return cleaned_data

def determine_outlier(outliers, cleaned_data, error):
    for i in range(len(outliers)):
        if error[2] > outliers[i][2]:
            existing_outlier = outliers[i]
            outliers[i] = error
            error = existing_outlier
    cleaned_data.append(error)
    return outliers, cleaned_data

def outlier_cleaner2(predictions, ages_train, net_worths_train):
    # This approach actually performs worth since outlier length is 0.1n
    # The actual run time is 0.1n*n which is O(n2)
    cleaned_data = []
    outliers_length = int(len(predictions) * 0.1)
    outliers = [(ages_train[i], net_worths_train[i], (net_worths_train[i] - predictions[i]) ** 2) for i in range(outliers_length)]

    for i, prediction in enumerate(predictions):
        if i < outliers_length:
            continue
        error = (ages_train[i], net_worths_train[i], (net_worths_train[i] - prediction) ** 2)
        outliers, cleaned_data = determine_outlier(outliers, cleaned_data, error)
    return cleaned_data
    



ages = np.array(pickle.load(open('data/practice_outliers_ages.pkl', 'rb')))
net_worths = np.array(pickle.load(open('data/practice_outliers_net_worths.pkl', 'rb')))

ages = ages.reshape(len(ages), 1)
net_worths = net_worths.reshape(len(net_worths), 1)

ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths, test_size=0.1, random_state=42)

reg = linear_model.LinearRegression()
reg.fit(ages_train, net_worths_train)

try:
    plt.plot(ages, reg.predict(ages), color='b')
except NameError:
    pass

plt.scatter(ages, net_worths)
plt.show()

cleaned_data = []
try:
    predictions = reg.predict(ages_train)
    t1 = time.time()
    cleaned_data = outlier_cleaner(predictions, ages_train, net_worths_train)
    t2 = time.time()
    cleaned_data2 = outlier_cleaner2(predictions, ages_train, net_worths_train)
    t3 = time.time()

    print('Time spent using outlier_cleaner = {}'.format(t2 - t1))
    print('Time spent using outlier_cleaner2 = {}'.format(t3 - t2))
except NameError:
    print('Your regression object does not exist, or is not name reg. Cannot make pred')

if len(cleaned_data) > 0:
    ages, net_worths, errors = zip(*cleaned_data)
    ages = np.reshape(np.array(ages), (len(ages), 1))
    net_worths = np.reshape(np.array(net_worths), (len(net_worths), 1))

    try:
        reg.fit(ages, net_worths)
        plt.plot(ages, reg.predict(ages), color='b')
    except NameError:
        print('Your do not seem to have regression imported')

    plt.scatter(ages, net_worths)
    plt.xlabel('ages')
    plt.ylabel('net worths')
    plt.show()
else:
    print('outlier_cleaner is returning an empty list, no refitting to be done')