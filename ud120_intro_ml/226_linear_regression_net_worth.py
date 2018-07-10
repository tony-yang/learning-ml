import random
import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model
from sklearn.model_selection import train_test_split

def age_net_worth_data(num_of_points=100):
    random.seed(42)
    noises = [random.uniform(-100, 100) for i in range(num_of_points)]
    ages = np.array([random.randint(0, 100) for i in range(num_of_points)])
    ages = ages.reshape(num_of_points, 1)
    net_worths = np.array([6.25 * age + noises[i] for i, age in enumerate(ages)])
    net_worths = net_worths.reshape(num_of_points, 1)

    ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths, test_size=0.25, random_state=42)
    return ages_train, ages_test, net_worths_train, net_worths_test

def student_reg(ages_train, net_worths_train):
    reg = linear_model.LinearRegression()
    reg.fit(ages_train, net_worths_train)
    return reg

ages_train, ages_test, net_worths_train, net_worths_test = age_net_worth_data()

reg = student_reg(ages_train, net_worths_train)

print('My net worth prediction = {}'.format(reg.predict([[20]])))
print('slope = {}'.format(reg.coef_))
print('intercept = {}'.format(reg.intercept_))

print('############### stats on test dataset')
print('r-squred score = {}'.format(reg.score(ages_test, net_worths_test)))

print('############### stats on training dataset')
print('r-squred score = {}'.format(reg.score(ages_train, net_worths_train)))

plt.clf()
plt.scatter(ages_train, net_worths_train, color='b', label='train_data')
plt.scatter(ages_test, net_worths_test, color='r', label='test_data')
plt.plot(ages_test, reg.predict(ages_test), color='black')
plt.legend(loc=2)
plt.xlabel('ages')
plt.ylabel('net worths')
plt.show()