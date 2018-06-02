import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn import svm

mu_vec1 = np.array([0,0])
print('mu_vec1 = ')
print(mu_vec1)

cov_mat1 = np.array([[2,0], [0, 2]])
x1_samples = np.random.multivariate_normal(mu_vec1, cov_mat1, 100)
mu_vec1 = mu_vec1.reshape(1, 2).T
print('cov_mat1 = ')
print(cov_mat1)
print('x1_samples = ')
print(x1_samples)
print('mu_vec1 = ')
print(mu_vec1)

mu_vec2 = np.array([1,2])
print('mu_vec2 = ')
print(mu_vec2)

cov_mat2 = np.array([[1,0], [0, 1]])
x2_samples = np.random.multivariate_normal(mu_vec2, cov_mat2, 100)
mu_vec2 = mu_vec2.reshape(1, 2).T
print('cov_mat2 = ')
print(cov_mat2)
print('x2_samples = ')
print(x2_samples)
print('mu_vec2 = ')
print(mu_vec2)

fig = plt.figure()
plt.scatter(x1_samples[:,0], x1_samples[:,1], marker='+')
plt.scatter(x2_samples[:,0], x2_samples[:,1], c='green', marker='o')

training_data = np.concatenate((x1_samples, x2_samples), axis = 0)
training_label = np.array([0]*100 + [1]*100)

print('training_data =')
print(training_data)
print('training_label =')
print(training_label)

C = 1.0
clf = svm.SVC(kernel='linear')
clf.fit(training_data, training_label)

w = clf.coef_[0]
a = -w[0]/w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]
print('w = {} and a = {}'.format(w, a))
print('xx = {} and yy = {}'.format(xx, yy))

plt.plot(xx, yy, 'k-')
plt.show()
