from time import time

from data.email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

features_train, features_test, labels_train, labels_test = preprocess()

clf = GaussianNB()
training_start_time = time()
clf.fit(features_train, labels_train)
training_end_time = time()
pred = clf.predict(features_test)
test_end_time = time()

score = accuracy_score(labels_test, pred)
print('The prediction accuracy score = {}'.format(score))
print('The training time = {} and the prediction time = {}'.format(training_end_time - training_start_time, test_end_time - training_end_time))
