from time import time

from data.email_preprocess import preprocess
from sklearn import tree
from sklearn.metrics import accuracy_score

features_train, features_test, labels_train, labels_test = preprocess()

# With the default gini impurity, it took about 1 minute to train. Accuracy was very good, over 99%
# Using entropy, it's much faster, about 20 seconds to train, with an accuracy also over 99%


# clf = tree.DecisionTreeClassifier()
clf = tree.DecisionTreeClassifier(criterion='entropy')
training_start_time = time()
clf.fit(features_train, labels_train)
training_end_time = time()
pred = clf.predict(features_test)
test_end_time = time()

score = accuracy_score(labels_test, pred)
print('The prediction accuracy score = {}'.format(score))
print('The training time = {} and the prediction time = {}'.format(training_end_time - training_start_time, test_end_time - training_end_time))
