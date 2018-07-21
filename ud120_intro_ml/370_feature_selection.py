import numpy as np
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

words_file = 'data/your_word_data.pkl'
authors_file = 'data/your_email_authors.pkl'

word_data = pickle.load(open(words_file, 'rb'))
authors = pickle.load(open(authors_file, 'rb'))

features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
features_train_transformed = vectorizer.fit_transform(features_train).toarray()
# print('type of features train = {}'.format(type(features_train_transformed)))
features_test_transformed = vectorizer.transform(features_test).toarray()

# print('######### label train')
# print(labels_train)
# print('type of labels train = {}'.format(type(labels_train)))

# print('######### features train')
# print(features_train_transformed)
# print('len of features train = {} and shape = {}'.format(features_train_transformed.size, features_train_transformed.shape))
# print('######### features_test')
# print(features_test_transformed)
# print('features test shape = {}'.format(features_test_transformed.shape))

selector = SelectPercentile(f_classif)
selector.fit(features_train_transformed, labels_train)
features_train_transformed = selector.transform(features_train_transformed)
# print("###############the new feature after selector transform")
# print(features_train_transformed)
# print('len of features train = {} and shape = {}'.format(features_train_transformed.size, features_train_transformed.shape))
features_test_transformed = selector.transform(features_test_transformed)

print('# of chris email = {}'.format(sum(labels_train)))
print('# of sara email = {}'.format(len(labels_train) - sum(labels_train)))

clf = DecisionTreeClassifier(random_state=42)
clf = clf.fit(features_train_transformed, labels_train)
pred = clf.predict(features_test_transformed)

accuracy = accuracy_score(pred, labels_test)
score = clf.score(features_train_transformed, labels_train)

print('Accuracy of prediction = {}'.format(accuracy))
print('Score = {}'.format(score))

max_features = clf.max_features_
print('Max features = {}'.format(max_features))
n_features = clf.n_features_
print('n  features = {}'.format(n_features))

feature_names = vectorizer.get_feature_names()
feature_importance = clf.feature_importances_
for i, importance in enumerate(feature_importance):
    if importance > 0.1:
        print('index i = {} and importance = {}'.format(i, importance))
        feature_name = feature_names[i]
        print('The featured word = {}'.format(feature_name))

signature_index = np.argmax(feature_importance)
signature_word = feature_names[signature_index]
print('#############')
print('signature index = {} and signature word = {}'.format(signature_index, signature_word))


# Now try to overfit the data
print('###############################')
print('Try to overfit the data')
features_train_reduced = features_train_transformed[:150]
labels_train_reduced   = labels_train[:150]

clf2 = DecisionTreeClassifier(random_state=42)
clf2 = clf2.fit(features_train_reduced, labels_train_reduced)
pred = clf2.predict(features_test_transformed)

accuracy = accuracy_score(pred, labels_test)
score = clf2.score(features_train_transformed, labels_train)

print('Accuracy of prediction = {}'.format(accuracy))
print('Score = {}'.format(score))

feature_importance = clf2.feature_importances_
for i, importance in enumerate(feature_importance):
    if importance > 0.1:
        print('index i = {} and importance = {}'.format(i, importance))
        feature_name = feature_names[i]
        print('The featured word = {}'.format(feature_name))

signature_index = np.argmax(feature_importance)
signature_word = feature_names[signature_index]
print('#############')
print('signature index = {} and signature word = {}'.format(signature_index, signature_word))