import numpy
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif

def preprocess(words_file='./data/word_data.pkl', authors_file='./data/email_authors.pkl'):
    # The words (features) and authors (labels) are largely preprocessed
    with open(authors_file, 'rb') as authors_file_handler:
        authors = pickle.load(authors_file_handler)

    with open(words_file, 'rb') as words_file_handler:
        word_data = pickle.load(words_file_handler)

    features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed = vectorizer.transform(features_test)

    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(features_train_transformed, labels_train)
    features_train_transformed = selector.transform(features_train_transformed).toarray()
    features_test_transformed = selector.transform(features_test_transformed).toarray()

    print('Number of Chris training emails: {}'.format(sum(labels_train)))
    print('Number of Sara training emails: {}'.format(len(labels_train)-sum(labels_train)))

    return features_train_transformed, features_test_transformed, labels_train, labels_test
