import os

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from data_preprocessing import tokenize

# Read data
train = pd.read_csv("../input/train.tsv", sep="\t")
test = pd.read_csv("../input/test.tsv", sep="\t")

# Remove missing values in train
X_train = train[train['tags'].notnull()]

train_x = X_train['description'].as_matrix()  # train-description
test_x = test['description'].as_matrix()  # test-description
train_y = X_train['tags'].str.split().as_matrix()  # train-tags

# Transform train-tags into a multi-label binary format
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform(train_y)

# Applying CountVectorizer on character n-grams (specifically for tri-grams range)
count_vectorizer = CountVectorizer(stop_words="english", tokenizer=tokenize, ngram_range=(1, 3),
                                   max_features=10000, analyzer="char")

# Learn and transform train-description
count_vectorizer_train_x = count_vectorizer.fit_transform(train_x)
count_vectorizer_test_x = count_vectorizer.transform(test_x)

# Applying TfIdfVectorizer on individual words(specifically for tri-grams range)
tfidf_vectorizer = TfidfVectorizer(stop_words="english", tokenizer=tokenize, ngram_range=(1, 3),
                                   max_features=30000, analyzer="word")

# Learn and transform train-description
tfidf_vectorizer_train_x = tfidf_vectorizer.fit_transform(train_x)
tfidf_vectorizer_test_x = tfidf_vectorizer.transform(test_x)

# GradientBoostingClassifier with parameter tuning
params = {"n_estimators": 170, "max_depth": 5, "random_state": 10, "min_samples_split": 4, "min_samples_leaf": 2}
classifier = OneVsRestClassifier(GradientBoostingClassifier(**params))

# Generate predictions using counts
classifier.fit(count_vectorizer_train_x, train_labels)
file_cnt = "loaded_model/count_vectorizer_model.pkl"  # serialize model with pickle
os.makedirs(os.path.dirname(file_cnt), exist_ok=True)
with open(file_cnt, "w") as f:
    joblib.dump(classifier, file_cnt)
print("CountVectorizer based trained classifier ready to be exported")

# Calling fit() more than once will overwrite what was learned by any previous fit()
# Generate predictions using tf-idf representation
classifier.fit(tfidf_vectorizer_train_x, train_labels)
file_tfidf = "loaded_model/tfidf_vectorizer_model.pkl"  # serialize model with pickle
os.makedirs(os.path.dirname(file_tfidf), exist_ok=True)
with open(file_tfidf, "w") as f:
    joblib.dump(classifier, file_tfidf)
print("TfidfVectorizer based trained classifier ready to be exported")
