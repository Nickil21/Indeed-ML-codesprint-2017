import numpy as np
import pandas as pd
from sklearn.externals import joblib

from data_preprocessing import join_strings
from model import mlb, count_vectorizer_test_x, tfidf_vectorizer_test_x, file_cnt, file_tfidf

count_vectorizer_model, tfidf_vectorizer_model = joblib.load(file_cnt), joblib.load(file_tfidf)
print("Both the trained models have been imported successfully!")
print()
print("Making predictions...")
pred1 = count_vectorizer_model.predict(count_vectorizer_test_x.toarray())
pred2 = tfidf_vectorizer_model.predict(tfidf_vectorizer_test_x.toarray())

# Combine predictions and map the labels if the values do not equal 0, else assign empty string
arr = np.where((pred1 + pred2) != 0, mlb.classes_, "")
# Load the array into a DataFrame constructor and join non-empty strings
predictions = pd.DataFrame(arr).apply(join_strings, axis=1).to_frame("tags")
# Submit predictions
print("Submitting predictions...")
predictions.to_csv("tags.tsv", index=False)
print("done")
