import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score,accuracy_score
import pickle

nltk.download("stopwords")

### Loading the dataset
dataset = pd.read_csv('reviews.txt',sep = '\t', names =['Reviews','Comments'])

# Data Preprocessing
stopset = set(stopwords.words('english'))

vectorizer = TfidfVectorizer(use_idf = True,lowercase = True, strip_accents='ascii',stop_words=stopset)

# Splitting the data into train and test set
X = vectorizer.fit_transform(dataset.Comments)
y = dataset.Reviews
pickle.dump(vectorizer, open('tranform.pkl', 'wb'))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Naive Bayes Model
clf = naive_bayes.MultinomialNB()
clf.fit(X_train,y_train)

clf = naive_bayes.MultinomialNB()
clf.fit(X,y)

# Creating a pickle file for the classifier
filename = 'nlp_model.pkl'
pickle.dump(clf, open(filename, 'wb'))