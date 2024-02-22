import pickle

import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pandas as pd
import string 
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk

train_data = pd.read_csv("train_data.txt", sep=':::', names=['Title', 'Genre', 'Description'], engine='python')
test_data = pd.read_csv("test_data.txt", sep=':::', names=['Title', 'Genre', 'Description'], engine='python')

stemmer = LancasterStemmer()
stop_words = set(stopwords.words('english'))
def clean(dataText) :
    dataText = dataText.lower()
    dataText = re.sub(r'@\S+','',dataText)
    dataText = re.sub(r'http\S+','',dataText) # POUR SUPPRIMER LES LIENS 
    dataText = re.sub(r'pic.\S+','',dataText) # 
    dataText = re.sub(r"[^a-zA-Z+']", ' ', dataText)  # Keep only characters
    dataText = re.sub(r'\s+[a-zA-Z]\s+', ' ', dataText + ' ')  # Keep words with length > 1 only
    dataText = "".join([i for i in dataText if i not in string.punctuation])# supprimer la ponctuation 
    words = nltk.word_tokenize(dataText)
    stopwords = nltk.corpus.stopwords.words('english')  # Remove stopwords > les mot qui ont pas de sens comme MY ME THEIRE
    dataText = " ".join([i for i in words if i not in stopwords and len(i) > 2])
    dataText = re.sub("\s[\s]+", " ", dataText).strip()  # Remove repeated/leading/trailing spaces
    return dataText


train_data['clean_text'] = train_data['Description'].apply(clean)
test_data['clean_text'] = test_data['Description'].apply(clean)
print(train_data['clean_text'].head())


tfidf_vectorizer = TfidfVectorizer()
X_train = tfidf_vectorizer.fit_transform(train_data['clean_text'])
X_test = tfidf_vectorizer.transform(test_data['clean_text'])

X = X_train
y = train_data['Genre'] # the target variable 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# cet objet permet de faire la classification 
Logistic_classifier = LogisticRegression(max_iter=1000)
Logistic_classifier.fit(X_train, y_train)
# Make predictions on the validation set
y_pred = Logistic_classifier.predict(X_val)
y_val = np.array(y_val)
# creating the pickle of our model
pickle.dump(Logistic_classifier, open("model.pkl", "wb"))
import os

# Check if the model.pkl file exists
if os.path.exists("model.pkl"):
    print("The model.pkl file has been successfully created.")
else:
    print("Error: The model.pkl file was not created.")

