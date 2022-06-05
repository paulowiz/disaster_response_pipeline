from sqlalchemy import create_engine
# import libraries
import pandas as pd


engine = create_engine('sqlite:///desaster_project.db')
df = pd.read_sql_table('disaster_message', engine)
df.head()

import nltk
# nltk.download()
nltk.download(['punkt', 'wordnet'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def tokenize(text):
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])


X = df['message']
y = df[df.columns[7::]]
X_train, X_test, y_train, y_test = train_test_split(X, y)
pipeline.fit(X_train, y_train)


def display_results(y_test, y_pred):
    labels = np.unique(y_pred)
    accuracy = (y_pred == y_test).mean()

    print("Labels:", labels)
    print("Accuracy:", accuracy)

y_pred = pipeline.predict(X_test)
display_results(y_test, y_pred)


from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
parameters = {
}

cv2 = GridSearchCV(pipeline, param_grid=parameters)


y_new_pred = cv2.fit(X_train, y_train)
display_results(y_test, y_new_pred)


import joblib
joblib.dump(cv2, 'model.pkl')