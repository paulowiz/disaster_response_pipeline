import sys
import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sqlalchemy import create_engine

nltk.download(['punkt', 'wordnet'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
import joblib

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(database_filepath):
    """
    Load database into dataframe and return features,target and columns

    input:
    database_filepath  sqlite database path .db

    return X features,
          y targets,
          columns column names  from dataframe
    """
    engine = create_engine('sqlite:///' + database_filepath)
    print(engine)
    df = pd.read_sql_table('disaster_message', engine)
    X = df['message']
    y = df[df.columns[7::]]
    columns = df.columns[7::]
    return X, y, columns


def tokenize(text):
    """
    Tokenize a text into tokens and apply lemmatizer method on them.

    input
     text  disaster message

    return clean_tokens  array of words from the disaster message
    """
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


def build_model():
    """
    Build the machine learning model instance

    return pipeline    Machine learning model instance
    """
    from sklearn.linear_model import LogisticRegression

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.6, 1.0),
        'tfidf__use_idf': (True, False)
    }

    # cv = GridSearchCV(pipeline, param_grid=parameters, cv=3)

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model results and print them
    input:
       model   Machine learning model
       X_test  Features test
       Y_test  Target test
       categories_names   disaster category names

    return None
    """

    # predicted values
    Y_pred = model.predict(X_test)
    df_predict = pd.DataFrame()
    for col, idx in zip(category_names, range(0, len(category_names))):
        arr_tmp = []
        for item in Y_pred:
            arr_tmp.append(item[idx])
        df_predict[col] = arr_tmp

    for col in category_names:
        print("category: ", col)
        print(classification_report(Y_test[col], df_predict[col]))
    pass


def save_model(model, model_filepath):
    """
    Export model to .pkl
    input:
       model   Machine learning model
       model_filepath   reference path where the model will be saved
    return None
    """
    joblib.dump(model, model_filepath)
    pass


def main():
    """
    Main function where call all functions in the right sequence.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        # model.fit(X_train, Y_train)
        import joblib

        model = joblib.load('models/classifier.pkl')

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
