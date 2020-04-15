# import libraries
import sys

import pandas as pd
import pickle
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report 
from sklearn.model_selection import GridSearchCV

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk import pos_tag

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')


import re
import numpy as np

def load_data(database_filepath):
	# load data from database
	database_filepath = "sqlite:///" + database_filepath
	engine = create_engine(database_filepath)
	df = pd.read_sql_table('MessagesTable', engine)
	X = df['message'].values
	Y = df.drop(['id', 'message', 'original', 'genre'], axis=1).values
	category_names = df.columns[3:]
	
	return X, Y, category_names

def tokenize(text):
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # numbers_regex = r"[^0-9]"

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")  
        
    # Case Normalization and punctuation removal
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    stop_words = list(set(stop_words) - {'over', 'not'})
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]
    tokens = [PorterStemmer().stem(w) for w in tokens]    
    
    return tokens


def build_model(grid_search=1):
    
	if grid_search:
	
		clf = MultiOutputClassifier(RandomForestClassifier())
		
		pipeline = Pipeline([
			('vect', CountVectorizer(tokenizer=tokenize)),
			('tfidf', TfidfTransformer()),
			('clf', clf)
		])

		parameters_loc = {
			'vect__ngram_range': [(1, 3)],
			'vect__max_df': (0.75, 1.0),
			'vect__max_features': (5000, 10000),
			'vect__min_df': (1,5),        
			'clf__estimator__n_estimators': [50, 200],
			'clf__estimator__min_samples_split': [2, 5]
		}
		
		cv = GridSearchCV(pipeline, param_grid=parameters_loc)
	
	else:
	
		clf = MultiOutputClassifier(RandomForestClassifier(min_samples_split=2, n_estimators=500))

		cv = Pipeline([
			('vect', CountVectorizer(tokenizer=tokenize, max_df=0.75, max_features=1000, min_df=3, ngram_range=(1, 3))),
			('tfidf', TfidfTransformer()),
			('clf', clf)
		])
		
	return cv


def evaluate_model(model, X_test, y_test, category_names):

	y_pred_cv = model.predict(X_test)

	acc = (y_pred_cv == y_test).mean(axis=0)
	for ii, (name, acc_loc) in enumerate(zip(category_names, acc)):
		print('='*50)
		print('- '+name.upper()+':' )
		print('-'*50)
		print('Accuracy : {:.2f}'.format(acc_loc))
		print('-'*50)
		print(classification_report(y_test[:, ii], y_pred_cv[:, ii]))
		
	pass


def save_model(model, model_filepath):
	pickle.dump(model, open(model_filepath, 'wb'))
	
	pass


def main():
    if len(sys.argv) == 3:
	
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(grid_search=0)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()