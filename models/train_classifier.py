# import libraries
import sys

import pandas as pd
import pickle
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, f1_score 
from sklearn.model_selection import GridSearchCV

from sklearn.base import BaseEstimator, TransformerMixin

from scipy.stats import hmean

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
    """
	load data from database
	
	Input:
	database_filepath : path to database
	
	Output:
	X, Y, category_names : inputs to ML model
	"""
	
    database_filepath = "sqlite:///" + database_filepath
    engine = create_engine(database_filepath)
    df = pd.read_sql_table('MessagesTable', engine)
    X = df['message'].values
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1).values
    category_names = df.columns[3:]
	
    print('Loaded {} samples'.format(len(df)))
	
    return X, Y, category_names

def tokenize(text):
    """
	Tokenization function.
	
	Input:
	text : string to tokenize
	
	Output:
	tokens : list of tokens	
	"""
	
	# URL processing
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")  
        
    # Case Normalization and punctuation removal
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
	# Remove stopwords
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    stop_words = list(set(stop_words) - {'over', 'not'})
    
    # lemmatize and stem
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word, pos='n') for word in tokens]
    tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]
    tokens = [PorterStemmer().stem(w) for w in tokens]    
    
    return tokens
	
def generalized_fscore(y_true, y_pred):
    """
    Scoring function used for gridsearch.
    We use a generalization of the F score for precision recall for the '1' class of each category. 
    
    Input : 
    y_true, y_pred : arrays of true values and model predictions
    
    Output:
    f1score : model score	
    """
    score_list = []
    for column in range(y_true.shape[1]):
		# First compute each F score for the '1' class of each category
        score = f1_score(y_true[:,column], y_pred[:,column], average='binary', labels=[1])
		# Add 0.01 to avoid zero scores
        score = min(1, score+0.01)
        score_list.append(score)
    f1_score_vec = np.asarray(score_list)
	# Compute harmonic mean 
    f1score = hmean(f1_score_vec)
    return  f1score
	

class QuestionExtractor(BaseEstimator, TransformerMixin):
    """
	Transformer that returns 'True' if there is a question in the text. 
	This feature was proven useful through grid-search.
	"""
    def __init__(self, used=True):
		# 'used' is boolean to decide if the feature is used or not. 
        self.used = used
        
    def question_matcher(self, text):     
        if text.find('?') != -1 and self.used:
            return True
        else:
            return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.question_matcher)
        return pd.DataFrame(X_tagged)


class DollarsExtractor(BaseEstimator, TransformerMixin):
    """
	Transformer that returns 'True' if there is an amount in dollars in the text, like $25,000.
	This feature was proven useful through grid-search.
	"""
            
    def __init__(self, used=True):
		# 'used' is boolean to decide if the feature is used or not. 
        self.used = used
        
    def dollars_matcher(self, text):       
        money_regex = r"\$\ ?[+-]?[0-9]{1,3}(?:,?[0-9])*(?:\.[0-9]{1,2})?"
        z = re.findall(money_regex, text)
        if len(z)>0 and self.used:
            return True
        else:
            return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.dollars_matcher)
        return pd.DataFrame(X_tagged)

class NumSentExtractor(BaseEstimator, TransformerMixin):
    """
	Transformer that returns the number of sentences in the text.
	This feature was proven useful through grid-search.
	"""            
    def __init__(self, used=True):
		# 'used' is boolean to decide if the feature is used or not. 
        self.used = used
          
    def numsent_extractor(self, text):        
        try:
            sentences = sent_tokenize(text)
            if self.used:
                return len(sentences)
            else:
                return 0
        except:
            return 0

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.numsent_extractor)
        return pd.DataFrame(X_tagged)

def build_model(grid_search=1):
    """
    Model construction.

    Input : 
    grid_search : if 1, the gridsearch is ran to find the best hyper-parameters. Warning : this is a long process. 
    If 0, we take the model that was tuned. 
    
    Output:
    cv : optimal model with fined-tuned hyper-parameters
    """
    if grid_search:

        scorer = make_scorer(generalized_fscore, greater_is_better = True)
        clf = MultiOutputClassifier(RandomForestClassifier())
		
        pipeline = 	Pipeline([

	        ('features', FeatureUnion([

	            ('nlp_pipeline', Pipeline([
	                ('vect', CountVectorizer(tokenizer=tokenize, max_df=0.75, max_features=1000, min_df=3, ngram_range=(1, 3))),
	                ('tfidf', TfidfTransformer())
	            ])),

	            ('dollars_extractor', DollarsExtractor()), 
	            ('question_extractor', QuestionExtractor()), 
	            ('numsent_extractor', NumSentExtractor())

	    	]))

	    ])

        parameters_loc = {
			'vect__ngram_range': [(1, 3)],
			'vect__max_df': (0.75, 1.0),
			'vect__max_features': (5000, 10000),
			'vect__min_df': (1,5),        
			'clf__estimator__n_estimators': [50, 100, 200],
			'clf__estimator__min_samples_split': [2, 5]
		}
		
        cv = GridSearchCV(pipeline, param_grid=parameters_loc, scoring=scorer, n_jobs=-1, verbose=10)
	
    else:

        cv = Pipeline([

	        ('features', FeatureUnion([

	            ('nlp_pipeline', Pipeline([
	                ('vect', CountVectorizer(tokenizer=tokenize, max_df=0.75, max_features=1000, min_df=3, ngram_range=(1, 3))),
	                ('tfidf', TfidfTransformer())
	            ])),

	            ('dollars_extractor', DollarsExtractor()), 
	            ('question_extractor', QuestionExtractor()), 
	            ('numsent_extractor', NumSentExtractor())

	        ])),

	    	('clf', MultiOutputClassifier(RandomForestClassifier(min_samples_split=2, n_estimators=10)))

	    ])

    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """
	Model evaluation.

	Input : 
	model : model fit on data
	X_test : features of the test set. 
	y_test : ground-truth on test set.  
	category_names : names of categories
	
	Output:
	No output. The function computes and shows the model performance. 
	"""

    y_pred = model.predict(X_test)

    acc = (y_pred == y_test).mean(axis=0)
    for ii, (name, acc_loc) in enumerate(zip(category_names, acc)):
        print('='*50)
        print('- '+name.upper()+':' )
        print('-'*50)
        print('Accuracy : {:.2f}'.format(acc_loc))
        print('-'*50)
        print(classification_report(y_test[:, ii], y_pred[:, ii]))

    pass


def save_model(model, model_filepath):
    """
    Model saving.
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    
    pass


def main():
    """
	Main function.
	"""
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