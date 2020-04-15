import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
	"""
	Load data from the csv files for messages and catetories then merge then. 

	Input:
	--------------------
	messages_filepath : string, path to the messages file.
	categories_filepath : string, path to the categories file.

	Output:
	--------------------
	df : merged dataframe.
	"""

	# Load data from csv files
	messages = pd.read_csv(messages_filepath)
	categories = pd.read_csv(categories_filepath)

	# Merging the messages and categories datasets using the common id
	df = pd.merge(messages, categories, on='id')
	
	return df


def clean_data(df):
	"""
	Performs some cleaning operation on the dataframe like constructing the categories columns with their names, dropping duplicates, etc... 

	Input:
	--------------------
	df : merged dataframe, output from the load_data function.

	Output:
	--------------------
	df : cleaned dataframe. 
	"""

	# Split categories into separate category columns
	categories = df['categories'].str.split(';', expand=True)
	
	# Renaming the columns of the categories dataframe
	row = categories.iloc[0, :]
	category_colnames = row.apply(lambda x : x[:-2]).values
	categories.columns = category_colnames
	
	# Convert category values to just numbers 0 or 1:
	for column in categories:
		# set each value to be the last character of the string
		categories[column] = categories[column].apply(lambda x : x[-1])		
		# convert column from string to numeric
		categories[column] = categories[column].astype(int)
		
	# drop the original categories column from `df`
	df = df.drop('categories', axis=1)
	
	# concatenate the original dataframe with the new `categories` dataframe
	df = pd.concat([df, categories], axis=1)
	
	# Remove duplicates
	df = df.drop_duplicates()
	
	return df


def save_data(df, database_filename):
	"""
	Saves the clean dataframe into an SQL database.

	Input:
	--------------------
	df : cleaned dataframe, output from the clean_data function.

	Output:
	--------------------
	df : no output.
	"""

	# Save the clean dataset into an sqlite database
	engine = create_engine('sqlite:///' + database_filename)
	df.to_sql('MessagesTable', engine, index=False)
	
	pass  


def main():
	"""
	Main function of the script, call ll other functions successively.

	Arguments:
	--------------------
	messages_filepath, categories_filepath, database_filepath

	Output:
	--------------------
	df : no output.
	"""
	
	if len(sys.argv) == 4:
	
		messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
		
		print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
		df = load_data(messages_filepath, categories_filepath)
		
		print('Cleaning data...')
		df = clean_data(df)
		
		print('Saving data...\n    DATABASE: {}'.format(database_filepath))
		save_data(df, database_filepath)
		
		print('Cleaned data saved to database!')
		
	else:
	
		print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()