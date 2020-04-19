# Disaster Response Pipeline Project

## Description
This project consists of a web app that classifies in real time messages into multiples categories. The dataset was provided by Figure Eight and consists in messages collected during real-life disaster. In order to permit to use machine learning, these messages come with labels that represent categories they are related to like shelter, security, water, food, etc...
One message can be associated with multiple labels, and we have a total of 36 labels. 
After training, the user can enter a message in the app box and the predicted categories will appear. 

This project is decomposed into three different parts:  
* The ETL phase (Extract-Transform-Load) : in this part, we extract some data from multiple csv file, we transform and clean the collected data to construct the dataframe of interest, and we load the results into an SQL database. 
* The Modelling part : in this part, we construct features and feed a machine learning model to predict the categories. The caracteristics of this phase are the following:
	* We use nlp techniques to extract relevant information from the messages
	* We use Pipelining combined with grid search to tune the hyperparameters of the model and test the efficiency of new features
	* We use multi-output classifier since one message can have multiple labels
* The web app part : We used Flask to construct it

## Dependencies

For smooth running of the notebook, please install the following packages:  
* pandas >= 0.23.4  
* numpy >=  1.15.4  
* nltk >= 3.4  
* sklearn >= 0.20.1  
* pickle >= 4.0  
* SQLlite Database Libraries: SQLalchemy
* Web App and Data Visualization: Flask, Plotly


## Files Descriptions

The different directories are the following:
* The 'app' folder contains the web app related files. In particular, the 'run.py' is the file to lauch to run the web app.
* The 'data' folder contains the file related to the ETL part. It contains the initial csv files, and the saved database after ETL completion. The Python script 'process_data.py' contains the successive steps to run the ETL pipeline. 
* The 'models' folder contains the file used to produce the Machine Learning models. The .pkl file is the saved model and the 'train_classifier.py' file contains the ML pipeline. 

### Instructions to construc the data and launch the app:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Acknowledgements
I would like to thank Udacity for the project and Figure Eight for providing the data. 
