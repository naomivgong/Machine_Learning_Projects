import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#for turning text data into numerical feature
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#Data Collection and Preprocessing
raw_mail_data = pd.read_csv('/Users/naomigong/Coding/ML Projects/spamMailDetector/mail_data.csv')
raw_mail_data.info()
raw_mail_data.head()

#replace the null values with a null string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '') #replace all null values with null strings
mail_data.loc[mail_data['Category'] == 'spam', 'Category',]  = 0 #for all in which the categroy column is == spam replace with 0
#second category specifies this operation should only affect that Column
mail_data.loc[mail_data['Category'] == 'ham', 'Category',]  = 1

#seperating into text and labels
X = mail_data['Message']
Y = mail_data['Category']


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 2)

#feature extraction -- turn text data into feature vectors
#TfidVectorizer goes through dataset and tries to look for repeated words
#more repeated words = higher score
#if words repeated only once min_df = 1 dont include
#Ignore stop words (ex. me, i, a , it)
feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase = True)

#fits so it finds the common features (the patterns)
#transforms actually applies that patterns and gives us feature vectors
X_train_features = feature_extraction.fit_transform(X_train)
#no need to fit with testing data
X_test_features = feature_extraction.transform(X_test)

#convert Y_train and Y_test into integers
#as of now Y_train is of objects make them as ints
Y_train = Y_train.astype(int)
Y_test = Y_test.astype(int)

model = LogisticRegression()
model.fit(X_train_features, Y_train)
y_pred = model.predict(X_test_features)


score = accuracy_score(Y_test, y_pred)
print(score)


