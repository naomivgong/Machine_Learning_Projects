#import dependencies
#import dependecies
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


import nltk
nltk.download('stopwords')

news_data = pd.read_csv("/Users/naomigong/Coding/ML Projects/FakeNewsDetection/train (1).csv")
#0 is reliable
#1 is unreliable
news_data.isnull().sum() #shows there are null values
news_data = news_data.fillna("")
print(news_data.isnull().sum())

news_data["content"] = news_data["author"] + news_data["title"]
X = news_data.drop(columns = 'label', axis = 1)
Y = news_data['label']

port_stem = PorterStemmer()
stop_words = set(stopwords.words('english'))

def stemming(content):
    if isinstance(content, str):
        # Remove non-alphabetic characters, lowercase, and split into words
        stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
        stemmed_content = stemmed_content.lower().split()

        # Apply stemming and remove stopwords
        stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stop_words]
        
        # Join stemmed words back into a string
        stemmed_content = " ".join(stemmed_content)
        
        return stemmed_content
    else:
        return ''
news_data['content'] = news_data['content'].apply(stemming)
X = news_data['content'].values
Y = news_data['label'].values

#converting the textual data to a feature vector
# converting the textual data to feature vectors
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state= 2)

#You want to use a logistic model because you have 0 or 1 outcomes
model = LogisticRegression()
model.fit(X_train, Y_train) #fit trains our data
#train on training data first
y_preds = model.predict(X_train) 
training_data_accuracy = accuracy_score(y_preds, Y_train) #Y_train is the original labels
print("for the training data", training_data_accuracy)

#then test it on the test data
#train on training data first
y_preds = model.predict(X_test) 
training_data_accuracy = accuracy_score(y_preds, Y_test) #Y_train is the original labels
print("for the testing data", training_data_accuracy)

#Building a predicitive system
x_new = X_test[0] #some set of values

pred = model.predict(x_new)
print(pred)
if (pred[0] == 0):
    print("The source is reliable")
else:
    print("It is unreliable")

