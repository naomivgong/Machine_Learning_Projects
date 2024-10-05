import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

pd.set_option('display.max_columns', None)

loan_df = pd.read_csv("/Users/naomigong/Coding/ML Projects/loanStatusPrediction/loandata.csv")

print(loan_df.shape)

#unlike what i originally did, in this case you should remove the missing values.
#This is ok when missing values are small and randomly distributed across the dataset
#If you have categorical data, you should drop the missing values

loan_df = loan_df.dropna() #all rows with missing columns are dropped
print(loan_df.shape) #you can see the number of rows changed

#label encoding (Y/N)-- #you can also use sklearn.preprocessing labelencoder
loan_df.replace({'Loan_Status': {'N':0, 'Y':1}}, inplace = True)
#taking loan status column, if it is N replace with 0 and Y replace with 1

#label encoding
label_encoder= LabelEncoder()

#for gender
label_encoder.fit(loan_df['Gender']) #finds pattern
gender_labels = label_encoder.transform(loan_df['Gender']) #actually applies it
loan_df['Gender'] = gender_labels
#for marriage
label_encoder.fit(loan_df['Married']) #finds pattern
married_labels = label_encoder.transform(loan_df['Married']) #actually applies it
loan_df['Married'] = married_labels
#for dependents
label_encoder.fit(loan_df['Dependents']) #finds pattern
dependent_labels = label_encoder.transform(loan_df['Dependents']) #actually applies it
loan_df['Dependents'] = dependent_labels
#for education
label_encoder.fit(loan_df['Education']) #finds pattern
edu_labels = label_encoder.transform(loan_df['Education']) #actually applies it
loan_df['Education'] = edu_labels
#self-employed
label_encoder.fit(loan_df['Self_Employed']) #finds pattern
emp_labels = label_encoder.transform(loan_df['Self_Employed']) #actually applies it
loan_df['Self_Employed'] = emp_labels
#property area
label_encoder.fit(loan_df['Property_Area']) #finds pattern
emp_labels = label_encoder.transform(loan_df['Property_Area']) #actually applies it
loan_df['Property_Area'] = emp_labels

print(loan_df.head())

'''''
#You want to check if there is any correlation (ex. married/education and getting loan)

sns.countplot(x = 'Education', hue = 'Loan_Status', data = loan_df)
plt.show()
#we can see that we have more orange when people are graduated
#so number of loans approved for graduates is greater than not graduated

sns.countplot(x = 'Married', hue = 'Loan_Status', data = loan_df)
plt.show()
#People who are married have a greater chance of being approved for a loan
sns.countplot(x = 'Gender', hue = 'Loan_Status', data = loan_df)
plt.show()

'''
#seperate data and label
X = loan_df.drop(columns = ['Loan_Status', 'Loan_ID'], axis = 1)
Y = loan_df['Loan_Status']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 3)
#stratify based on Y means we want to split the Y outcomes evenly
#ex. if you dont mention stratify you can get a lot of 0s in train but none in test

#Training the model using support vector machine model
classifier = svm.SVC(kernel = 'linear')
classifier.fit(X_train, Y_train) #fit trains our model

xtrain_pred = classifier.predict(X_train)
score = accuracy_score(xtrain_pred, Y_train)
print('Training data', score)
#you want to see training data because you want to see if there is overfitting
#if there is overffiting then the training data would be very high compared to the 
#testing data

y_pred = classifier.predict(X_test)
score = accuracy_score(y_pred, Y_test)
print('Testing data', score)


#cross scores
cross_val_scores = cross_val_score(classifier, X_train, Y_train, cv=5)
print(f'Cross-Validation Accuracy Scores: {cross_val_scores}')
print(f'Mean Cross-Validation Accuracy: {cross_val_scores.mean()}')

