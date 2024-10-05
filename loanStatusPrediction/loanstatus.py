'''''
In this part we used logistic regression, it is seen that it only has around an 82%
accuracy score. We can do better. In the testing data, it gives it at 73, which is worse. 
'''


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


loan_df = pd.read_csv("/Users/naomigong/Coding/ML Projects/loanStatusPrediction/loandata.csv")
# Set the display option to show all columns
pd.set_option('display.max_columns', None)
print(loan_df.shape)
print(loan_df.info())
print(loan_df.isnull().sum())
print(loan_df.describe())
print(loan_df.head())


#filling in missing values
fig, ax = plt.subplots(figsize=(10,5)) #you can use any dimesion
loan_df['Dependents'].fillna("", inplace = True)
loan_df['Gender'].fillna("" ,inplace = True)
loan_df['Married'].fillna("",inplace = True)
loan_df['Self_Employed'].fillna("",inplace = True)
loan_df['LoanAmount'].fillna(loan_df['LoanAmount'].median(),inplace = True)
loan_df['Loan_Amount_Term'].fillna(loan_df['Loan_Amount_Term'].median(),inplace = True)
loan_df['Credit_History'].fillna(loan_df['Credit_History'].median(),inplace = True)
print(loan_df.isnull().sum())

#you can improve the model by fixing the class imbalance
approved = loan_df[loan_df.Loan_Status == 'Y']
not_approved = loan_df[loan_df.Loan_Status == 'N']

approved_sample = approved.sample(192)
loan_df_combined  =  pd.concat([not_approved, approved_sample], axis = 0)
'''''
#label encoding
label_encoder= LabelEncoder()
#for status
label_encoder.fit(loan_df['Loan_Status']) #finds pattern
labels = label_encoder.transform(loan_df['Loan_Status']) #actually applies it
loan_df['Loan_Status'] = labels
#for gender
label_encoder.fit(loan_df['Gender']) #finds pattern
gender_labels = label_encoder.transform(loan_df['Gender']) #actually applies it
loan_df['Gender'] = gender_labels
#for marriage
label_encoder.fit(loan_df['Married']) #finds pattern
married_labels = label_encoder.transform(loan_df['Married']) #actually applies it
loan_df['Married'] = married_labels

label_encoder.fit(loan_df['Dependents']) #finds pattern
dependent_labels = label_encoder.transform(loan_df['Dependents']) #actually applies it
loan_df['Dependents'] = dependent_labels

label_encoder.fit(loan_df['Education']) #finds pattern
edu_labels = label_encoder.transform(loan_df['Education']) #actually applies it
loan_df['Education'] = edu_labels

label_encoder.fit(loan_df['Self_Employed']) #finds pattern
emp_labels = label_encoder.transform(loan_df['Self_Employed']) #actually applies it
loan_df['Self_Employed'] = emp_labels

label_encoder.fit(loan_df['Property_Area']) #finds pattern
emp_labels = label_encoder.transform(loan_df['Property_Area']) #actually applies it
loan_df['Property_Area'] = emp_labels
'''

#to improve the model you may want to try one hot encoding
# One-hot encoding
loan_df_encoded = pd.get_dummies(loan_df_combined, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'])


Y = loan_df_encoded['Loan_Status']
X =loan_df_encoded.drop(columns = ['Loan_Status', 'Loan_ID'], axis = 1)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)

model = LogisticRegression(max_iter = 1000)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

score = accuracy_score(y_pred, Y_test)
print(score)

print(loan_df["Loan_Status"].value_counts())


