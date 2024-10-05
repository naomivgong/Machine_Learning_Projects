#dependencies
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import imblearn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score

loan_df = pd.read_csv("/Users/naomigong/Coding/ML Projects/loanStatusPrediction/loandata.csv")

# Set the display option to show all columns
pd.set_option('display.max_columns', None)
print(loan_df.shape)
print(loan_df.info())
print(loan_df.isnull().sum())
print(loan_df.describe())
print(loan_df.head())

# Filling in missing values
loan_df['Dependents'].fillna("0", inplace=True)
loan_df['Gender'].fillna(loan_df['Gender'].mode()[0], inplace=True)
loan_df['Married'].fillna(loan_df['Married'].mode()[0], inplace=True)
loan_df['Self_Employed'].fillna(loan_df['Self_Employed'].mode()[0], inplace=True)
loan_df['LoanAmount'].fillna(loan_df['LoanAmount'].median(), inplace=True)
loan_df['Loan_Amount_Term'].fillna(loan_df['Loan_Amount_Term'].median(), inplace=True)
loan_df['Credit_History'].fillna(loan_df['Credit_History'].median(), inplace=True)

# One-hot encoding
loan_df_encoded = pd.get_dummies(loan_df, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'])

# Define X and Y
Y = loan_df_encoded['Loan_Status']
X = loan_df_encoded.drop(columns=['Loan_Status', 'Loan_ID'])

# Handling class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, Y_resampled = smote.fit_resample(X, Y)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2, random_state=2)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic regression model with hyperparameter tuning
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

# Accuracy score
score = accuracy_score(y_pred, Y_test)
print(f'Accuracy: {score}')

# Cross-validation to better estimate model performance
cross_val_scores = cross_val_score(model, X_resampled, Y_resampled, cv=5)
print(f'Cross-Validation Accuracy Scores: {cross_val_scores}')
print(f'Mean Cross-Validation Accuracy: {cross_val_scores.mean()}')
