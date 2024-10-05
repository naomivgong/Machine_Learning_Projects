import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import seaborn as sns


creditcard_df = pd.read_csv("creditcardfraud/creditcard.csv")

#preprocessing
print(creditcard_df.head())
print(creditcard_df.info())
print(creditcard_df.isnull().sum()) #find that there is no null values
print(creditcard_df["Class"].value_counts()) #we notice there is an impalance set so we want to balance

fraud = creditcard_df[creditcard_df.Class == 1]
legit = creditcard_df[creditcard_df.Class == 0]

legit_sample = legit.sample(492)
print(fraud.shape)
print(legit_sample.shape)

resample_creditcard_df = pd.concat([legit_sample, fraud], axis = 0)

#data analysis
# Calculate the correlation matrix
corr_matrix = creditcard_df.corr()
plt.figure(figsize=(20,16))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, annot_kws={"size": 8})

legit.Amount.describe()
'''count    284315.000000
mean         88.291022
std         250.105092
min           0.000000
25%           5.650000
50%          22.000000
75%          77.050000
max       25691.160000'''

fraud.Amount.describe()
'''''count     492.000000
mean      122.211321
std       256.683288
min         0.000000
25%         1.000000
50%         9.250000
75%       105.890000
max      2125.870000''' #we notice fraud mean is higher

#compare the values for both transactions
print(creditcard_df.groupby('Class').mean()) 
#this comparison helps us use machine learning sample


#seperate input and output
X = resample_creditcard_df.drop('Class', axis = 1)
Y = resample_creditcard_df['Class']


#train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify= Y, random_state= 2)

#you can standardize to make more accurate
scaler = StandardScaler()
scaler.fit(X_train)
X_train_standardized = scaler.transform(X_train)
X_test_standardized = scaler.transform(X_test)



model = LogisticRegression()
model.fit(X_train_standardized, Y_train)
y_pred = model.predict(X_train_standardized)
cross_val_scores_training = cross_val_score(model, X_train_standardized, Y_train, cv = 5)
print(f'Cross-Validation Training Accuracy Scores: {cross_val_scores_training}')
print(f'Mean Cross-Validation Training Accuracy: {cross_val_scores_training.mean()}')


y_pred = model.predict(X_test_standardized)
cross_val_scores_testing = cross_val_score(model, X_test, Y_test, cv = 5)
print(f'Cross-Validation Testing Accuracy Scores: {cross_val_scores_testing}')
print(f'Mean Cross-Validation Testing Accuracy: {cross_val_scores_testing.mean()}')
