#This model will use Linear Regresson
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

insurance_df = pd.read_csv("/Users/naomigong/Coding/ML Projects/medicalInsurance/insurance.csv")

#data preprocessing

#checking rows and columns
insurance_df.shape
#checking if there are null values
print(insurance_df.isnull().sum()) #no null values

#encoding
labelencoder = LabelEncoder()
sex_labels = labelencoder.fit_transform(insurance_df['sex'])
insurance_df['sex'] = sex_labels
children_labels = labelencoder.fit_transform(insurance_df['children'])
insurance_df['children'] = children_labels
smoker_labels = labelencoder.fit_transform(insurance_df['smoker'])
insurance_df['smoker'] = smoker_labels

print(insurance_df.head())
print(insurance_df['region'].value_counts())
region_labels = labelencoder.fit_transform(insurance_df['region'])
insurance_df['region'] = region_labels
#other option: insurance_df.replace({sex: {'male': 0, 'female': 1}})

#data analysis -- find distribution of age value 
plt.figure(figsize=(6,6))
sns.histplot(insurance_df['age']) #can see its more so contrentated on younger ppl


#for categorical use count
sns.countplot(x = 'sex', data = insurance_df)
plt.title('sex distribution') #we can see roughly equal distribution


#split input and outpit
X = insurance_df.drop(columns = 'charges', axis = 1)
Y = insurance_df['charges']

#split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state = 2)
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)
y_pred_train = lin_reg.predict(X_train)
y_pred_test = lin_reg.predict(X_test)

#calculating r^2 --- if r is closer to 1 means model does pretty well
#remember r^2 tells us what percent of the variability is because of the model

r2_train = metrics.r2_score(Y_train, y_pred_train)
print("R^2 training value is ", r2_train)
r2_test = metrics.r2_score(Y_test, y_pred_test)
print("R^2 testing value is ", r2_test)
#since training and testing accuracy is close = good model


#repeat with standardizing data. 
print("REDOING WITH STANDARDIZING")
scaler = StandardScaler()
scaler.fit(X_train)
X_train_standard = scaler.transform(X_train)
X_test_standard = scaler.transform(X_test)

lin_reg_with_standardization = LinearRegression()
lin_reg_with_standardization.fit(X_train_standard, Y_train)


y_train_pred_std = lin_reg_with_standardization.predict(X_train_standard)
r2_train_std = metrics.r2_score(Y_train, y_train_pred_std)
print("R^2 training value with standardization is ", r2_train_std)


y_test_pred_std = lin_reg_with_standardization.predict(X_test_standard)
r2_test_std = metrics.r2_score(Y_test, y_test_pred_std)
print("R^2 testing value  with standardization is ", r2_test_std)

print("Redoing with Random Forest")
forest_model = RandomForestRegressor().fit(X_train, Y_train)
y_pred_train_forest = forest_model.predict(X_train)
r2_train_randomforest = metrics.r2_score(Y_train, y_pred_train_forest)
print("the r2 score for random forest is", r2_train_randomforest)

y_pred_test_forest = forest_model.predict(X_test)
r2_test_randomforest = metrics.r2_score(Y_test, y_pred_test_forest)
print("the r2 score for random forest is", r2_test_randomforest)
