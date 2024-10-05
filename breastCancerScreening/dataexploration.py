"""EDA Checklist

Understanding the dataset and its shape
Checking the datatype of each column
Categorical and Numerical Columns
Checking for missing values
Descriptive summary of dataset
Grouby for classification problems"""

#import dependencies
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder #we dont want character based datatypes so try to use numerical values
import matplotlib.pyplot as plt
import seaborn as sns

breastCancerData = pd.read_csv('/Users/naomigong/Desktop/data.csv')


#1 understanding the dataset
#exploring the data 
#shows the first 5 rows of the data
print(breastCancerData.head())
    #from here we noticed that there is unnamed file


#remove the empty row using drop
#axis 0 means row, axis 1 means column we want column so axis = 1
#Inplace = true modifies the original dataframe
breastCancerData.drop(columns = "Unnamed: 32", axis = 1, inplace = True)
print(breastCancerData.head())

print("the shape is", breastCancerData.shape) 
#the first output is the rows, the second is the columns (569, 32) 569 rows, 32 columns


#2 check data types
print(breastCancerData.info()) #Non-Null Count tells you the amount of entries (full number = no missing values)

#3 determine numericla or categorical data
#we know diagnosis is the only categorical data type

#4 check for missing values
print(breastCancerData.isnull().sum()) #sum gives you the total in each column

#5 descriptive summary
print(breastCancerData.describe()) #descriptive statistics <- mean, std percentile (only done for numerical)

#6 Classification of Categorical Variables
#check distribution of target value
print(breastCancerData['diagnosis'].value_counts()) 
    #tallies the M and B 


#we need to give M and B labels (numerical)

label_encode = LabelEncoder()
labels = label_encode.fit_transform(breastCancerData['diagnosis']) #will give M and B 1 and 0
breastCancerData['target'] = labels #creates a new column called target made out of labels
breastCancerData.drop(columns = 'diagnosis', axis =1 , inplace= True) #removes the original


#1 -> M
#0 -> B

#Grouping the data based on the target
print(breastCancerData.groupby('target').mean()) #groups the data into the datas of target so 0 or 1. then takes the mean of the corresponding data
# by grouping you can compare the mean values for the various data points like radium mean, texture mean et



'''Doing Data Visualization'''
# Set the backend to 'TkAgg' (you can also try 'Qt5Agg' or 'MacOSX')
plt.switch_backend('TkAgg')

#histogram of the target values
sns.countplot(x = 'target', data = breastCancerData)

#to get columns in the data frame
for i in breastCancerData:
    #building a distribution plot
    sns.displot(x = i, data = breastCancerData) # i is each column
    plt.show()