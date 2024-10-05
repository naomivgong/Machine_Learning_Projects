import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


breastCancerData = pd.read_csv('breastCancerScreening/breast_cancer.csv')
breastCancerData.shape
#create a dataframe from this data set

#preprocess the data like you did earlier
breastCancerData.drop(columns = 'Unnamed: 32', axis = 1, inplace = True)
breastCancerData.drop(columns = 'id', axis = 1, inplace = True)
#remove null values
breastCancerData.dropna(inplace = True)

labelencoder = LabelEncoder()
labels = labelencoder.fit_transform(breastCancerData['diagnosis'])
breastCancerData['target'] = labels
breastCancerData.drop(columns = 'diagnosis', axis = 1, inplace = True)

#seperate the feature and target
targetData = breastCancerData['target']
breastCancerData.drop(columns = 'target', axis = 1, inplace = True)


#20% saved for testing, #the random state will be split in the same way -- how to produce the same splits
X_train, X_test, y_train, y_test = train_test_split(breastCancerData, targetData, test_size=0.2, random_state=42)
#since we are now working with a dataframe we do not use paranethese
print(breastCancerData.shape)
print(targetData.shape)

'''model training -- using logistic regression which is often used for binary classification'''
model = LogisticRegression() #loading instance of logistic regression
#fit to the training data
model.fit(X_train, y_train)

'''Model evaluation'''
y_pred = model.predict(X_test)
print(y_pred[:10])
print(y_test[:10])
accuracy = accuracy_score(y_pred, y_test)
print("The accuracy on our testing data is", accuracy)

'''Building a predictive system'''
inputData = (13.03,18.42,82.61,523.8,0.08983,0.03766,0.02562,0.02923,0.1467,0.05863,0.1839,2.342,1.17,14.16,0.004352,0.004899,0.01343,0.01164,0.02671,0.001777,13.3,22.81,84.46,545.9,0.09701,0.04619,0.04833,0.05013,0.1987,0.06169)
numpyinput = np.asarray(inputData) #changes the input data to a numpy array
inputDataReshape = numpyinput.reshape(1, -1) # one row and number of columns inferred from the length
prediction = model.predict(inputDataReshape) #this is returned as an array
print(prediction)
if prediction == 0:
    print("It is Benign")
else:
    print("it is malignant")





