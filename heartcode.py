import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading the csv data as panda data file
heart_data=pd.read_csv('heart.csv')

#print first 5 rows
print(heart_data.head())
# print(heart_data.tail())

#print the number of colums and rows function
print(heart_data.shape)

#getting more information
#checking for mising values or data
print(heart_data.info())

#to print null caharacters
# print(heart_data.isnull().sum())

#for mathematical details
# print(heart_data.describe())

#no of values of 0 and 1
print(heart_data['target'].value_counts())

X=heart_data.drop(columns='target',axis=1)
Y=heart_data['target']

print(X)


#SPLITTING THE DATA into training and testing
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

print(X.shape,X_test.shape,X_train.shape)

#model and training

model=LogisticRegression()
model.fit(X_train,Y_train)

#model evaluation
#accuracy on training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print('The accuracy is',training_data_accuracy)


#accuracy on testing data
X_test_prediction=model.predict(X_test)
testing_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print('The testing accuracy is',testing_data_accuracy)

#building a predictive system
input_data=(34,0,1,118,210,0,1,192,0,0.7,2,0,2)
input_data_as_array=np.asarray(input_data)
input_data_reshaped=input_data_as_array.reshape(1,-1)

prediction=model.predict(input_data_reshaped)
if(prediction[0]==0):
    print("The person is unlikely from suffering any heart related diseases")
else:
    print("There is a hich chance that the person might be suffering from heart diseases")