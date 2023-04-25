#Importing Modules:
import pandas as pd
from sklearn.model_selection import train_test_split
from Modules import Linear_Regression,Polynomial_Regression,Lasso_Regression,Ridge_Regression
from preprocessing import Preprocessing

#Loading Movies Dataset:
Movie_Data = pd.read_csv("movies-regression-dataset.csv")

#Data Preprocessing:
X = Movie_Data.iloc[:,0:-1]#Features
Y = Movie_Data.iloc[:,-1]#Label

#Data Splitting:
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2,shuffle=True,random_state=10)

#Data Preprocessing:
Train_data = pd.concat([X_train, y_train], axis=1, join="inner")
Test_data = pd.concat([X_test, y_test], axis=1, join="inner")
Train_data,Test_data = Preprocessing(Train_data,Test_data)
X_train=Train_data.iloc[:,0:-1]
y_train=Train_data.iloc[:,-1]
X_test=Test_data.iloc[:,0:-1]
y_test=Test_data.iloc[:,-1]

#Linear Regression:
Linear_Regression(X_train, X_test, y_train, y_test)

#Polynomial Regression:
Polynomial_Regression(X_train, X_test, y_train, y_test)

#Lasso Regression:
Lasso_Regression(X_train, X_test, y_train, y_test)

#Ridge Regression:
Ridge_Regression(X_train,X_test,y_train,y_test)