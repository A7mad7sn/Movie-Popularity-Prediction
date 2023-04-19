#Importing Modules:
import pandas as pd
from Modules import Data_splitter,Linear_Regression,Polynomial_Regression,Lasso_Regression
from preprocessing import Preprocessing

#Loading Movies Dataset:
Movie_Data = pd.read_csv("movies-regression-dataset.csv")

#Data Preprocessing:
Movie_Data = Preprocessing(Movie_Data)
X = Movie_Data.iloc[:,0:-1]#Features
Y = Movie_Data.iloc[:,-1]#Label

#Data Splitting:
X_train, X_test, y_train, y_test =  Data_splitter(X,Y)

#Linear Regression(resulting run time error for now as the data isn't preprocessed yet):
Linear_Regression(X_train, X_test, y_train, y_test)

#Polynomial Regression:
Polynomial_Regression(X_train, X_test, y_train, y_test)

#Lasso Regression:
Lasso_Regression(X_train, X_test, y_train, y_test)
