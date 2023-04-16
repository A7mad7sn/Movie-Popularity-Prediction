#Importing Modules:
import pandas as pd
from Modules import Data_splitter,Linear_Regression

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)

#Loading Movies Data:
Data = pd.read_csv("movies-regression-dataset.csv")
print(Data)

X = Data.iloc[:,0:-1]#Features
Y = Data.iloc[:,-1]#Label

#Data Preprocessing:
"""
THIS SECTION STILL REQUIRED
"""
#Data Splitting:
X_train, X_test, y_train, y_test =  Data_splitter(X,Y)

#Linear Regression(resulting run time error for now as the data isn't preprocessed yet):
Linear_Regression(X_train, X_test, y_train, y_test)