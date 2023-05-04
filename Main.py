#Importing Modules:
import pandas as pd
from sklearn.model_selection import train_test_split
from Modules import Linear_Regression,Polynomial_Regression,Lasso_Regression,Ridge_Regression
from preprocessing import Preprocessing_regression,Test_Script_regression,Preprocessing_classification,Test_Script_classification

print('\nWelcome to Movie Popularity Prediction, please choose models you want:\n')
print('[1] Regression models.')
print('[2] Classification models.\n')
Choice = 0
while(True):
    Choice = (input('Choice --> '))
    if(Choice.isdigit() == False or int(Choice)!=1 and int(Choice)!=2):
        print('Invalid Choice!')
        continue
    else:
        Choice = int(Choice)
        break
    

if(Choice==1):
    #Loading Movies-regression Dataset:
    Movie_Data = pd.read_csv("movies-regression-dataset.csv")
    X = Movie_Data.iloc[:,0:-1]#Features
    Y = Movie_Data.iloc[:,-1]#Label
    
    #Data Splitting:
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2,shuffle=True,random_state=10)
    
    #Data Preprocessing:
    Train_data = pd.concat([X_train, y_train], axis=1, join="inner")
    Test_data = pd.concat([X_test, y_test], axis=1, join="inner")
    
    Train_data,Trained_variables = Preprocessing_regression(Train_data)
    Test_data = Test_Script_regression(Test_data,Trained_variables)
    
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


elif(Choice==2):
    #Loading Movies-regression Dataset:
    Movie_Data = pd.read_csv("movies-classification-dataset.csv")
    X = Movie_Data.iloc[:,0:-1]#Features
    Y = Movie_Data.iloc[:,-1]#Label
    
    #Data Splitting:
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2,shuffle=True,random_state=10)
    
    #Data Preprocessing:
    Train_data = pd.concat([X_train, y_train], axis=1, join="inner")
    Test_data = pd.concat([X_test, y_test], axis=1, join="inner")
    
    Train_data,Trained_variables = Preprocessing_classification(Train_data)
    Test_data = Test_Script_classification(Test_data,Trained_variables)
    
    X_train=Train_data.iloc[:,0:-1]
    y_train=Train_data.iloc[:,-1]
    X_test=Test_data.iloc[:,0:-1]
    y_test=Test_data.iloc[:,-1]
    
    print("Classification Models did not implemented yet!\n")
    print("We need to implement three Classifiers\n[1]Logistic Regression\n[2]SVM\n[3]Decision Trees\n")