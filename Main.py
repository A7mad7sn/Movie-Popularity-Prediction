#Importing Modules:
import pandas as pd
from sklearn.model_selection import train_test_split
from Models import Linear_Regression,Polynomial_Regression,Lasso_Regression,Ridge_Regression,Logistic_Regression,SVM,Dceision_Tree,KNN
from Preprocessing import Preprocessing_Training_Regression,Preprocessing_Testing_Regression,Test_Script_Regression,Preprocessing_Training_Classification,Preprocessing_Testing_Classification,Test_Script_Classification
import warnings
warnings.filterwarnings("ignore")

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
    

while(True):
    #Regression:
    if(Choice==1):
        #Loading Movies-regression Dataset:
        Movie_Data = pd.read_csv("movies-regression-dataset.csv")
        X = Movie_Data.iloc[:,0:-1]#Features
        Y = Movie_Data.iloc[:,-1]#Label
        
        #Data Splitting:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2,shuffle=True,random_state=10)
        
        #Discussion Day Code for Testing Data:
        #Test_Data = pd.read_csv("REGRESSION TESTING DATA.csv")
        #X_test = Test_Data.iloc[:,0:-1]#Features
        #y_test = Test_Data.iloc[:,-1]#Label
        
        
        #Data Preprocessing:
        Train_data = pd.concat([X_train, y_train], axis=1, join="inner")
        Test_data = pd.concat([X_test, y_test], axis=1, join="inner")
        
        Train_data,Trained_variables = Preprocessing_Training_Regression(Train_data)
        Test_data = Preprocessing_Testing_Regression(Test_data,Trained_variables)
        #Test_data = Test_Script_Regression(Test_data,Trained_variables)
        
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
        
        #See if user want to continue:
        print('\nsee Classification models ? [y/n]')
        while(True):
            Option = input('-->')
            if(type(Option)!=str or str(Option).lower() != 'y'and str(Option).lower() != 'n'):
                print('Invalid Option!')
                continue
            elif(str(Option).lower() == 'y'):
                Choice = 2
                break
            else:
                break
        if(Choice == 2):
            continue
        else:
            break
    
    #Classification:
    elif(Choice==2):
        #Loading Movies-regression Dataset:
        Movie_Data = pd.read_csv("movies-classification-dataset.csv")
        X = Movie_Data.iloc[:,0:-1]#Features
        Y = Movie_Data.iloc[:,-1]#Label
        
        #Data Splitting:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2,shuffle=True,random_state=10)
        
        #Discussion Day Code for Testing Data:
        #Test_Data = pd.read_csv("CLASSIFICATION TESTING DATA.csv")
        #X_test = Test_Data.iloc[:,0:-1]#Features
        #y_test = Test_Data.iloc[:,-1]#Label
        
        #Data Preprocessing:
        Train_data = pd.concat([X_train, y_train], axis=1, join="inner")
        Test_data = pd.concat([X_test, y_test], axis=1, join="inner")
        
        Train_data,Trained_variables = Preprocessing_Training_Classification(Train_data)
        Test_data = Preprocessing_Testing_Classification(Test_data,Trained_variables)
        #Test_data = Test_Script_Classification(Test_data,Trained_variables)
        
        X_train=Train_data.iloc[:,0:-1]
        y_train=Train_data.iloc[:,-1]
        X_test=Test_data.iloc[:,0:-1]
        y_test=Test_data.iloc[:,-1]
        
        
        #Logistic_Regression:
        Logistic_Regression(X_train,X_test,y_train,y_test)
            
        #Support_Vector_Machine:
        SVM(X_train,X_test,y_train,y_test)
        
        #Decision_Tree:
        Dceision_Tree(X_train,X_test,y_train,y_test)
        
        #K_Nearest_Neighbors:
        KNN(X_train,X_test,y_train,y_test)
        
        #See if user want to continue:
        print('\nsee Regression models ? [y/n]')
        while(True):
            Option = input('-->')
            if(type(Option)!=str or str(Option).lower() != 'y'and str(Option).lower() != 'n'):
                print('Invalid Option!')
                continue
            elif(str(Option).lower() == 'y'):
                Choice = 1
                break
            else:
                break
        if(Choice == 1):
            continue
        else:
            break