#Importing Modules:
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split

def Data_splitter(X,Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2,shuffle=True,random_state=10)
    #some operations on y_train , y_test:
    y_train = np.asarray(y_train)
    y_train = y_train[:, np.newaxis]
    y_test = np.asarray(y_test)
    y_test = y_test[:, np.newaxis]
    X_train = np.array(X_train)
    return X_train, X_test, y_train, y_test

def Linear_Regression(X_train, X_test, y_train, y_test):
    XM = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    θM = np.zeros((X_train.shape[1]+1,1))
    m = len(XM)
    epochs = 10000
    α = 0.31
    for i in range(epochs):
        ypred = np.dot(θM.T, XM.T)
        djw = -(2 / m) * np.dot(XM.T,y_train-ypred.T)
        θM = θM - α * djw
    
    θ = []
    for i in range(θM.shape[0]):
        θ.append('θ'+str(i))
    print('Linear Module trained and values of θ are :\n',pd.DataFrame(θM,index=θ,columns=['θ']))
    print("----------------------------------------------------------------------------------------------")
    
    #Predicting & Testing:
    print('Testing & Calculating MSE.......\n')
    XM_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    predicted = np.dot(θM.T, XM_test.T)
    predicted = predicted.T
    
    MSE = metrics.mean_squared_error(y_test, predicted)
    print('MSE after Linear Regression', MSE)