#Importing Modules:
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso

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
    print('----------------------------------------------------------------------------------------------')
    print('Linear Regression:')
    print('------------------')
    XM = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    θM = np.ones((X_train.shape[1]+1,1))
    m = len(XM)
    epochs = 10000
    α = 0.1
    for i in range(epochs):
        ypred = np.dot(θM.T, XM.T)
        #print(ypred)
        djw = -(2 / m) * np.dot(XM.T,y_train-ypred.T)
        θM = θM - α * djw
    
    θ = []
    for i in range(θM.shape[0]):
        θ.append('θ'+str(i))
    print('Linear Module trained and values of θ are :\n',pd.DataFrame(θM,index=θ,columns=['θ']))
    
    #Predicting & Testing:
    XM_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    predicted = np.dot(θM.T, XM_test.T)
    predicted = predicted.T
    
    
    print('MSE : '+str(metrics.mean_squared_error(y_test,predicted)))
    print('R2 Score : '+str(metrics.r2_score(y_test,predicted)))
    print('----------------------------------------------------------------------------------------------')

def Polynomial_Regression(X_train, X_test, y_train, y_test):
    print('Polynomial Regression:')
    print('----------------------')
    pol = PolynomialFeatures(degree=2)
    X_pol = pol.fit_transform(X_train)


    linear = LinearRegression()
    linear.fit(X_pol,y_train)

    y_pred = linear.predict(pol.transform(X_test))
    print('MSE : '+str(metrics.mean_squared_error(y_test,y_pred)))
    print('R2 Score : '+str(metrics.r2_score(y_test,y_pred)))  
    print('----------------------------------------------------------------------------------------------')

    
def Lasso_Regression(x_train,x_test,y_train,y_test):
    print('Lasso Regression:')
    print('-----------------')
    lasso=Lasso(alpha=1.0)
    lasso.fit(x_train,y_train)
    y_pred=lasso.predict(x_test)
    print('MSE : '+str(metrics.mean_squared_error(y_test,y_pred)))
    print('R2 Score : '+str(metrics.r2_score(y_test,y_pred)))
    print('----------------------------------------------------------------------------------------------')
