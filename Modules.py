#Importing Modules:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso,Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def Linear_Regression(X_train, X_test, y_train, y_test):
    print('----------------------------------------------------------------------------------')
    print('Linear Regression:')
    print('------------------')
    #some operations on y_train , y_test:
    y_train = np.asarray(y_train)
    y_train = y_train[:, np.newaxis]
    y_test = np.asarray(y_test)
    y_test = y_test[:, np.newaxis]
    X_train = np.array(X_train)
    XM = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    θM = np.ones((X_train.shape[1]+1,1))
    m = len(XM)
    epochs = 100000
    α = 0.663
    for i in range(epochs):
        ypred = np.dot(θM.T, XM.T)
        djw = -(2 / m) * np.dot(XM.T,y_train-ypred.T)
        θM = θM - α * djw
    
    θ = []
    for i in range(θM.shape[0]):
        θ.append('θ'+str(i))
    print('Linear Module trained and values of θ are :\n',pd.DataFrame(θM,index=θ,columns=['θ']))
    
    #Predicting & Testing:
    XM_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    y_pred = np.dot(θM.T, XM_test.T)
    y_pred = y_pred.T
    print('MSE : '+str(metrics.mean_squared_error(y_test,y_pred)))
    print('R2 Score : '+str(metrics.r2_score(y_test,y_pred)))
    print('----------------------------------------------------------------------------------')

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
    print('----------------------------------------------------------------------------------')

    
def Lasso_Regression(X_train,X_test,y_train,y_test):
    print('Lasso Regression:')
    print('-----------------')
    lasso=Lasso(alpha=0.001)
    lasso.fit(X_train,y_train)
    y_pred=lasso.predict(X_test)
    print('MSE : '+str(metrics.mean_squared_error(y_test,y_pred)))
    print('R2 Score : '+str(metrics.r2_score(y_test,y_pred)))
    print('----------------------------------------------------------------------------------')
    
def Ridge_Regression(X_train,X_test,y_train,y_test):
    print('Ridge Regression:')
    print('-----------------')
    ridge=Ridge(alpha=1.0)
    ridge.fit(X_train,y_train)
    y_pred=ridge.predict(X_test)
    print('MSE : '+str(metrics.mean_squared_error(y_test,y_pred)))
    print('R2 Score : '+str(metrics.r2_score(y_test,y_pred)))
    print('----------------------------------------------------------------------------------')

def Logistic_Regression(X_train, X_test, y_train, y_test):
    print('----------------------------------------------------------------------------------')
    print('Logistic Regression:')
    print('--------------------')
    
    # train a logistic regression classifier using one-vs-one strategy
    lr = LogisticRegression(C=9.0).fit(X_train, y_train)

    
    # make predictions on test data
    y_pred = lr.predict(X_test)
    print('Confusion Matrix:')
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
    cm_display.plot()
    plt.show()
    
    # calculate accuracy
    accuracy = metrics.accuracy_score(y_test, y_pred) 
    print("Accuracy :", accuracy)
    print('----------------------------------------------------------------------------------')

def SVM(X_train,X_test,y_train,y_test):
    print('Support Vector Machine:')
    print('-----------------------')
    svm = SVC(C = 3)
    svm.fit(X_train,y_train)
    y_pred=svm.predict(X_test)
    print('Confusion Matrix:')
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
    cm_display.plot()
    plt.show()
    print("Accuracy : ",metrics.accuracy_score(y_test,y_pred))
    print('----------------------------------------------------------------------------------')


def Dceision_Tree(X_train, X_test, y_train, y_test):
    print('Decision Tree:')
    print('--------------')
    clf = DecisionTreeClassifier(max_depth=7)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Confusion Matrix:')
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
    cm_display.plot()
    plt.show()
    print("Accuracy :",metrics.accuracy_score(y_test, y_pred))
    print('----------------------------------------------------------------------------------')

def KNN(X_train,X_test,y_train,y_test):
    print('K Nearest Neighbours:')
    print('---------------------')
    knn=KNeighborsClassifier(n_neighbors=21)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    print('Confusion Matrix:')
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
    cm_display.plot()
    plt.show()
    print('Accuracy : ' +str(metrics.accuracy_score(y_test,y_pred)))
    print('----------------------------------------------------------------------------------')