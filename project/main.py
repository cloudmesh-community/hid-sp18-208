import numpy as np
import pandas as pd
import sklearn

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



import requests
from flask import Flask
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC
from os import listdir
from flask import Flask, request

app = Flask(__name__)

def download_data(url, filename):
    r = requests.get(url, allow_redirects=True)
    open(filename, 'wb').write(r.content)

def download_data_1(url, filename):
    r = requests.get(url, allow_redirects=True)
    open(filename, 'wb').write(r.content)

def download_data_2(url, filename):
    r = requests.get(url, allow_redirects=True)
    open(filename, 'wb').write(r.content)


def age_approx(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 0
        elif Pclass == 2:
            return 0
        else:
            return 0
    else:
        return Age



def clean_train_data(filename):

    train = pd.read_csv(filename)
    train.columns = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
    train.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)
    train['Age'] = train[['Age', 'Pclass']].apply(age_approx, axis=1)
    train.dropna(inplace=True)
    gender = pd.get_dummies(train['Sex'],drop_first=True)
    embark_location = pd.get_dummies(train['Embarked'],drop_first=True)
    train.drop(['Sex', 'Embarked'],axis=1,inplace=True)
    train_new = pd.concat([train,gender,embark_location],axis=1)
    train_new.drop(['Fare', 'Pclass'],axis=1,inplace=True)

    x_train = train_new.drop("Survived", axis=1)
    y_train = train_new["Survived"]

    return x_train, y_train


def clean_test_data(filename):

    test = pd.read_csv(filename)
    test.columns = ['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
    test.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)


    test['Age'] = test[['Age', 'Pclass']].apply(age_approx, axis=1)
    test.fillna(0, inplace=True)
    gender_test = pd.get_dummies(test['Sex'],drop_first=True)
    embark_test = pd.get_dummies(test['Embarked'],drop_first=True)
    test.drop(['Sex', 'Embarked'],axis=1,inplace=True)
    test_new = pd.concat([test,gender_test,embark_test],axis=1)
    test_new.drop(['Fare', 'Pclass'],axis=1,inplace=True)


    test_survive = pd.read_csv('data/gender_submission.csv')
    test_survive.columns = ['PassengerId','Survived']

    x_test = test_new
    y_test = test_survive["Survived"]

    return x_test, y_test

        
@app.route('/')
def index():
    return "Ronnie Project!"
#Passing dynamic data
#http://127.0.0.1:5000/api/download/url/url='www.google.com'
#When passing a url make sure to remove https or http part from there
@app.route('/api/download/url/<url>', methods = ['GET'])
def dynamicdownload(url):
    data = url
    
    return data

@app.route('/api/download/data')
def download():
    url = 'https://www.dropbox.com/s/jsckwgmfdor69cf/train.csv?dl=0'
    download_data(url=url, filename='train.csv')
    return "Train Data Downloaded"

@app.route('/api/download/data1')
def download_1():
    url = 'https://www.dropbox.com/s/dn5vxvzsn8djjio/test.csv?dl=0'
    download_data_1(url=url, filename='test.csv')
    return "Test Data Downloaded 2"

@app.route('/api/download/data2')
def download_2():
    url = 'https://www.dropbox.com/s/focadrr2cughv3c/gender_submission.csv?dl=0'
    download_data_2(url=url, filename='gender_submission.csv')
    return "Predict Data Downloaded"

    
@app.route('/api/gettrain')
def gettraindata():
    Xtrain, ytrain = clean_train_data("data/train.csv")
    
    return "Return Xtrain and Ytrain arrays" + str(Xtrain)


@app.route('/api/gettest')
def gettestdata():
    Xtest, ytest = clean_test_data("data/test.csv")
    
    return "Return Xtest and Ytest arrays" + str(Xtest)



@app.route('/api/logreg')
def log():
    Xtest, ytest = clean_test_data("data/test.csv")
    Xtrain, ytrain = clean_train_data("data/train.csv")

    
    LogReg = LogisticRegression()
    LogReg.fit(Xtrain, ytrain)

    ypred = LogReg.predict(Xtest)
    matrix = confusion_matrix(ytest, ypred)

    wrong = matrix[0][1] + matrix[1][0]
    currect = matrix[0][0] + matrix[1][1]

    accuracy = (currect)/(wrong + currect)  

    print("Prediction Accuracy: "+str(accuracy))
    return "Prediction Accuracy: "+str(accuracy)

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port =8080)

