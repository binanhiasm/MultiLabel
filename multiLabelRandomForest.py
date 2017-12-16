#TRAN QUANG DAT - 14520156
#created in December 15 2017
#MultiLabel-DecicisionTree

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.datasets import load_iris
import random
import math

#Load Iris Data from sklearn
iris = load_iris()
#find Average Accuracy
total = 0
for i in range(0,10):
    #Use Pandas method to import data into pandas DataFrame, save them in balanced_data
    balance_data = pd.DataFrame(iris.data, columns=iris.feature_names)
    #Check the length and dimension of DataFrame
    lengthData = len(balance_data)
#print ("Dataset Length:",lengthData)
#print ("Dataset Shape:: ",balance_data.shape )
#print ("Dataset:")
#print (balance_data)
#Split data into train and test
    X = balance_data.values[:, 1:4]
    Y = balance_data.values[:,0]
    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)
    #Devide dataset into train and test with 7:3
    numof_data_train = math.ceil(lengthData*0.7)
    numof_data_test = math.floor(lengthData*0.3)
    #Determine data_train by 1 and data_test by 0
    data_train = [1 for i in range(0,numof_data_train)]
    data_train.extend([0 for i in range(0,numof_data_test)])
    #Use random Shuffe to mix data
    random.shuffle(data_train)
    #Creat a column to represent for Category of Data and a column to represent for Data in Train or Test
    balance_data['Category'] = pd.Categorical.from_codes(iris.target,iris.target_names)
    balance_data['Infor'] = data_train
    #Train data
    train, test = balance_data[balance_data['Infor']==1], balance_data[balance_data['Infor']==0]
    real_labels = pd.factorize(test['Category'])[0]
    y = pd.factorize(train['Category'])[0]
    x = train[balance_data.columns[:4]]
    #Apply Random Forest of Sklearn
    clf = RandomForestClassifier(n_jobs=2, random_state=0)
    clf = clf.fit(x,y)
    #Test
    predicted_labels = clf.predict(test[balance_data.columns[:4]])
    #Compare real and test
    print("Real labels: ")
    print(real_labels)
    print("Predicted labels: ")
    print(predicted_labels)
    #Apply Accuracy_score for checking accuracy

    temp = accuracy_score(real_labels,predicted_labels)*100
    total = total + temp
    print ("Accuracy is", temp)
print ("Average Accuracy of 10 tests:", total/10)