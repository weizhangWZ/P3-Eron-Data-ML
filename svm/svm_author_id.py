#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import collections

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

### get less data
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 


#########################################################
### your code goes here ###

### linear kernel, training time = 195.456s, 0.114 for 1% data
#clf = SVC(kernel = "linear")
clf = SVC(kernel = "rbf",C=10000)

### try search parameters
#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10000]}
#svc = SVC()
#clf = GridSearchCV(svc, parameters)


t0 = time()
clf.fit(features_train,labels_train)
print "training time:", round(time()-t0, 3), "s"
pred = clf.predict(features_test)
t0 = time()
print "Predict time:", round(time()-t0, 3), "s"
pred_rate = accuracy_score(pred, labels_test)
print "accuracy", pred_rate

# know the best params
#print clf.best_params_


# elements in SVM
#print pred[10],pred[26],pred[50]

# Who's Chris
results = collections.Counter(pred)

print results[1]
#########################################################


