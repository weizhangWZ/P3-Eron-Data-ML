#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import numpy as np

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
#plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary


##### K nearestNEIGHBOR ALGORITHM

function_name = "AdaBoostClassifier(new)"

#adaboost classifier
clf = AdaBoostClassifier(SVC(probability=True,kernel='rbf',C=10001),n_estimators=10,random_state = 1)


### try search parameters
#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10000]}
#svc = SVC()
#clf = GridSearchCV(svc, parameters)

#adaboost regressor
#clf = AdaBoostRegressor(DecisionTreeRegressor(min_samples_split=2,max_depth=6),n_estimators=9, random_state=1)

clf.fit(features_train,labels_train)
pred = clf.predict(features_test)

# know the best params
#print clf.best_params_

# simple pred
pred_rate = accuracy_score(pred, labels_test)

# numpred
#pred_mat = [round(pred[i]) for i in pred]
#pred_rate = accuracy_score(pred_mat, labels_test)
print pred_rate

### draw the decision boundary with the text points overlaid
prettyPicture(function_name,clf, features_test, labels_test)

