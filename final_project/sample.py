estimators = [("reduce_dim",PCA()),
              ('clf',AdaBoostClassifier())]
clf = Pipeline(estimators)
params = { "clf__n_estimators":[10,20, 25, 30, 40, 50],
           "reduce_dim__n_components":[2,10,21]}
clf = GridSearchCV(clf, params)
clf = clf.fit(features_train, labels_train)
clf = clf.best_estimator_
print clf
