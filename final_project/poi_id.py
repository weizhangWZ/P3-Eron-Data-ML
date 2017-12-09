#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
from time import time
from sklearn.preprocessing import MinMaxScaler
from tester import test_classifier

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#### features_list all included below

features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances',
                 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 
                 'expenses', 'exercised_stock_options', 'long_term_incentive', 
                 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 
                 'from_messages', 'from_this_person_to_poi','shared_receipt_with_poi',
                 'fraction_to_poi', 'fraction_from_poi',
                  'other']

#features_list = ['poi','bonus','exercised_stock_options','fraction_to_poi']
#features_list = ['poi','fraction_to_poi', 'fraction_from_poi']
#features_list = ['poi','bonus','exercised_stock_options']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
   

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

#### Task 2: estimate the outliers and repete to remove outliers
def estimate_outliers(dataset,features_list):
    data = featureFormat(dataset, features_list)
    for point in data:
        salary = point[1]
        bonus = point[2]
        matplotlib.pyplot.scatter(salary, bonus)

    matplotlib.pyplot.xlabel("salary")
    matplotlib.pyplot.ylabel("bonus")
    matplotlib.pyplot.show()

def remove_outliers(dataset):
    remove_key = ""
    for n in dataset:
        if dataset[n]["salary"]!="NaN" and dataset[n]["salary"] > 20000000:
            remove_key = n
    dataset.pop(remove_key,0)
    print "=============================================================="
    print remove_key, "is an outliers"
    remove_key = 'THE TRAVEL AGENCY IN THE PARK'
    dataset.pop(remove_key,0)
    print remove_key, "is an outliers"
    remove_key = 'LOCKHART EUGENE E'
    dataset.pop(remove_key,0)
    print remove_key, "is an outliers"
    return dataset

my_dataset = remove_outliers(my_dataset)
#estimate_outliers(my_dataset,features_list)

#### Task 3: create new features
def create_features(dataset):
    fraction_from_poi = 0
    fraction_to_poi = 0
    print "=============================================================="
    print "Create new features fraction_to_poi, fraction_from_poi"
    for n in dataset:
        fraction_from_poi = computeFraction(dataset[n]['from_poi_to_this_person'], dataset[n]['to_messages'])
        fraction_to_poi = computeFraction(dataset[n]['from_this_person_to_poi'], dataset[n]['from_messages'])
        dataset[n]['fraction_from_poi'] = fraction_from_poi
        dataset[n]['fraction_to_poi'] = fraction_to_poi
    return dataset

def computeFraction(poi_messages, all_messages):
    fraction = 0
    if poi_messages != "NaN" and all_messages != "NaN":
        fraction = float(poi_messages)/all_messages
    return "%.3f" % fraction

my_dataset = create_features(my_dataset)
#print my_dataset

#### Analyzing the dataset
def analyze_data(dataset, features):
    print "=============================================================="
    print "The Enron data contains: ",len(data_dict)
    
    # poi description
    dataset,count_poi = turn_poi(dataset)
    print "There are", count_poi ,"POI"
    # features description
    features_string = ""
    counter = 0
    for feature in features:
        if counter > 0:
            features_string = features_string + feature + "|"
            dataset,count = turn_NaN(dataset, feature)
            print "There are", count, "NaN in feature", feature
        counter = 1
    print "=============================================================="
    print "There are", len(features)-1,"features, including", features_string
    return dataset

##### deal with poi and turn them into 1 and 0
def turn_poi(dataset):
    count_poi = 0
    for n in dataset:
        if dataset[n]["poi"] == True:
            dataset[n]["poi"] = 1
            count_poi+=1
        else:
            dataset[n]["poi"] = 0
    return dataset,count_poi

def turn_NaN(dataset,feature):
    count_NaN = 0
    for n in dataset:
        if dataset[n][feature] == "NaN":
            dataset[n][feature] = 0
            count_NaN+=1
    return dataset,count_NaN
        
my_dataset = analyze_data(my_dataset,features_list)
#print my_dataset.keys()


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#### Scaling negtive numbers
def scaling_data(dataset, features, features_list):
    scaler = MinMaxScaler()
    scaler.fit(features)
    features = scaler.transform(features)
    for n in dataset:
        for feature in features:
            for i, feature_name in enumerate(features_list):
                if i > 0:
                    dataset[n][feature_name] = feature[i-1]

#scaling_data(my_dataset, features, features_list)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
 
# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV


#### Gaussian_classifier
def Gaussian_classifier(features_train, labels_train, features_test, labels_test):
    t0 = time()
    estimators = [("reduce_dim",SelectKBest(k=6)),
              ('clf',GaussianNB())]
    clf = Pipeline(estimators)
    clf.fit(features_train,labels_train)
    print "=============================================================="
    print "This is a NB Classifier"
    print "The Training time:", round(time()-t0, 3), "s"

    clf.predict(features_test)
    t0 = time()
    print "The Predict time:", round(time()-t0, 3), "s"
    print "The accuracy:", "%.3f" % clf.score(features_test,labels_test)
    return clf


#### AdaBoost_classifier
def AdaBoost_classifier(features_train, labels_train, features_test, labels_test):
    t0 = time()
    estimators = [#("reduce_dim",PCA()),
                  ("reduce_dim",SelectKBest()),
                  ('clf',AdaBoostClassifier(DecisionTreeClassifier(criterion ='entropy', min_samples_split = 5),algorithm ='SAMME'))]
    clf = Pipeline(estimators)
    params = { #"reduce_dim__n_components":[13,14,15,16,17,18,19,20,21]
               "reduce_dim__k":[10,11,12,13,14,15],
               "clf__n_estimators":[1,2,5,10,15,20,50],
               }
    clf = GridSearchCV(clf, param_grid=params)
    clf = clf.fit(features_train, labels_train)
    print "=============================================================="
    print "This is a Adaboost Classifier"
    print "The Training time:", round(time()-t0, 3), "s"
    clf = clf.best_estimator_
    clf.predict(features_test)
    t0 = time()
    print "The Predict time:", round(time()-t0, 3), "s"
    print "The accuracy:", "%.3f" % clf.score(features_test,labels_test)
    return clf

#### KNeighbors_classifier
def KNeighbors_classifier(features_train, labels_train, features_test, labels_test):
    t0 = time()
    estimators = [#("reduce_dim",PCA()),
                  ("reduce_dim",SelectKBest()),
                  ('clf',KNeighborsClassifier())]
    clf = Pipeline(estimators)
    params = { #"reduce_dim__n_components":[13,14,15,16,17,18,19,20,21],
               "reduce_dim__k":[7,8,9,10,11,12,13,14,15],
               "clf__n_neighbors":[2,5,10,15,20,50],
               "clf__p":[2,3,4],
               "clf__leaf_size":[5,10]
               }
    clf = GridSearchCV(clf, param_grid=params)
    clf = clf.fit(features_train, labels_train)
    print "=============================================================="
    print "This is a KNeighbors Classifier"
    print "The Training time:", round(time()-t0, 3), "s"
    clf = clf.best_estimator_
    clf.predict(features_test)
    t0 = time()
    print "The Predict time:", round(time()-t0, 3), "s"
    print "The accuracy:", "%.3f" % clf.score(features_test,labels_test)
    return clf

####### The algorithm is chosen what features as parameters
def chosen_parameters(num,features_train,labels_train, features_list):
    print "=============================================================="
    clf = SelectKBest(k=num)
    clf.fit(features_train,labels_train)
    rankings = sorted(clf.scores_,reverse=True)
    for i in range(num):
        for ii, score in enumerate(clf.scores_):
            if rankings[i] == score:
                print "The score of feature", features_list[ii+1], "is", score


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

#### have chose six features as considerations for better performance
#chosen_parameters(len(features_list)-1,features_train,labels_train,features_list)

clf = Gaussian_classifier(features_train, labels_train, features_test, labels_test)

#clf = AdaBoost_classifier(features_train, labels_train, features_test, labels_test)

#clf = KNeighbors_classifier(features_train, labels_train, features_test, labels_test)


print "=============================================================="
#clf = GaussianNB()
#clf.fit(features_train,labels_train)

test_classifier(clf, my_dataset, features_list)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
