#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data_dict.pop(features[0], 0 )
data_dict.pop(features[1], 0 )
data = featureFormat(data_dict, features)


for point in data:
    salary = point[0]
    bonus = point[1]
    if salary > 25000000:
        pass
    else:
        matplotlib.pyplot.scatter( salary, bonus )

   
for name in data_dict:
    if data_dict[name]["bonus"] == 8000000:
        print name

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

### your code below



