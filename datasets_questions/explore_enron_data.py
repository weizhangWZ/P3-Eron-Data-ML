#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
POI = []
ALL = []
for i in enron_data:
    if enron_data[i]["poi"] == 1:
        POI.append(i)
    ALL.append(i)
    #print len(list(enron_data[i].keys()))
#print ALL

list_name = ["Lay","Skilling","Fastow"]

def name_correction(name):
    list_names = name.split(" ")
    first_name = list_names[0]
    last_name = list_names[-1]
    new_name = last_name+" " + first_name
    middle_name = list_names[1:-1]
    if middle_name!=None:
        for n in middle_name:
            new_name = new_name + " "+ n[0]
    return new_name.upper()

# correct the typing style
#name =  name_correction(name)

# blur search
def name_exist(name, ALL):
    for n in ALL:
        if n.find(name.upper())!=-1:
            return n
    return False

salary_flag, email_flag = 0,0
for name in enron_data:
    if enron_data[name]["total_payments"] == "NaN":
 
        print name,enron_data[name]["poi"]
    if enron_data[name]["poi"] == 1:
        email_flag +=1
    print 
print email_flag,len(enron_data)


#with open("../final_project/poi_names.txt",'r') as names:
#    for n,i in enumerate(names):
#        print n,i
        
