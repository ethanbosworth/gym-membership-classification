# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 2020

@author: Ethan Bosworth

A script to import group data from a Json file and output the data
transformed to a csv file
"""
#%% import modules
import pandas as pd 
import numpy as np
import re

#%% import data
data_test = pd.read_json("../Case/Input/test.json",orient = "split")
data_train = pd.read_json("../Case/Input/train.json",orient = "split")

#combine the data as each needs the same transformations
data = pd.concat([data_test,data_train]) 
data = data.reset_index()
test_size = len(data_test) # keep the size of the test data for later
#initiate empty lists
groups = [] # list of all groups including duplicates
idg = [] # id of the person in the group
date = [] # date the group was joined

temp = data.groups # temporary variable for reading the JSON
output = data.id #setup of the dataframe to output

for i in range(len(data)): # loop over all the JSON file
    temp1 = temp.iloc[i] # take the first entry 
    temp2 = temp1["data"] #open the data category to find the subentries
    for j in range(0,len(temp2)): # loop over the subentries
        temp3 = temp2[j]["group_name"] # take the name of the entry
        groups.append(temp3)  # add the entry to the groups list
        idg.append(i) # add the ID to the ID list
        date.append(temp2[j]["date_joined"]) # add the date to the date list
data_tuples = list(zip(idg,groups,date)) # concatenate the data into a refined form for use
group_data = pd.DataFrame(data_tuples, columns=["id",'group',"date joined"]) #rename the columns

#%% Refining the data
#%%% Number of groups

counts = group_data['group'].value_counts() # get a count of each group
#using counts it is possible to see the most popular groups
print(counts.head())
output = pd.concat([output,group_data['id'].value_counts()],axis = 1,keys=['id', 'n_of_groups']) # add to the output dataframe the number of groups each person is a member of
output["n_of_groups"] = output["n_of_groups"].fillna(0) # fill with 0 if person has no groups
 
counts = counts.reset_index() # reset index of counts for easy use

#%%% time in a group
time = pd.DataFrame(data["id"]) # setup a variable for the time in a group (years)
#convert the time joined to be the amount of years since joined
group_data["time_group"] = pd.to_datetime("now")-pd.to_datetime(group_data["date joined"])
group_data['time_group'] = group_data['time_group']//np.timedelta64(1,'Y')  


#%%% grouping the groups

#create some simple lists for groups based on words the group can contain
gym = ("Gym|Bodybuild|Weightlifting|Strongman")
health = ('Health|excercise|Nutrition|Workout|Fitness|lose weight|Vegan|ABS|Crossfit')
food = ("Recipes|Cooking|Slow food")
active = ("horses|Travel|trips|Runnning|Bicycle|Dance|Runn|sports|Tennis|Dancing")
unhealthy = ("Fast food|Pizza")

#create a function to terate over
def count_group(variable,name_group,ignore): 
    #takes groups including words in the name_group and ignores if containing ignore
    temp = counts[counts['index'].str.contains(name_group,flags=re.IGNORECASE)&(~counts['index'].str.contains(ignore))]
    group_data[variable] = group_data["group"].isin(temp["index"]) # adds to the group_data True if contained in the temporary variable
    output[variable] = group_data.groupby(['id'])[variable].sum()!=0 # groups the results by each person to return true if one or more of their groups is true
    timetemp = group_data[group_data["group"].isin(temp["index"])] # creates a subgroup of times for groups included in the temporary variable
    timetemp = timetemp.groupby("id")["time_group"].mean() # calculates the average time each person is in a group contained in temp
    time_output = pd.concat([time,timetemp],axis = 1) # adds the average time to the time variable
    return output,time_output

#loop over all the groups with the name, words to include and words to exclude
function_loop = pd.DataFrame([["gym","health","food","active","unhealthy"],[gym,health,food,active,unhealthy],["Gymnasium","Gym","abcxyz","abcxyz","abcxyz"]])
for i in function_loop:
    output,time = count_group(function_loop[i][0],function_loop[i][1],function_loop[i][2])

#adds a group for anybody not apart of the groups I created
nothing = output[["gym","health","food","active","unhealthy"]].sum(axis = 1)
output["no_group"] = nothing == 0 # outputs true if not in a created group

#%% Preparing the output data
#create the correct names of the time variable without the id column
#fill the missing values with -1 as this person is not in this group and convert all to int type
time.columns = ["id",'gym_time', 'health_time',"active_time","food_time","unhealthy_time"]
time.drop("id",axis = 1,inplace = True)
time.fillna(-1,inplace = True)
time = time.astype("int")

output = pd.concat([output,time],axis = 1) # concat the time data to the output
#%% outputting the data
 
#resplit into the test and train data
test_output = output.iloc[0:test_size]
train_output = output.iloc[test_size:len(output)]

#output the data as a csv file
test_output.to_csv("../Case/Refined/groups_test.csv",index = False)
train_output.to_csv("../Case/Refined/groups_train.csv",index = False)