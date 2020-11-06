# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 2020

@author: Ethan Bosworth

A script to import raw data with the group data from before and combine them then to
transform and investigate the data that it can be used in classification in the next script
"""
#%% import modules
import pandas as pd 
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn")

#%% import the data
data_test = pd.read_csv("../Case/Input/test.csv",index_col = 0)
data_train = pd.read_csv("../Case/Input/train.csv",index_col = 0)
groups_test = pd.read_csv("../Case/Refined/groups_test.csv",index_col = 0)
groups_train = pd.read_csv("../Case/Refined/groups_train.csv",index_col = 0)

output_train = pd.DataFrame(data_train["target"]) # create output variables so original data
output_test = pd.DataFrame(data_test.index) # remains unchanged


#%% transforming the data


#print a dataframe showing the missing data in each variable for both train and test data and the type of the variable
#and create a variable to refer to moving through
missing = pd.concat([data_train.isna().sum(),data_test.isna().sum(),data_train.dtypes],axis = 1,keys = ["Train","Test","Type"])
print(missing)

#%%% Name and sex data
# name data is useless for the task however it can be useful for finding the missing sex data
# in addition the data consists of Polish names where womens names mainly end with "a" 


#create a function for guessing the sex based on the name
def names_to_sex(input_data):

    sex = input_data[["name","sex"]].copy() # create a copy of the data
    sex["end_a"] = pd.DataFrame(sex["name"].str.endswith("a").fillna(False)) # find a boolean list of all names beginning with a
    print("\n" +  str(sex[sex["end_a"]].sex.value_counts())) # checks all are female

    sex = sex.fillna("missing") 

    #keeps all cases the same except for when the sex is missing and the name ends with "a" 
    #and returns female if this is the case
    sex["sex"].where(((sex.sex != "missing") | (sex.end_a != True)),"female",inplace = True)
    sex = sex.drop("end_a",axis = 1) # drops the now not needed column

    #creates a list of unique names with the sex 
    names = sex[sex["sex"] != "missing"].drop_duplicates("name")
    
    #for loop to find the sex based on if somebody else has the same name with a sex
    for i in range(len(sex)): # runs over all of the names
        if sex["sex"][i] == "missing": # checks first to only do something if sex is missing
            if sex["name"][i] != "missing": # checks second if name is missing
                if names.name.str.contains(sex["name"][i]).any() == True: # checks third if there is the name anywhere in the list of names
                    sex["sex"][i] = names[names["name"] == sex["name"][i]]["sex"].iloc[0]
                    # if all are true then finds the sex attached to the name in the names dataframe
                    #and puts it in for the mising sex value
    return sex
sex_train = names_to_sex(data_train) # runs the function for both train and test data
sex_test = names_to_sex(data_test)

#adds the transformed data to the output
output_train["sex"] = sex_train["sex"]
output_test["sex"] = sex_test["sex"]

#check to see the impact of each value on the target (men seem to be about double than women)
info = output_train[["target","sex"]].groupby("sex").mean()
sns.barplot(info.index,info["target"] )
plt.show()

del sex_train,sex_test # clean up variables not used more

#%%% date of birth
#want to convert the dob to an age in years to impute the mean if possible

def to_years(data): # a simple function converting the dob to age
    dob = pd.to_datetime(data["dob"]) # creates a dob variable in datetime for modifying
    dob = pd.to_datetime("now")-pd.to_datetime(dob) # finds time between now and dob
    dob = dob//np.timedelta64(1,'Y')   # converts time to years
    return dob
dob = to_years(data_train)
print("\n")
print(dob.describe()) # prints data information to check if imputing with mean is fine
dob = dob.fillna(round(dob.mean()))
print(dob.describe()) # check how much the data was affected by the imputation
# reveals not much change in the values

dob_test = to_years(data_test) # applies the mean of the train data dob to the missing test data
dob_test = dob_test.fillna(round(dob.mean())) # this will lead to a lower accuracy but will avoid overfitting in future

output_train["dob"] = dob # puts the data into the output dataframes
output_test["dob"] = dob_test

#check impact of age on target 
info = output_train[["target","dob"]].groupby("dob").mean()
sns.lineplot(info.index,info["target"])
plt.show()

del dob,dob_test # delete variables now not used

#%%% location data
#with the location data I want to try to simplify the data to avoid overfitting


#%%%% transfomations

loc_columns = ["location","location_population","location_from","location_from_population"] #creates location columns
loc_data = data_train[loc_columns].copy() # sets up the location data with the columsn from before
loc_test = data_test[loc_columns].copy()

def migration(data):
    #at first check to see if the person changed city
    data["new_city"] =data["location"] != data["location_from"] 
    #check how much the population difference between city from and city
    data["pop_change_10,000"] = data["location_population"] - data["location_from_population"]
    data["pop_change_10,000"] = round(data["pop_change_10,000"]/10000) # convert to change in 10,000s
    return data

loc_data = migration(loc_data)
loc_test = migration(loc_test)

print(loc_data["location"].value_counts()) # a check of how many diffent locations and top locations
# as there is 645 different locations need to group places
#the top 10 locations account for about 1/4 of the data and afterwards drops off to under 50 people per location so will  use this
print(loc_data["location"].value_counts().head(10).sum()) 
top_loc = loc_data["location"].value_counts().head(10).index # creates a list of top locations
#changes data to be "other" if not from a top location
loc_data["location"] = loc_data["location"].where(loc_data["location"].isin(top_loc),"Other")
#loc test data transformed with top train data to avoid overfitting on new datahowever will reduce accuracy on this data
loc_test["location"] = loc_test["location"].where(loc_test["location"].isin(top_loc),"Other")


#puts useful columns into the output data
output_train[["location","location_population","new_city","pop_change_10,000"]] = loc_data[["location","location_population","new_city","pop_change_10,000"]]
output_test[["location","location_population","new_city","pop_change_10,000"]] = loc_test[["location","location_population","new_city","pop_change_10,000"]]

#regplot shows how target varies with population(regplot as many locations with only a few or one person)
info = output_train[["target","location_population"]].groupby("location_population").mean()
sns.regplot(info.index,info["target"])
plt.show()

info = output_train[["target","location"]].groupby("location").mean()
sns.barplot(info.index,info["target"])
plt.show()

info = output_train[["target","new_city"]].groupby("new_city").mean()
sns.barplot(info.index,info["target"])
plt.show()
# the pop change data is quite messy and a regplot shows some relationship but not a very clear one
info = output_train[["target","pop_change_10,000"]].groupby("pop_change_10,000").mean()
sns.regplot(info.index,info["target"])
plt.show()

del loc_data,loc_test,loc_columns,top_loc
#%%% occupation 

#check the occupation data
print(data_train["occupation"].value_counts())
# as occupation data seems to be in categories already and not too many then 
# although it is possble to make fewer groups it is not needed
output_train["occupation"] = data_train["occupation"]
output_test["occupation"] = data_test["occupation"]

#barplot shows quite a lot of variance between jobs
info = output_train[["target","occupation"]].groupby("occupation").mean()
sns.barplot(info.index,info["target"])
plt.show()
#print the occupations that impact the target the most
print(info.sort_values("target").tail())
print(info.sort_values("target").head())

#%%% hobbies
#hobbies seems to be a list from a premade list and so I want to expand the list to get
#individual elements and maybe group them if needed

def hobby_expand(data):
    hobbies_id = pd.DataFrame(data.index) # create a dataframe to hold the boolean values
    hobbies = data['hobbies'].str.split(',',expand=True) # create a hobbies dataframe to work with

    #create a list of hobbies to visualise top hobbies
    hobbies_list = pd.concat([hobbies[0],hobbies[1],hobbies[2],hobbies[3],hobbies[4],hobbies[5]]).dropna().value_counts().reset_index()

    # change dataframe to be a list to work with
    hobbies_id_list = pd.concat([hobbies[0],hobbies[1],hobbies[2],hobbies[3],hobbies[4],hobbies[5]]).dropna().reset_index()
    hobbies_id_list.columns = ["user_id","hobby"]
    return hobbies_id,hobbies_id_list,hobbies_list

hobbies_id,hobbies_id_list,hobbies_list = hobby_expand(data_train)
hobbies_id_test,hobbies_id_list_test,hobbies_list_test = hobby_expand(data_test)


#create a list of sports hobbies
hobbies_sport = hobbies_list[hobbies_list["index"].str.contains("Parkour|cycling|horse|ball|tennis|skating|surf|water|sport|dance|jiu|tai chi|mountain|rock|rapp|orient|jog|archery|rugby|taek|scuba|hunting|skii|board|kayak|run|swim",flags=re.IGNORECASE)&(~hobbies_list['index'].str.contains("Web|Motor|Ghost",flags=re.IGNORECASE))]

print(hobbies_list.head(13)) # there are 174 hobbies but after 13 hobbies it appears to drop of fast
hobbies_top = hobbies_list[0:12] # create a list of the top 13 hobbies

def hobby_bool(hobby_id,hobby_id_list,hobby_list):
    #create two copies of the hobby id list to modify
    x = hobby_id_list.copy()
    y = hobby_id_list.copy()
    #create a loop to check if a person is in the top hobbies
    for i in hobbies_top["index"]:
        temp = hobby_id_list[hobby_id_list["hobby"]== i] 
        hobby_id[i] = hobby_id["user_id"].isin(temp["user_id"])
        x = x[x["hobby"] != i] # shrink the copy x of the hobby id list to not include those already counted
    #check if person is in a sports hobby
    hobby_id_list = x # set the hobby_id_list to x to avoid double counting a sport as one of those in the top 13
    temp = hobby_id_list[hobby_id_list["hobby"].isin(hobbies_sport["index"])]
    hobby_id["sport"] = hobby_id["user_id"].isin(temp["user_id"])

    #count number of hobbies person has
    hobby_id["count"] = y.groupby("user_id").count()
    hobby_id["count"] = hobby_id["count"].fillna(0)
    
    #count number of hobbies a person is a part of outside of the already counted ones
    hobby_id["non_sport_hobbies"] = hobby_id_list.groupby("user_id").count()
    hobby_id["non_sport_hobbies"] = hobby_id["count"].fillna(0)
    
    return hobby_id
# run the function for both train and test data
hobby_id = hobby_bool(hobbies_id,hobbies_id_list,hobbies_list)
hobby_id_test = hobby_bool(hobbies_id_test,hobbies_id_list_test,hobbies_list_test)

#send the data to the output variable and remove the now excess hobbies variable in the output
if "Squash" not in output_train.columns: # if loop stops a problem of output growing infinitely during testing
    output_train = pd.concat([output_train,hobby_id],axis = 1).drop("user_id",axis = 1)
    output_test = pd.concat([output_test,hobby_id_test],axis = 1).drop("user_id",axis = 1)

#looking at count it seems a small number of people with  hobbies significantly skew the data
info = output_train[["target","count"]].groupby("count").mean()
sns.regplot(info.index,info["target"])
plt.show()
#non sports hobbies shows that the more sports people have the lower the percentage roughly
info = output_train[["target","non_sport_hobbies"]].groupby("non_sport_hobbies").mean()
sns.regplot(info.index,info["target"])
plt.show()

###reminder to put here code for a plot of the top hobbies

del hobbies_id,hobbies_id_list,hobbies_list,hobbies_id_test,hobbies_id_list_test,hobbies_list_test,hobbies_sport,hobbies_top,hobby_id,hobby_id_test


#%%% daily commute 
#similar to dob a I want here to simply check the range of values and possibly impute with the mean

print(data_train["daily_commute"].describe())
output_train["daily_commute"] = data_train["daily_commute"].fillna(data_train["daily_commute"].mean())
print(output_train["daily_commute"].describe())
#looking at these number the change is very minor and so is fine to do this way

#apply the mean of the train data to the test output again reducing accuracy for this data but also reducing overfitting
output_test["daily_commute"] = data_test["daily_commute"].fillna(data_train["daily_commute"].mean())

#looking at the data the daily commute seems to show no information at all for finding the target and so should be ignored
info = output_train[["target","daily_commute"]].groupby("daily_commute").mean()
sns.regplot(info.index,info["target"])
plt.show()
#drop the daily commute data as it gives no more information
output_train.drop("daily_commute",inplace = True,axis = 1)
output_test.drop("daily_commute",inplace = True,axis = 1)

#%%% friends number
# nothing is missing or needs to be transoformed here so just a quick inspection
print(data_train["friends_number"].describe())
print(data_test["friends_number"].describe())
#numbers seem to make sense and so data is output without transformation

output_train["friends_number"] = data_train["friends_number"]
output_test["friends_number"] = data_test["friends_number"]


#graph appears to show a parabolic shape between two values and so need binning
info = output_train[["target","friends_number"]].groupby("friends_number").mean()
sns.regplot(info.index,info["target"])
plt.show()

# bin the friends number into 12 seperate bins

#set the maximum value for friends to 1000 will not change the bins significantly but will future proof it to the test data
bins = output_train["friends_number"]
bins = bins.sort_values()
bins.iloc[-1] = 1000
bins = bins.sort_index()

#create bins based on the train data
bins = pd.qcut(bins,q = 12,retbins = True)
output_train["friends_number"] = bins[0] # apply bins to train data and convert to numeric

#graph of the bins shows a much better set of data for use 
info = output_train[["target","friends_number"]].groupby("friends_number").mean()
sns.barplot(info.index,info["target"])
plt.show()

#apply bins from taining data to test data
output_test["friends_number"] = pd.cut(output_test["friends_number"], bins=bins[1])




#%%% relationship status

print(data_train["relationship_status"].value_counts())
print(data_test["relationship_status"].value_counts())
#data seems to be fine and with no indication of why relationship status could be missing
#it will be replaced by simply "missing"
output_train["relationship_status"] = data_train["relationship_status"].fillna("Missing")
output_test["relationship_status"] = data_test["relationship_status"].fillna("Missing")

#graph of the data seems that all is good with the data
info = output_train[["target","relationship_status"]].groupby("relationship_status").mean()
sns.barplot(info.index,info["target"])
plt.show()


#%%%  education

print(data_train["education"].value_counts())
print(data_test["education"].value_counts())
#data appears to be without problems however it is not possible to find a good imputaiton for the missing data
#as a result it will be filled with -1

print(data_train[["target","education"]].fillna("-1").groupby("education").mean())

output_train["education"] = data_train["education"].fillna(-1)
output_test["education"] = data_test["education"].fillna(-1)

info = output_train[["target","education"]].groupby("education").mean()
sns.barplot(info.index,info["target"])
plt.show()

#%%% credit card

print(data_train["credit_card_type"].value_counts())
print(data_test["credit_card_type"].value_counts())

output_train["cc_type"] = data_train["credit_card_type"].fillna("Missing")
output_test["cc_type"] = data_test["credit_card_type"].fillna("Missing")

#explore how much difference between the types
#what is seen is little difference between card types however missing is significantly lower and so should be own category

info = output_train[["target","cc_type"]].groupby("cc_type").mean()
sns.barplot(info.index,info["target"])
plt.show()
 


#%% exploring the data
#firstly combine the data with thr group data from before 

if "gym" not in output_train.columns: # if statement prevents infinite concat when rerunning this part of code
    output_train = pd.concat([output_train,groups_train],axis = 1)
    output_test = pd.concat([output_test,groups_test],axis = 1)

#quickly check for any missing values and the data types

print(output_test.dtypes)
print(pd.concat([output_train.isna().sum(),output_test.isna().sum(),output_train.dtypes],axis = 1,keys = ["Train","Test","Type"]))


#need to convert any floats to ints and objects to categorical
def type_convert(data):
    for i in data:
        if data[i].dtypes == "float64":
           data[i] = data[i].astype("int")
        if data[i].dtypes == "object":
            data[i] = data[i].astype("category")
    return data

#some boolean values got turned into categorical data and so need to fix
boolean = ["gym","health","food","active","unhealthy"]   
output_train[boolean] = output_train[boolean].astype("bool")         
output_test[boolean] = output_test[boolean].astype("bool") 

output_train = type_convert(output_train)
output_test = type_convert(output_test)
# check to make sure all types got changes correctly
print(output_train.dtypes)
# create a correlation matrix to check the data correlation
corr = output_train.corr()
plt.matshow(corr)
plt.show()
#mostly the data is not highly correlated except in a few places




#%% output the data

#convert output data to dummies to better work with a machine learning model
dummies_train = pd.get_dummies(output_train)
dummies_test = pd.get_dummies(output_test)
#output the transformed data to a csv
dummies_test.to_csv("../Case/Refined/test.csv",index = False)
dummies_train.to_csv("../Case/Refined/train.csv",index = False)