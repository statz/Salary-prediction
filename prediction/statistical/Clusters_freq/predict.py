import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb


data_path = os.getcwd().split("\\prediction")[0]+"\\test_train\\"

train_data_x = pd.read_csv(data_path+"down_x_train_normalized.csv")
test_data_x = pd.read_csv(data_path+"down_x_test_normalized.csv")


text_train_x = train_data_x["TextOfVacancy"].tolist()
train_y = pd.read_csv(data_path+"down_y_train_normalized.csv")["down"].tolist()
text_test_x = test_data_x["TextOfVacancy"].tolist()
test_y = pd.read_csv(data_path+"down_y_test_normalized.csv")["down"].tolist()
words = pd.read_csv("var.csv")["words"].tolist()

train_x = np.zeros([len(train_y), len(words)])
test_x = np.zeros([len(test_y), len(words)])

d = {words[w]: w for w in range(len(words))}

words = set(words)
for i in range(len(train_y)):
    #print(i)
    tokens = text_train_x[i].split(" ")
    for j in tokens:
        if j in words:
            train_x[i, d.get(j)] = 1
    
train_x = np.append(np.array(train_data_x[["City", "Exp", "EmploymentType", "WorkHours"]]).reshape([len(train_y), 4]), train_x, axis=1)
for i in range(len(test_y)):
    #print(i)
    tokens = text_test_x[i].split(" ")
    for j in tokens:
        if j in words:
            test_x[i, d.get(j)] = 1

test_x = np.append(np.array(test_data_x[["City", "Exp", "EmploymentType", "WorkHours"]]).reshape([len(test_y), 4]), test_x, axis=1)


from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor

print("start pred")
clf = DecisionTreeRegressor(max_depth=25)
clf.fit(train_x, train_y)
pr = clf.predict(test_x)
err = np.sum(np.abs(pr[:]-test_y[:])/test_y[:])/len(test_y)
print(err)


