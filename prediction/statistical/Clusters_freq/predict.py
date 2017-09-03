import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

data_path = os.getcwd().split("\\prediction")[0]+"\\test_train\\"

gl_train_data_x = pd.read_csv(data_path + "down_x_train_normalized.csv")
gl_test_data_x = pd.read_csv(data_path + "down_x_test_normalized.csv")
gl_train_y = pd.read_csv(data_path + "down_y_train_normalized.csv")["down"]
gl_test_y = pd.read_csv(data_path + "down_y_test_normalized.csv")["down"]

all_feats = pd.read_csv("features.csv")
errors = np.zeros([400])
l = 0
el= 0
p = 0
for cl in range(400):
    words = all_feats[str(cl)].dropna()

    text_train_x = gl_train_data_x[gl_train_data_x["main"] == cl]["TextOfVacancy"].tolist()
    text_test_x = gl_test_data_x[gl_test_data_x["main"] == cl]["TextOfVacancy"].tolist()

    ind_train_x = []
    cl_train = gl_train_data_x["main"].tolist()
    for i in range(len(cl_train)):
        if cl_train[i] == cl:
            ind_train_x.append(i)
    train_y = gl_train_y.iloc[ind_train_x].tolist()

    ind_test_x = []
    cl_test = gl_test_data_x["main"].tolist()
    for i in range(len(cl_test)):
        if cl_test[i] == cl:
            ind_test_x.append(i)
    test_y = gl_test_y.iloc[ind_test_x].tolist()

    train_x = np.zeros([len(train_y), len(words)])
    test_x = np.zeros([len(test_y), len(words)])

    d = {words[w]: w for w in range(len(words))}

    words = set(words)
    for i in range(len(train_y)):
        tokens = str(text_train_x[i]).split(" ")
        for j in tokens:
            if j in words:
                train_x[i, d.get(j)] = 1

    train_x = np.append(np.array(gl_train_data_x[gl_train_data_x["main"] == cl]
                                 [["City", "Exp", "EmploymentType", "WorkHours", "job_start"]]).reshape([len(train_y), 5]),
                        train_x, axis=1)
    for i in range(len(test_y)):
        tokens = str(text_test_x[i]).split(" ")
        for j in tokens:
            if j in words:
                test_x[i, d.get(j)] = 1

    test_x = np.append(np.array(gl_test_data_x[gl_test_data_x["main"] == cl]
                                [["City", "Exp", "EmploymentType", "WorkHours", "job_start"]]).reshape([len(test_y), 5]),
                       test_x, axis=1)

    print("start pred")
    clf = RandomForestRegressor(n_estimators= 200)
    clf.fit(train_x, train_y)
    pr = clf.predict(test_x)
    for i in range(len(test_y)):
        if test_y[i] < 200000:
            errors[cl] += (np.abs(pr[i]-test_y[i])/test_y[i])
    el += errors[cl]
    l += len(test_y)
    if errors[cl]/len(test_y) > 3:
        print("##############################")
        print(len(words))
        for i in range(len(test_y)):
            print(pr[i], test_y[i], (pr[i]-test_y[i])/test_y[i])
        p += 1
        el -= errors[cl]
        l -= len(test_y)
        print("##############################")
    print(cl, errors[cl]/len(test_y), len(train_y), len(test_y), np.sum(errors[:cl+1])/l)
print(np.sum(errors)/len(gl_test_y))
print(el/l)
print(p)