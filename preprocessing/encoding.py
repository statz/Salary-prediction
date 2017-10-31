import sqlite3
import pandas as pd
import os
import numpy as np

data_path = os.path.split(os.getcwd())[0] + "\\data"
cnx = sqlite3.connect(data_path + "\\vacancies2.db")
data = pd.read_sql_query("SELECT * FROM hh", cnx)
cnx.close()
labels = ["City", "Exp", "EmploymentType", "WorkHours"]
enc = np.empty([data.shape[0], len(labels)+2])
k = 0
for l in labels:
    uniq = []
    column = data[l]
    j = 0
    for i in column:
        if i not in uniq:
            uniq.append(i)
            enc[j, k] = (len(uniq) - 1)
        else:
            enc[j, k] = uniq.index(i)
        j += 1
    k += 1

city = list(data[l])
for i in range(len(city)):
    if city[i] == "москва":
        enc[i, -2] = 1
    if city[i] == "санктпетербург":
        enc[i, -1] = 1
enc = pd.DataFrame(enc, columns=(labels+["москва", "спб"]))
ndf = pd.concat([enc, data[["TextOfVacancy", "TitleOfVacancy", "down", "up"]]], axis=1)
cl = pd.read_csv(data_path + "\\clusters.csv", encoding="utf-8", sep=",")
ndf = pd.concat([ndf, cl], axis=1)

cnx = sqlite3.connect(data_path + "\\vacancies3.db")
cnx.execute("DROP TABLE IF EXISTS hh")
ndf.to_sql(name='hh', con=cnx, index=False)
cnx.close()
