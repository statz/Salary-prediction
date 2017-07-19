import sqlite3
import pandas as pd
import numpy as np
import os
import re

def db_read(db_name):
    cnx = sqlite3.connect(db_name)
    df = pd.read_sql_query("SELECT * FROM hh", cnx)
    return df

data_path = os.path.split(os.getcwd())[0]+"\\data\\vacancies.db"
data = db_read(data_path)

data = data.drop_duplicates(["TextOfVacancy"])
number_vacancies = data.shape[0]
indexies = []
for i in range(number_vacancies):
    if data["Salary"].iloc[i].find("0 руб") == -1:
        indexies.append(i)
data = data.drop(data.index[indexies])
number_vacancies = data.shape[0]

salaries = np.zeros([number_vacancies, 2])
for i in range(number_vacancies):
    str = data["Salary"].iloc[i].replace(u'\xa0', u'')
    arr = re.findall(r"\d+", str)
    if str.find("от") != -1:
        if str.find("до") != -1 and len(arr) == 2:
            salaries[i, 0] = int(arr[0])
            salaries[i, 1] = int(arr[1])
        else:
            salaries[i, 0] = int(arr[0])
    else:
        salaries[i, 1] = int(arr[0])
