import sqlite3
import pandas as pd
import numpy as np
import os
import re

data_path = os.path.split(os.getcwd())[0]+"\\data"
cnx = sqlite3.connect(data_path+"\\vacancies.db")
data = pd.read_sql_query("SELECT * FROM hh", cnx)
cnx.close()

data = data.drop_duplicates(["TextOfVacancy"])
number_vacancies = data.shape[0]
indexies = []
for i in range(number_vacancies):
    if data["Salary"].iloc[i].find("0 руб") == -1:
        indexies.append(i)
data = data.drop(data.index[indexies]).reset_index()
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
salaries = pd.DataFrame(salaries, columns=["down", "up"])

title = []
for i in range(number_vacancies):
    str = data["TitleOfVacancy"].iloc[i]
    str = str.split('(')[0]
    str = re.sub(r"[^а-я^А-Я^a-z^A-Z^ё^Ё ]", " ", str)
    title.append(str)
title = pd.DataFrame(title, columns=["TitleOfVacancy"])

text = []
for i in range(number_vacancies):
    str = data["TextOfVacancy"].iloc[i]
    str = re.sub(r"[^а-я^А-Я^a-z^A-Z^ё^Ё^ ]", " ", str)
    str = str.lower()
    text.append(str)
text = pd.DataFrame(text, columns=["TextOfVacancy"])


city = []
for i in range(number_vacancies):
    str = data["City"].iloc[i]
    str = re.sub(r"[^а-я^А-Я]", "", str)
    str = str.lower()
    city.append(str)
city = pd.DataFrame(city, columns=["City"])

ndf = pd.concat([title, text, city, salaries, data[['NameOfCompany', 'Exp', 'EmploymentType', 'WorkHours', 'MainProfAreas','SubProfAreas']]], axis=1)

cnx = sqlite3.connect(data_path+"\\clean_vacancies.db")
cnx.execute("DROP TABLE IF EXISTS hh")
ndf.to_sql(name='hh', con=cnx)
cnx.close()

