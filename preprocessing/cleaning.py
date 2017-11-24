import sqlite3
import pandas as pd
import numpy as np
import os
import re

data_path = os.path.split(os.getcwd())[0] + "\\data"
cnx = sqlite3.connect(data_path + "\\vacancies.db")
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
    vacancy = data["Salary"].iloc[i].replace(u'\xa0', u'')
    arr = re.findall(r"\d+", vacancy)
    if vacancy.find("от") != -1:
        if vacancy.find("до") != -1 and len(arr) == 2:
            salaries[i, 0] = int(arr[0])
            salaries[i, 1] = int(arr[1])
        else:
            salaries[i, 0] = int(arr[0])
    else:
        salaries[i, 1] = int(arr[0])
    if vacancy.find("НДФЛ") != -1:
        salaries[i, :] *= 1-0.13
salaries = pd.DataFrame(salaries, columns=["down", "up"])

title = []
for i in range(number_vacancies):
    vacancy = data["TitleOfVacancy"].iloc[i]
    vacancy = vacancy.split('(')[0]
    vacancy = vacancy.lower()
    vacancy = re.sub(r"[^а-я^А-Я^a-z^A-Z^ё^Ё ]", " ", vacancy)
    title.append(vacancy)
title = pd.DataFrame(title, columns=["TitleOfVacancy"])

text = []
for i in range(number_vacancies):
    print(i)
    vacancy = data["TextOfVacancy"].iloc[i]
    vacancy.replace("C++", "cpp")
    vacancy.replace("C#", "csh")
    wr = re.search("[А-Я]*[а-я]+[А-Я]{1,1}[а-я]{1,}", vacancy)
    if wr != None:
        words = vacancy.split(" ")
        ns = []
        for w in words:
            p = re.search("(?P<first>[А-Я]*[а-я]+)(?P<second>[А-Я]{1,1}[а-я]{1,})", w)
            if p != None:
                ns.append(p.groups()[0])
                ns.append(p.groups()[1])
            else:
                ns.append(w)
        vacancy = " ".join(ns)
    vacancy = vacancy.lower()
    vacancy = vacancy.replace("nan", "нан")
    vacancy = re.sub(r"[^а-я^a-z^ё^ ]", " ", vacancy)
    vacancy = re.sub(r" {2,}", " ", vacancy)
    vacancy = re.sub(r"^ ", "", vacancy)
    text.append(vacancy)
text = pd.DataFrame(text, columns=["TextOfVacancy"])

city = []
for i in range(number_vacancies):
    vacancy = data["City"].iloc[i]
    vacancy = re.sub(r"[^а-я^А-Я^A-Za-z^ё^Ё]", "", vacancy)
    vacancy = vacancy.lower()
    city.append(vacancy)
city = pd.DataFrame(city, columns=["City"])

name = []
for i in range(number_vacancies):
    vacancy = data["NameOfCompany"].iloc[i]
    vacancy = re.sub(r"[^а-я^А-Я^A-Za-z^ё^Ё]", "", vacancy)
    vacancy = vacancy.lower()
    name.append(vacancy)
name = pd.DataFrame(name, columns=["NameOfCompany"])


ndf = pd.concat([title, text, city, name, salaries, data[['Exp', 'EmploymentType',
                                                    'WorkHours', 'MainProfAreas', 'SubProfAreas']]], axis=1)
ndf = ndf[(ndf["down"] >= 8000) | (ndf["down"] == 0)]
ndf = ndf[(ndf["up"] >= 8000) | (ndf["up"] == 0)]

cnx = sqlite3.connect(data_path + "\\vacancies1.db")
cnx.execute("DROP TABLE IF EXISTS hh")
ndf.to_sql(name='hh', con=cnx)
cnx.close()
