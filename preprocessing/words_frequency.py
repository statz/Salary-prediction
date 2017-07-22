import sqlite3
import pandas as pd
import os
from nltk.corpus import stopwords
import re

data_path = os.path.split(os.getcwd())[0]+"\\data\\clean_vacancies.db"
cnx = sqlite3.connect(data_path)
data = pd.read_sql_query("SELECT TextOfVacancy FROM hh", cnx)["TextOfVacancy"]
stop_words = stopwords.words('russian')
dict = {}
for i in range(data.shape[0]):
    str = data.iloc[i]
    str = re.sub(r" {2,}", " ", str)
    str = re.sub(r"^ ", "", str)
    tokens = str.split(" ")
    tokens = [i for i in tokens if (i not in stop_words)]
    for i in tokens:
        if dict.get(i) == None:
            dict.update({i : 1})
        else:
            dict.update({i: dict.get(i)+1})
df = pd.DataFrame([dict])
df = df.transpose()
df.to_csv("words.csv",encoding='utf-8')
