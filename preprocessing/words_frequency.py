import sqlite3
import pandas as pd
import os
from nltk.corpus import stopwords
import re

data_path = os.path.split(os.getcwd())[0]+"\\data"
cnx = sqlite3.connect(data_path+"\\vacancies1.db")
data = pd.read_sql_query("SELECT * FROM hh", cnx)
cnx.close()

TextOfVacancy = data["TextOfVacancy"]
stop_words = stopwords.words('russian')
words = {}
cleaned_text = []
for i in range(TextOfVacancy.shape[0]):
    text = TextOfVacancy.iloc[i]
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"^ ", "", text)
    tokens = text.split(" ")
    tokens = [i for i in tokens if (i not in stop_words)]
    cleaned_text.append(" ".join(tokens))
    for j in tokens:
        if words.get(j) is None:
            words.update({j: 1})
        else:
            words.update({j: words.get(j) + 1})
data["TextOfVacancy"] = pd.DataFrame(cleaned_text)

df = pd.DataFrame([words])
df = df.transpose()
df.to_csv(data_path+"\\words_frequency.csv",encoding='utf-8')

TitleOfVacancy = data["TitleOfVacancy"]
cleaned_title = []
for i in range(TitleOfVacancy.shape[0]):
    title = TitleOfVacancy.iloc[i]
    title = re.sub(r" {2,}", " ", title)
    title = re.sub(r"^ ", "", title)
    tokens = title.split(" ")
    tokens = [i for i in tokens if (i not in stop_words)]
    cleaned_title.append(" ".join(tokens))
data["TitleOfVacancy"] = pd.DataFrame(cleaned_title)

cnx = sqlite3.connect(data_path+"\\vacancies2.db")
cnx.execute("DROP TABLE IF EXISTS hh")
data.to_sql(name='hh', con=cnx)
cnx.close()