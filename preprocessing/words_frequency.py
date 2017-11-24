import sqlite3
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
import re
nltk.download('stopwords')
data_path = os.path.split(os.getcwd())[0]+"\\data"
cnx = sqlite3.connect(data_path+"\\vacancies1.db")
data = pd.read_sql_query("SELECT * FROM hh", cnx)
cnx.close()

TextOfVacancy = data["TextOfVacancy"]
stop_words = stopwords.words('russian')+stopwords.words('english')
words = {}
cleaned_text = []
for i in range(TextOfVacancy.shape[0]):
    text = TextOfVacancy.iloc[i]
    tokens = text.split(" ")
    tokens = [t for t in tokens if (t not in stop_words)]
    cleaned_text.append(" ".join(tokens))
data["TextOfVacancy"] = pd.DataFrame(cleaned_text)

TitleOfVacancy = data["TitleOfVacancy"]
cleaned_title = []
for i in range(TitleOfVacancy.shape[0]):
    title = TitleOfVacancy.iloc[i]
    title = re.sub(r" {2,}", " ", title)
    title = re.sub(r"^ ", "", title)
    tokens = title.split(" ")
    tokens = [i for i in tokens if (i not in stop_words)]
    if len(tokens):
        cleaned_title.append(" ".join(tokens))
data["TitleOfVacancy"] = pd.DataFrame(cleaned_title)

cnx = sqlite3.connect(data_path+"\\vacancies2.db")
cnx.execute("DROP TABLE IF EXISTS hh")
data.to_sql(name='hh', con=cnx)
cnx.close()
