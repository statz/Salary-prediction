import sqlite3
import pandas as pd
import os
from pymystem3 import Mystem
import re
from subprocess import Popen, PIPE

data_path = os.path.split(os.getcwd())[0]+"\\data"
cnx = sqlite3.connect(data_path+"\\vacancies3.db")
data = pd.read_sql_query("SELECT * FROM hh", cnx)
cnx.close()
labels = ["TitleOfVacancy", "TextOfVacancy"]
for l in labels:
    text = data[l]

    program = os.path.split(os.getcwd())[0]+"\\mystem.exe"
    process = Popen([program, "-l", "-c", "-s", "-", data_path+"\\norm.txt"], stdin=PIPE)
    j = 0
    for i in text:
        print(j)
        j += 1
        process.stdin.write((str(i)+';').encode('utf-8'))
        process.stdin.write('\n'.encode("utf-8"))
        process.stdin.flush()
    process.kill()
    f = open(data_path+"\\norm.txt", encoding="utf-8")
    text = f.read()
    text = text.split(';')
    new_text = []
    for t in text:
        lemmas = re.findall(r"[а-яa-zё|?]+", t)
        list = []
        for l1 in lemmas:
            list.append(re.match(r"[а-яa-zё]+",l1).group(0))
        new_text.append(" ".join(list))
    f.close()
    new_text = pd.DataFrame(new_text)
    data[l] = new_text

comb_text = data["TitleOfVacancy"]+data["TextOfVacancy"]
comb_text =pd.DataFrame(comb_text, columns=["combined_text"])
data = pd.concat([data, comb_text], axis = 1)

cnx = sqlite3.connect(data_path+"\\vacancies3_normalized.db")
cnx.execute("DROP TABLE IF EXISTS hh")
data.to_sql(name='hh', con=cnx, index=False)
cnx.close()