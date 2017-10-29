import sqlite3
import pandas as pd
import os
import re
from subprocess import Popen, PIPE
import pymorphy2

data_path = os.path.split(os.getcwd())[0] + "\\data"
cnx = sqlite3.connect(data_path + "\\vacancies3.db")
data = pd.read_sql_query("SELECT * FROM hh", cnx)
cnx.close()
labels = ["TitleOfVacancy", "TextOfVacancy"]
for l in labels:
    print("asdadasdads")
    text = data[l]

    program = os.path.split(os.getcwd())[0] + "\\mystem.exe"
    process = Popen([program, "-l", "-c", "-s", "-", data_path + "\\norm.txt"], stdin=PIPE)
    j = 0
    for i in text:
        print(l, "a", j)
        j += 1
        process.stdin.write((str(i) + ';').encode('utf-8'))
        process.stdin.write('\n'.encode("utf-8"))
        process.stdin.flush()
    process.kill()
    f = open(data_path + "\\norm.txt", encoding="utf-8")
    text = f.read()
    text = text.split(';')
    new_text = []
    j = 0
    for t in text:
        print(l, "b", j)
        j += 1
        lemmas = re.findall(r"[а-яa-zё|?]+", t)
        processed = []
        for l1 in lemmas:
            processed.append(re.match(r"[а-яa-zё]+", l1).group(0))
        new_text.append(" ".join(processed))
    f.close()
    new_text = pd.DataFrame(new_text)
    data[l] = new_text

comb_text = data["TitleOfVacancy"] + " " + data["TextOfVacancy"]
comb_text = pd.DataFrame(comb_text, columns=["combined_text"])
data = pd.concat([data, comb_text], axis=1)

cnx = sqlite3.connect(data_path + "\\vacancies3_normalized.db")
cnx.execute("DROP TABLE IF EXISTS hh")
data.to_sql(name='hh', con=cnx, index=False)
cnx.close()
