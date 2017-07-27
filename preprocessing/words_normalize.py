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

text = data["TextOfVacancy"]

program = os.path.split(os.getcwd())[0]+"\\mystem.exe"
process = Popen([program, "-l", "-c", "-s", "-", data_path+"\\norm.txt"], stdin=PIPE)
for i in text:
    process.stdin.write(str(i+";").encode('utf-8'))
    process.stdin.write('\n'.encode("utf-8"))
    process.stdin.flush()

f = open(data_path+"\\norm.txt", encoding="utf-8")
text = f.read()
text = text.split(';')
new_text = []
for str in text:
    lemmas = re.findall(r"[а-яa-zё|?]+", str)
    list = []
    for l in lemmas:
        list.append(re.match(r"[а-яa-zё]+",l).group(0))
    new_text.append(" ".join(list))
f.close()

new_text = pd.DataFrame(new_text)
data["TextOfVacancy"] = new_text
cnx = sqlite3.connect(data_path+"\\vacancies3_normalized.db")
cnx.execute("DROP TABLE IF EXISTS hh")
data.to_sql(name='hh', con=cnx, index=False)
cnx.close()