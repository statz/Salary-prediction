import sqlite3
import pandas as pd
import os
import re
from subprocess import Popen, PIPE
import pymorphy2

data_path = os.path.split(os.getcwd())[0]+"\\data"
cnx = sqlite3.connect(data_path+"\\vacancies3_normalized.db")
data = pd.read_sql_query("SELECT * FROM hh", cnx)
text = data["combined_text"]
cnx.close()

morph = pymorphy2.MorphAnalyzer()
dict = set()
j = 0
for t in text:
    print(j)
    j+=1
    words = str(t).split(" ")
    for w in words:
        if w not in dict:
            dict.add(w)
dict = list(dict)
dict = {dict[i]: morph.parse(dict[i])[0].normal_form for i in range(len(dict))}
print(dict)
new_text = []
j = 0
for t in text:
    print(j)
    j+=1
    words = str(t).split(" ")
    s = []
    for w in words:
        s.append(dict[w])
    new_text.append(" ".join(s))
new_text = pd.DataFrame(new_text)
data["combined_text"] = new_text
cnx = sqlite3.connect(data_path+"\\vacancies3_normalized2.db")
cnx.execute("DROP TABLE IF EXISTS hh")
data.to_sql(name='hh', con=cnx, index=False)
cnx.close()