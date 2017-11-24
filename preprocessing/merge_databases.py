import sqlite3
import pandas as pd
import os

data_path = os.path.split(os.getcwd())[0] + "\\data"
cnx = sqlite3.connect(data_path + "\\vacancies00.db")
ndf = pd.read_sql_query("SELECT * FROM hh", cnx)
cnx.close()

names = ["vacancies01.db", "vacancies02.db"]
for n in names:
    cnx = sqlite3.connect(data_path + "\\"+n)
    data = pd.read_sql_query("SELECT * FROM hh", cnx)
    ndf = pd.concat([ndf, data], axis=0, ignore_index=True)
    cnx.close()

ndf = ndf.drop_duplicates()
cnx = sqlite3.connect(data_path + "\\vacancies.db")
cnx.execute("DROP TABLE IF EXISTS hh")
ndf.to_sql(name='hh', con=cnx)
cnx.close()