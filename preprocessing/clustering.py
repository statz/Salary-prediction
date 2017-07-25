import sqlite3
import pandas as pd
import os
import re
import numpy as np
from sklearn.cluster import KMeans
import time

data_path = os.path.split(os.getcwd())[0]+"\\data"
cnx = sqlite3.connect(data_path+"\\words_cleaned_vacancies3.db")
data = pd.read_sql_query("SELECT * FROM hh", cnx)
cnx.close()
main_areas = []
MainProfAreas = data["MainProfAreas"]
n = MainProfAreas.shape[0]
for i in range(MainProfAreas.shape[0]):
    area = re.sub(r"[^а-я^А-Я^a-z^A-Z^ё^Ё^;]", "", MainProfAreas.iloc[i])
    tmp = area.split(';')
    for j in tmp:
        if j not in main_areas:
            main_areas.append(j)

enc = np.zeros([n, len(main_areas)])
for i in range(MainProfAreas.shape[0]):
    area = re.sub(r"[^а-я^А-Я^a-z^A-Z^ё^Ё^;]", "", MainProfAreas.iloc[i])
    tmp = area.split(';')
    for j in tmp:
        enc[i, main_areas.index(j)] = 1
main_clusters = KMeans(n_clusters=1000).fit_predict(enc)
print("Done")
sub_areas = []
SubProfAreas = data["SubProfAreas"]
n = SubProfAreas.shape[0]
for i in range(SubProfAreas.shape[0]):
    area = re.sub(r"[^а-я^А-Я^a-z^A-Z^ё^Ё^;]", "", SubProfAreas.iloc[i])
    tmp = area.split(';')
    for j in tmp:
        if j not in sub_areas:
            sub_areas.append(j)

enc = np.zeros([n, len(sub_areas)])
for i in range(SubProfAreas.shape[0]):
    area = re.sub(r"[^а-я^А-Я^a-z^A-Z^ё^Ё^;]", "", SubProfAreas.iloc[i])
    tmp = area.split(';')
    for j in tmp:
        enc[i, sub_areas.index(j)] = 1
sub_clusters = KMeans(n_clusters=2000).fit_predict(enc)
print("Done")
nda = np.stack([main_clusters, sub_clusters], axis=1)
comb_cluster = KMeans(n_clusters=800).fit_predict(nda)

ndf = pd.DataFrame(np.stack([main_clusters, sub_clusters, comb_cluster], axis=1), columns=["main", "sub", "comb"])
ndf.to_csv(data_path+"\\clusters.csv")