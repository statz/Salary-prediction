import sqlite3
import pandas as pd
import os
import re
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

data_path = os.path.split(os.getcwd())[0]+"\\data"
cnx = sqlite3.connect(data_path+"\\vacancies2.db")
data = pd.read_sql_query("SELECT * FROM hh", cnx)
cnx.close()
main_areas = ["Началокарьерыстуденты"]
MainProfAreas = data["MainProfAreas"]
n = MainProfAreas.shape[0]
for i in range(MainProfAreas.shape[0]):
    area = re.sub(r"[^а-я^А-Я^a-z^A-Z^ё^Ё^;]", "", MainProfAreas.iloc[i])
    tmp = area.split(';')
    for j in tmp:
        if j not in main_areas:
            main_areas.append(j)
print("aa")
d = {main_areas[w]: w for w in range(len(main_areas))}
enc = np.zeros([n, len(main_areas)])
for i in range(MainProfAreas.shape[0]):
    area = re.sub(r"[^а-я^А-Я^a-z^A-Z^ё^Ё^;]", "", MainProfAreas.iloc[i])
    tmp = area.split(';')
    for j in tmp:
        enc[i, d.get(j)] = 1
job_start = enc[:, 0]
enc = enc[:, 1:]
main_clusters = KMeans(n_clusters=60).fit_predict(enc)
print("Done")

ndf = pd.DataFrame(np.stack([job_start, main_clusters], axis=1), columns=["job_start", "main"])
ndf.to_csv(data_path+"\\clusters.csv")

#sub_areas = []
#SubProfAreas = data["SubProfAreas"]
#n = SubProfAreas.shape[0]
#for i in range(SubProfAreas.shape[0]):
#    area = re.sub(r"[^а-я^А-Я^a-z^A-Z^ё^Ё^;]", "", SubProfAreas.iloc[i])
#    tmp = area.split(';')
#    for j in tmp:
#        if j not in sub_areas:
#            sub_areas.append(j)
#print(len(sub_areas))
#
#d = {sub_areas[w]: w for w in range(len(sub_areas))}
#enc = np.zeros([n, len(sub_areas)])
#for i in range(SubProfAreas.shape[0]):
#    area = re.sub(r"[^а-я^А-Я^a-z^A-Z^ё^Ё^;]", "", SubProfAreas.iloc[i])
#    tmp = area.split(';')
#    for j in tmp:
#        enc[i, d.get(j)] = 1
#sub_clusters = KMeans(n_clusters=100).fit_predict(enc)
#print("Done")
#print(Counter(sub_clusters))
#
#
#nda = np.stack([main_clusters, sub_clusters], axis=1)
#comb_cluster = KMeans(n_clusters=75).fit_predict(nda)
#print(Counter(comb_cluster))
#

#ndf = pd.DataFrame(np.stack([main_clusters, sub_clusters, comb_cluster], axis=1), columns=["main", "sub", "comb"])
#ndf.to_csv(data_path+"\\clusters.csv")