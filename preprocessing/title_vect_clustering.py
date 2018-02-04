from gensim.models import Word2Vec as w2v
import pandas as pd
import os
import sqlite3
import numpy as np
import h5py
import hdbscan
import time
from sklearn.cluster import MiniBatchKMeans
import collections
data_path = os.getcwd().split("\\preprocessing")[0]+"\\data\\"


if not os.path.exists(data_path+"v2w"):
    cnx = sqlite3.connect(data_path + "\\vacancies3.db")
    data = pd.read_sql_query("SELECT * FROM hh", cnx)
    cnx.close()
    vec_dim = 50
    text = list(data["TitleOfVacancy"].astype("str"))
    print(time.clock())
    print("Start training")
    text_words = list(map(str.split, text))
    model = w2v(text_words, min_count=100, size=vec_dim)
    model.save(data_path+"v2w")
print(time.clock())

if not os.path.exists(data_path+"mean_vect.h5"):
    model = w2v.load(data_path+"v2w")
    vocabulary = set(list(model.wv.vocab))
    mean_vect = np.zeros([len(text), vec_dim])
    for i in range(len(text)):
        print(i)
        sent = text[i].split(" ")
        j = 0
        for w in sent:
            if w in vocabulary:
                mean_vect[i, :] += model.wv[w]
                j += 1
        if j:
            mean_vect[i] /= j
            mean_vect[i] /= sum(mean_vect[i][:]**2)**0.5
    h5f = h5py.File(data_path+"mean_vect.h5", 'w')
    h5f.create_dataset('dataset_1', data=mean_vect)
    h5f.close()

h5f = h5py.File(data_path+"mean_vect.h5",'r')
#indexies = np.arange(800000)
#np.random.shuffle(indexies)
#indexies = indexies[:800000]
#indexies = np.sort(indexies)
b = h5f['dataset_1'][:]#[list(indexies)]
h5f.close()
print(time.clock())
#kmean = KMeans(n_clusters=1000)
#kmean.fit_transform(b)

print(1)
clusterer = MiniBatchKMeans(n_clusters=100, batch_size = 500, max_iter = 500)

cluster_labels = clusterer.fit_predict(b)
c = collections.Counter(cluster_labels)
print(c)
cluster_labels = pd.DataFrame(cluster_labels, columns=["cluster"])

cnx = sqlite3.connect(data_path + "\\vacancies3_normalized.db")
data = pd.read_sql_query("SELECT * FROM hh", cnx)
cnx.close()

data = pd.concat([data, cluster_labels], axis=1)
cnx = sqlite3.connect(data_path + "\\vacancies4_normalized.db")
cnx.execute("DROP TABLE IF EXISTS hh")
data.to_sql(name='hh', con=cnx, index=False)
cnx.close()

cnx = sqlite3.connect(data_path + "\\vacancies3.db")
data = pd.read_sql_query("SELECT * FROM hh", cnx)
cnx.close()

data = pd.concat([data, cluster_labels], axis=1)
cnx = sqlite3.connect(data_path + "\\vacancies4.db")
cnx.execute("DROP TABLE IF EXISTS hh")
data.to_sql(name='hh', con=cnx, index=False)
cnx.close()