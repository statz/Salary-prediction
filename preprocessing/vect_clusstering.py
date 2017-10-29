from gensim.models import Word2Vec as w2v
import pandas as pd
import os
import sqlite3
import numpy as np
import h5py
import hdbscan
import time
from sklearn.cluster import KMeans
data_path = os.getcwd().split("\\preprocessing")[0]+"\\data\\"

cnx = sqlite3.connect(data_path + "\\vacancies3_normalized2.db")
data = pd.read_sql_query("SELECT * FROM hh", cnx)
cnx.close()
vec_dim = 200
text = data["combined_text"]
print(time.clock())
if not os.path.exists(data_path+"v2w"):
    print("Start training")
    text_words = list(map(str.split, text))
    model = w2v(text_words, min_count=30, size=vec_dim)
    model.save(data_path+"v2w")
print(time.clock())

if not os.path.exists(data_path+"mean_vect.h5"):
    model = w2v.load(data_path+"v2w")
    vocabulary = set(list(model.wv.vocab))
    mean_vect = np.empty([len(text), vec_dim])
    for i in range(len(text)):
        print(i)
        sent = text.iloc[i].split(" ")
        j = 0
        for w in sent:
            if w in vocabulary:
                mean_vect[i, :] += model.wv[w]
                j += 1
        mean_vect[i] /= j
    h5f = h5py.File(data_path+"mean_vect.h5", 'w')
    h5f.create_dataset('dataset_1', data=mean_vect)
    h5f.close()

h5f = h5py.File(data_path+"mean_vect.h5",'r')
b = h5f['dataset_1'][:100000]
h5f.close()

print(time.clock())
#clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
#cluster_labels = clusterer.fit_predict(b)
kmean = KMeans(n_clusters=1000)
kmean.fit_transform(b)
print(time.clock())