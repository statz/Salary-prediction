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


if not os.path.exists(data_path+"v2w"):
    cnx = sqlite3.connect(data_path + "\\vacancies3_normalized2.db")
    data = pd.read_sql_query("SELECT * FROM hh", cnx)
    cnx.close()
    vec_dim = 200
    text = data["combined_text"]
    print(time.clock())
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
indexies = np.arange(100000)
np.random.shuffle(indexies)
indexies = indexies[:2000]
indexies = np.sort(indexies)
b = h5f['dataset_1'][list(indexies)]
h5f.close()

print(time.clock())
#kmean = KMeans(n_clusters=1000)
#kmean.fit_transform(b)
#clusterer = hdbscan.HDBSCAN(min_cluster_size=50)
#cluster_labels = clusterer.fit_predict(tsne)
from sklearn.manifold import TSNE
from sklearn.decomposition import IncrementalPCA
from matplotlib import pyplot as ppt
pca = IncrementalPCA(n_components=50).fit_transform(b)
tsne = TSNE(n_components=3).fit_transform(pca)
kmean = KMeans(n_clusters=12)
cluster_labels = kmean.fit_predict(tsne)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tsne[:, 0],tsne[:, 1], tsne[:, 2], c=cluster_labels)
plt.show()
print(time.clock())