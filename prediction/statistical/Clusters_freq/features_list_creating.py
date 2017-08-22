import pandas as pd
import os
from collections import Counter
import numpy as np

data_path = os.path.split(os.path.split(os.getcwd())[0])[0] + "//test_train//"
TextOfVacancy = pd.read_csv(data_path+"down_x_train_normalized.csv")["TextOfVacancy"].tolist()
words0 = {}
for text in TextOfVacancy:
    tokens = text.split(" ")
    for j in tokens:
        if words0.get(j) is None:
            words0.update({j: 1})
        else:
            words0.update({j: words0.get(j) + 1})


data = pd.read_csv("down_stats.csv")
words = data["words"].tolist()
stats = data.drop(["words"], axis=1).as_matrix()
val = np.zeros([len(words)])
for i in range(len(words)):
    print(i)
    if words0.get(words[i]) < 50000:
        val[i] = np.sum(stats[i, 0:900]/(stats[i, 900:1800]+1))
n = 1000
ind = np.argpartition(val, -n)[-n:]
words = pd.DataFrame(words).iloc[ind].to_csv("var.csv", index=False, encoding="utf-8")
