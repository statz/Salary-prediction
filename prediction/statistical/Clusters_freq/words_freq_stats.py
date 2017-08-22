import pandas as pd
import os
from collections import Counter
import numpy as np

ud = ["down", "up"]
data_path = os.path.split(os.path.split(os.getcwd())[0])[0] + "//test_train//"
for b in ud:
    train_x = pd.read_csv(data_path+b+"_x_train_normalized.csv")["TextOfVacancy"]
    train_clusters = pd.read_csv(data_path+b+"_x_train_normalized.csv")["comb"]
    words = pd.read_csv(b+"_words_list.csv")["words"]
    d = {words.iloc[w] : w for w in range(words.shape[0])}
    stats = np.zeros([len(words), 2 * 900])
    for i in range(train_x.shape[0]):
        print(i)
        lex = train_x.iloc[i].split(" ")
        numbers = Counter(lex).items()
        for w in numbers:
            if w[1] == 1:
                stats[d.get(w[0]), train_clusters.iloc[i]] += 1
            else:
                stats[d.get(w[0]), 900+train_clusters.iloc[i]] += 1
    stats = pd.DataFrame(stats)
    stats = pd.concat([words, stats], axis=1)
    stats.to_csv(b + "_stats.csv", encoding='utf-8', index=False)
