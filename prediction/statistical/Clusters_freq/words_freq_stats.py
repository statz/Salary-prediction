import pandas as pd
import os
from collections import Counter
import numpy as np

ud = ["down", "up"]
data_path = os.getcwd().split("\\prediction")[0]+"\\test_train\\"
for b in ud:
    train_x = pd.read_csv(data_path+b+"_x_train_normalized.csv")["combined_text"]
    train_clusters = pd.read_csv(data_path+b+"_x_train_normalized.csv")["main"].as_matrix()
    words = pd.read_csv(os.getcwd()+"\\data\\" + b+"_words_list.csv")["all"].dropna()
    d = {words.iloc[w]: w for w in range(words.shape[0])}
    stats = np.zeros([len(words), 400])
    for i in range(train_x.shape[0]):
        print(i)
        lex = str(train_x.iloc[i]).split(" ")
        for w in lex:
            stats[d.get(w), int(train_clusters[i])] += 1
    stats = pd.DataFrame(stats)
    stats = pd.concat([words, stats], axis=1)
    stats.to_csv(os.getcwd()+"\\data\\"+b + "_stats.csv", encoding='utf-8', index=False)
