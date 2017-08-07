import pandas as pd
import os
from collections import Counter
import numpy as np

ud = ["down", "up"]
data_path = os.path.split(os.path.split(os.getcwd())[0])[0] + "//test_train//"
for b in ud:
    train_x = pd.read_csv(data_path+b+"_x_train_normalized.csv")["TextOfVacancy"]
    #print(train_x.iloc[218160])
    train_clusters = pd.read_csv(data_path+b+"_x_train_normalized.csv")["comb"]
    words = pd.read_csv(b+"_words_list.csv")["words"]
    stats = pd.DataFrame(data=np.zeros([len(words), 2*900]), index=words)
    for i in range(train_x.shape[0]):
        print(i)
        lex = train_x.ix[i].split(" ")
        numbers = Counter(lex)
        for w in numbers:
            c = numbers.get(w)
            if w == "nan":
                continue
            if c == 1:
                stats.loc[w].iloc[train_clusters.iloc[i]] += 1
            else:
                stats.loc[w].iloc[900+train_clusters.iloc[i]] += 1
    stats.to_csv(b + "_stats.csv", encoding='utf-8', index=False)
