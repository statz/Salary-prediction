import pandas as pd
import os
from collections import Counter
import numpy as np

data_path = os.getcwd().split("\\prediction")[0]+"\\test_train\\"
ud = ["down", "up"]
for l in ud:
    data = pd.read_csv(data_path+l+"_x_train_normalized.csv")[["combined_text", "main"]]

    tmp = pd.read_csv(os.getcwd()+"\\data\\"+l+"_stats.csv")
    stats = tmp.drop(["all"], axis=1).as_matrix()
    words_all = tmp["all"].tolist()
    d = {words_all[w]: w for w in range(len(words_all))}

    words_by_clasters = pd.read_csv(os.getcwd()+"\\data\\"+l+"_words_list.csv")
    n = 1000
    df = pd.DataFrame()
    for i in range(400):
        print(i)
        text_of_vacancy = data[data["main"] == i]["combined_text"]
        words = words_by_clasters[str(i)].dropna()
        val = np.empty([len(words)])
        j = 0
        for w in words:
            ind = d.get(w)
            if stats[ind, i] > 0.9*len(text_of_vacancy):
                nv = stats[ind, i]/(np.sum(stats[ind, :])-stats[ind, i]+1)
            else:
                nv = 0
            val[j] = nv
            j += 1
        if j >= 1000:
            ind = np.argpartition(val, -n)[-n:]
            words = pd.DataFrame(words).iloc[ind].reset_index(drop=True)
        else:
            words = pd.DataFrame(words)
        df = pd.concat([df, words], axis=1)
    df.to_csv(os.getcwd()+"\\data\\"+l+"_features.csv", index=False, encoding="utf-8")
