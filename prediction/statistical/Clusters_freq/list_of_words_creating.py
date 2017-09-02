import pandas as pd
import os
import numpy as np

ud = ["down", "up"]
data_path = os.getcwd().split("\\prediction")[0]+"\\test_train\\"
for b in ud:
    data = pd.read_csv(data_path+b+"_x_train_normalized.csv")
    clusters = data["main"]
    words = set()
    train_x = data["TextOfVacancy"].as_matrix()
    for i in range(train_x.shape[0]):
        lex = str(train_x[i]).split(" ")
        for l in lex:
            if l not in words:
                words.add(l)
    df = pd.DataFrame(list(words), columns=["all",])
    for j in np.arange(60):
        print(j)
        words = set()
        train_x = data["TextOfVacancy"][data["main"] == j].as_matrix()
        for i in range(train_x.shape[0]):
            lex = str(train_x[i]).split(" ")
            for l in lex:
                if l not in words:
                    words.add(l)
        ndf = pd.DataFrame(list(words), columns=[str(j)])
        df = pd.concat([df,ndf], axis=1)
    df.to_csv(b+"_words_list.csv", encoding='utf-8', index=False)
