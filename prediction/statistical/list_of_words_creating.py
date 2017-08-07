import pandas as pd
import os

ud = ["down", "up"]
data_path = os.path.split(os.path.split(os.getcwd())[0])[0] + "//test_train//"
for b in ud:
    train_x = pd.read_csv(data_path+b+"_x_train_normalized.csv")["TextOfVacancy"]
    words = []
    for i in range(train_x.shape[0]):
        print(i)
        lex = train_x.ix[i].split(" ")
        for l in lex:
            if l not in words:
                words.append(l)
    df = pd.DataFrame(words, columns=["words",])
    df.to_csv(b+"_words_list.csv", encoding='utf-8', index=False)
