import pandas as pd
import os

ud = ["down", "up"]
data_path = os.path.split(os.path.split(os.getcwd())[0])[0] + "//test_train//"
for b in ud:
    train_dict = {}
    train_x = pd.read_csv(data_path+b+"_x_train_normalized.csv")["TextOfVacancy"]
    train_y = pd.read_csv(data_path+b+"_y_train.csv")[b]
    for i in range(train_x.shape[0]):
        words = train_x.ix[i].split(" ")
        for w in words:
            val = train_dict.get(w)
            if val is None:
                train_dict.update({w: [1, train_y.ix[i]]})
            else:
                train_dict.update({w: [val[0]+1, val[1]+train_y.ix[i]]})
    mean_sal = []
    for i in train_dict:
        val = train_dict.get(i)
        mean_sal.append([i, val[0], val[1]/val[0]])
    df = pd.DataFrame(mean_sal, columns=["word", "freq", "mean_sal"])
    df.to_csv(b+"_words_sal.csv", encoding='utf-8', index=False)













