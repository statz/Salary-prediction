import pandas as pd
import os
import numpy as np

ud = ["down", "up"]
data_path = os.path.split(os.path.split(os.getcwd())[0])[0] + "//test_train//"

for b in ud:
    test_x = pd.read_csv(data_path+b+"_x_test_normalized.csv")["TextOfVacancy"]
    test_y = pd.read_csv(data_path+b+"_y_test.csv")[b]
    mean_sal = pd.read_csv(b+"_words_sal.csv")[["word", "mean_sal"]]
    d = {}
    for i in range(mean_sal.shape[0]):
        d.update({mean_sal["word"].ix[i]: int(mean_sal["mean_sal"].ix[i])})
    mean_sal = d
    predict = []
    for t in test_x:
        sal = 0
        j = 1
        words = t.split(" ")
        for w in words:
            v = mean_sal.get(w)
            if not (v is None):
                sal += v
                j += 1
        predict.append(sal/j)
    error = 0
    for i in range(len(predict)):
        error += abs(predict[i] - test_y.ix[i])
    print(error/len(predict))