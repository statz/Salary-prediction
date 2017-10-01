from gensim.models import Word2Vec as w2v
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot
from sklearn.ensemble import RandomForestRegressor

data_path = os.getcwd().split("/prediction")[0]+"/test_train/"
ud = ["down"]
for l in ud:
    gl_train_data_x = pd.read_csv(data_path + l+"_x_train.csv")["TextOfVacancy"]
    gl_test_data_x = pd.read_csv(data_path + l+"_x_test.csv")["TextOfVacancy"]
    train_y = pd.read_csv(data_path + l + "_y_train.csv")[l].as_matrix()
    test_y = pd.read_csv(data_path + l + "_y_test.csv")[l].as_matrix()


    train_data_x = list(map(str.split, gl_train_data_x))
    test_data_x = list(map(str.split, gl_test_data_x))

    max_len = np.max(list(map(len, train_data_x+test_data_x)))
    vect_train_x = np.zeros([len(train_data_x), 400])
    vect_test_x = np.zeros([len(test_data_x), 400])
    model = w2v.load(os.getcwd()+"/data/"+l+"_model")
    print(model.most_similar("python", topn=10))
    #vect_size = 400