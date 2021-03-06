from gensim.models import Word2Vec as w2v
import pandas as pd
import os

data_path = os.getcwd().split("\\prediction")[0]+"\\test_train\\"
ud = ["down", "up"]
for l in ud:
    gl_train_data_x = pd.read_csv(data_path + l+"_x_train.csv")["TitleOfVacancy"].astype(str)
    gl_test_data_x = pd.read_csv(data_path + l+"_x_test.csv")["TitleOfVacancy"].astype(str)

    print("Start training")
    train_data_x = list(map(str.split, gl_train_data_x))
    test_data_x = list(map(str.split, gl_test_data_x))
    model = w2v(train_data_x+test_data_x, min_count=10, size=50)
    model.save(os.getcwd()+"\\data\\"+l+"_model")
