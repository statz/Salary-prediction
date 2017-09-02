import pandas as pd
import os
from collections import Counter
import numpy as np

data_path = os.getcwd().split("\\prediction")[0]+"\\test_train\\"
TextOfVacancy = pd.read_csv(data_path+"down_x_train_normalized.csv")["TextOfVacancy"].tolist()
data = pd.read_csv("down_stats.csv")
words = data["words"].tolist()
stats = data.drop(["words"], axis=1).as_matrix()
val = np.zeros([len(words)])

#n = 1000
#ind = np.argpartition(val, -n)[-n:]
#words = pd.DataFrame(words).iloc[ind].to_csv("var.csv", index=False, encoding="utf-8")
