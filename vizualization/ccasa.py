import pandas as pd
import numpy as np
import os
from matplotlib import pyplot

data_path = os.getcwd().split("prediction")[0]+"prediction\\test_train\\"
y = pd.read_csv(data_path +"down_y_train_normalized.csv").as_matrix()
pyplot.hist(y/1000, bins=np.arange(0, 250, 5), normed=True)
pyplot.show()