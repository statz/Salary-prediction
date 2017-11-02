import pandas as pd
import numpy as np
import os
from matplotlib import pyplot
data_path = os.getcwd().split("\\prediction")[0]+"\\test_train\\"
y = pd.read_csv(data_path +"down_y_train_normalized.csv").as_matrix()
print(np.std(y))
pyplot.hist(np.log(y), bins=np.arange(5, 15, 0.2))
pyplot.show()