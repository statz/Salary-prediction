import pandas as pd
import os
import numpy as np
import h5py
from keras.layers import LSTM
from keras.models import Model
from matplotlib import pyplot
from matplotlib import pyplot as plt
from keras.callbacks import History
from keras.layers import Dense, Dropout
from keras.utils.io_utils import HDF5Matrix
from keras.layers import Dense, Input, concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import keras.models
import matplotlib.mlab as mlab
from scipy import stats
import statsmodels.api as sm

data_path = os.getcwd().split("\\prediction")[0] + "\\test_train\\"
test_y = pd.read_csv(data_path + "down_y_test.csv")["down"].as_matrix() / 1000

if not os.path.exists(os.getcwd() + "\\data\\predictions.h5"):
    model = keras.models.load_model(os.getcwd() + "\\data\\" + "model2.h5")

    test_x1 = HDF5Matrix('data\\test_x1.h5', 'dataset_1')
    test_x2 = HDF5Matrix('data\\test_x2.h5', 'dataset_1')
    test_x3 = HDF5Matrix('data\\test_x3.h5', 'dataset_1')
    test_x4 = HDF5Matrix('data\\test_x4.h5', 'dataset_1')
    test_x5 = HDF5Matrix('data\\test_x5.h5', 'dataset_1')
    pr = model.predict([test_x1, test_x2, test_x3, test_x4, test_x5])
    print(pr)

    h5f = h5py.File(os.getcwd() + "\\data\\predictions.h5", 'w')
    h5f.create_dataset('dataset_1', data=pr)
    h5f.close()

h5f = h5py.File(os.getcwd() + "\\data\\predictions.h5",'r')
pr = h5f['dataset_1'][:]
h5f.close()
s = 0

errors = np.empty([len(test_y)])
for i in range(len(test_y)):
    s += (pr[i]-test_y[i])
    errors[i] = (pr[i]-test_y[i])/test_y[i]

print(np.mean(errors))
print(np.std(errors))
print(sm.stats.lilliefors(errors))
num_bins  = 200
fig, ax = plt.subplots()
n, bins, patches = ax.hist(errors, num_bins, normed=1)
y = mlab.normpdf(bins, np.mean(errors), 0.23)
ax.plot(bins, y, '--')
pyplot.show()

errors = np.abs(errors)
arr1 = np.zeros(40)
arr2 = np.zeros(40)
arr3 = np.zeros(40)
for i in range(3, 43):
    for j in range(len(test_y)):
        if (i-1)*5 <= test_y[j] <= i*5:
            arr1[i-3] += errors[j]
            arr2[i-3] += 1
for i in range(40):
    arr3[i] = arr1[i]+arr3[i-1]
    if arr2[i]:
        arr1[i] /= arr2[i]
for i in range(40):
    arr3[i] /= np.sum(arr2[:i+1])
pyplot.scatter(np.arange(len(arr1)), arr1)
pyplot.show()
pyplot.scatter(np.arange(len(arr1)), np.cumsum(arr2/np.sum(arr2)))
pyplot.show()
pyplot.scatter(np.arange(len(arr1)), (arr3))
pyplot.show()