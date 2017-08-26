import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import sys
sys.path.append(os.getcwd())
from ga import GA

ud = ["down", "up"]
data_path = os.getcwd().split("\\prediction")[0]+"\\test_train\\"

X = pd.read_csv(data_path + "down_x_train_normalized.csv")["TextOfVacancy"].as_matrix()
Y = pd.read_csv(data_path + "down_y_train_normalized.csv")["down"].as_matrix()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, train_size=0.1,  random_state=0)

words = {}
for i in range(x_train.shape[0]):
    lex = x_train[i].split(" ")
    for j in lex:
        if words.get(j) is None:
            words.update({j: 1})
        else:
            words.update({j: words.get(j) + 1})
nw = []
for w in words:
    k = words.get(w)
    if k > 50 and k < 2000:
        nw.append(w)

words = nw
print(len(words))
nw = None

binary_train_x = np.zeros([len(y_train), len(words)])
print(binary_train_x.shape)
binary_test_x = np.zeros([len(y_test), len(words)])

d = {words[w]: w for w in range(len(words))}
words = set(words)
for i in range(len(y_train)):
    tokens = x_train[i].split(" ")
    for j in tokens:
        if j in words:
            binary_train_x[i, d.get(j)] = 1

for i in range(len(y_test)):
    #print(i)
    tokens = x_test[i].split(" ")
    for j in tokens:
        if j in words:
            binary_test_x[i, d.get(j)] = 1

ga = GA(50, 5, len(words), 0.05, linear_model.LinearRegression())
ga.optimize(binary_train_x, y_train, binary_test_x, y_test)

