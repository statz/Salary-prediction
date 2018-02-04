import sqlite3
import pandas as pd
import os
from sklearn.model_selection import train_test_split

data_path = os.path.split(os.getcwd())[0] + "\\data"
test_train_path = os.path.split(os.getcwd())[0] + "\\test_train\\"

version = ["", "_normalized"]
for i in version:
    cnx = sqlite3.connect(data_path + "\\vacancies4" + i + ".db")
    data = pd.read_sql_query("SELECT * FROM hh", cnx)
    cnx.close()
    data = data[data["TextOfVacancy"] != '']
    columns_y = ["up", "down"]
    columns_x = []
    for c in data.columns.values:
        if c not in columns_y:
            columns_x.append(c)
    for j in columns_y:
        new_data = data[data[j] != 0]
        x = new_data.drop(columns_y, axis=1).as_matrix()
        y = new_data[j].as_matrix()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=10000, random_state=0)
        pd.DataFrame(x_train, columns=columns_x).to_csv("{0}{1}_x_train{2}.csv".format(test_train_path, j, i),
                                                        encoding="utf-8", index=False)
        pd.DataFrame(x_test, columns=columns_x).to_csv("{0}{1}_x_test{2}.csv".format(test_train_path, j, i),
                                                       encoding="utf-8", index=False)
        pd.DataFrame(y_train, columns=[j]).to_csv("{0}{1}_y_train{2}.csv".format(test_train_path, j, i),
                                                  encoding="utf-8", index=False)
        pd.DataFrame(y_test, columns=[j]).to_csv("{0}{1}_y_test{2}.csv".format(test_train_path, j, i),
                                                 encoding="utf-8", index=False)