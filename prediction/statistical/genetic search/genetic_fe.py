import pandas as pd
import os
import numpy as np
import sys
sys.path.append(os.getcwd())
from ga import GA

ud = ["down", "up"]
data_path = os.path.split(os.path.split(os.getcwd())[0])[0] + "//test_train//"

ga = GA(1,2,3,4,5)