import numpy as np
import pandas as pd

dataset_file_path = '/home/chli/chLi/Dataset/Elec/train.csv'

df = pd.read_csv(dataset_file_path)

print('====head====')
print(df.head())

print('====info====')
print(df.info())
