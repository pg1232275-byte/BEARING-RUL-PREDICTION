import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# path to your data
PATH = '../nasa-bearing-rul/2nd_test/'

# check files loaded correctly
files = sorted(os.listdir(PATH))
print(f"Total files: {len(files)}")
print(f"First file: {files[0]}")
print(f"Last file: {files[-1]}")

# load one file and inspect
sample = pd.read_csv(PATH + files[0], sep='\t', header=None)
sample.columns = ['B1', 'B2', 'B3', 'B4']
print(f"\nShape: {sample.shape}")
print(sample.head())
print(sample.describe())

