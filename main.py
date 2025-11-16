import os
import pandas as pd

from data.dataset_loader import Dataset_Loader

if not os.path.exists("data/digits_dataset.csv"):
    Dataset_Loader("data/digits")

df = pd.read_csv("data/digits_dataset.csv")