import os
import pandas as pd
import numpy as np


from data.dataset_loader import Dataset_Loader
from nn.layer import Layer

if not os.path.exists("data/digits_dataset.csv"):
    Dataset_Loader("data/digits")

df = pd.read_csv("data/digits_dataset.csv")

X = df.iloc[:, 1:].to_numpy()

input_layer = Layer(neurons_num=32,features_num=784)
input_layer_output = input_layer.forward(X[0:10])

print(input_layer.output)