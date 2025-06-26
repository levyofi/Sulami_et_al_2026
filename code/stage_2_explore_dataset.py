import rasterio as rs
from rasterio.plot import show
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sb
from itertools import product
import re
import os
import pickle
import time

import torch
from torch import nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F

from dataset_creator import Dataset

trainset_path = 'data/trainset_ofir.pkl'
testset_path = 'data/testset_ofir.pkl'

#check the range of temperatures in each train map
trainset = Dataset.load_data(trainset_path)
for i in range(len(trainset)):
    print(trainset[i][2].min().item(), trainset[i][2].max().item())

#check the range of temperatures in each test map
testset = Dataset.load_data(testset_path)
for i in range(len(testset)):
    print(testset[i][2].min().item(), testset[i][2].max().item())

#create histograms of the different model features
map_types = ['height', 'shade', 'real_solar', 'skyview', 'TGI']
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for i in range(5):
    train_data = torch.cat([trainset.data[j][0][i] for j in range(len(trainset))]).numpy().reshape(-1)
    test_data = torch.cat([testset.data[j][0][i] for j in range(len(testset))]).numpy().reshape(-1)
    ax = axes[i // 3, i % 3]
    sb.histplot(train_data, ax=ax, label='Train', kde=True, bins=20)
    sb.histplot(test_data, ax=ax, label='Test', kde=True, color='red', bins=20)
    ax.set_title(map_types[i])
    ax.legend()
plt.tight_layout()
plt.show()
plt.savefig('FigureS1.png', dpi=300)


#create histograms of the different meteorological features
WD = 'data/submaps_cropped'
#get column names
met_columns = pd.read_csv(f'{WD}/Zeelim_29.5.19_0830/meteorological_data.csv').columns[1:-3]
# plot the meteorological data
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
for i in range(8):
    train_data = torch.stack([trainset.data[j][1] for j in range(len(trainset))]).t()[i]
    test_data = torch.stack([testset.data[j][1] for j in range(len(testset))]).t()[i]
    ax = axes[i // 3, i % 3]
    sb.histplot(train_data, ax=ax, label='Train', kde=True)
    sb.histplot(test_data, ax=ax, label='Test', kde=True, color='red')
    ax.set_title(met_columns[i])
    ax.legend()
plt.tight_layout()
plt.show()
plt.savefig('FigureS2.png', dpi=300)


