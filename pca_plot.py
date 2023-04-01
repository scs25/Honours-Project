import sys
assert sys.version_info >= (3, 5) 
import os

import sklearn
assert sklearn.__version__ >= "0.20" 
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "Images"
IMAGE_PATH = os.path.join(PROJECT_ROOT_DIR, 'Presentation', CHAPTER_ID)
os.makedirs(IMAGE_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGE_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    plt.close()

# Download the data
df_all = pd.read_csv('ligand_data.csv')
dft_columns = []
for columns in df_all.columns:
    if 'dft' in columns.lower():
            dft_columns.append(columns)
if dft_columns:
    df_xtb = df_all.drop(columns=dft_columns)

data = df_xtb.copy()

df_placeholder = data.copy()

for column in df_placeholder:
    if df_placeholder[column].sum() == 0:
        data.drop(columns=[column], inplace=True)

# Stratified sampling 
df_xtb["cat_type"] = pd.cut(df_xtb["xtb_HH"], bins=[0., 3.5, 5.], labels=[1, 2])
df_xtb["cat_type"].hist()
plt.ylabel('Frequency')

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

df_xtb["cat_type"] = pd.cut(df_xtb["xtb_HH"], bins=[0., 3.5, 5.], labels=[1, 2])
df_xtb["cat_type"].hist()
plt.ylabel('Frequency')
plt.close()
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df_xtb, df_xtb["cat_type"]):
    train_set = df_xtb.loc[train_index]
    test_set = df_xtb.loc[test_index]
for set_ in (train_set, test_set):
    set_.drop("cat_type", axis=1, inplace=True)
xtb = train_set.copy()

def pca_elbow():
    data1 = xtb.copy()
    data = data1.drop(columns=['Names'])
    rb = RobustScaler()
    data = rb.fit_transform(data)

    pca = PCA()
    pca.fit(data)
    cumulative_sum = np.cumsum(pca.explained_variance_ratio_) 
    d = np.argmax(cumulative_sum >= 0.95) + 1 

    pca = PCA(n_components=d)
    data_reduced = pca.fit_transform(data)
    explained_variance = pca.explained_variance_ratio_
    print(explained_variance)
    print(1 - pca.explained_variance_ratio_.sum()) 

    data_inv = pca.inverse_transform(data_reduced) 
    np.allclose(data_inv, data) 
    error = np.mean(np.sum(np.square(data_inv - data), axis=1))


    # Plot Cumulative Explained Variance as a Function of Principal Components
    plt.figure(figsize=(12,8))
    plt.plot(cumulative_sum, linewidth=3)
    plt.xlabel("Principle Component")
    plt.xticks()
    plt.ylabel("Explained Variance")
    save_fig('explained_variance_plot')
    plt.close() 
    exit()