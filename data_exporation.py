import sys
assert sys.version_info >= (3, 5) 

import sklearn
assert sklearn.__version__ >= "0.20" 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler


import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import os
import seaborn as sns
import math
import colorcet as cc
from scipy import stats 
import statsmodels.api as sm

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# from matplotlib.colors import ListedColormap
import matplotlib.lines as mlines

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "Images"
IMAGE_PATH = os.path.join(PROJECT_ROOT_DIR, 'Trial', CHAPTER_ID)
os.makedirs(IMAGE_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGE_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    plt.close()

df_all = pd.read_csv('ligands_data_2d_3d.csv')
dft_columns = []
for columns in df_all.columns:
    if 'dft' in columns.lower():
            dft_columns.append(columns)
if dft_columns:
    df_xtb = df_all.drop(columns=dft_columns)

data = df_xtb.copy()

feature_names = data.drop(columns=['DG_H2_1', 'DG_H2_1s', 'DG_H2_alkox', 'DG_H2_solvalkox', 'DG_Hy_re', 'GH_Hy_si', 'DDGHy', 'DrG_H2', 'DrG_H2s']).columns

def xtb_dft_comparision(): 
    df = pd.read_csv('3d_descriptors.csv')
    print(df.head)

    df_all = df[~df['Names'].isin(['PPF3FAMP', 'PPFAMP-CN', 'PPh3_NH2'])] 

    dft_columns = []
    for columns in df_all.columns:
        if 'dft' in columns.lower():
                dft_columns.append(columns)
    if dft_columns:
        df_xtb = df_all.drop(columns=dft_columns)
        
    xtb_placeholder = df_xtb.copy()
    xtb_placeholder = xtb_placeholder.drop(columns=['Names', 'DG_H2_1', 'DG_H2_1s', 'DG_H2_alkox', 'DG_H2_solvalkox', 'DG_Hy_re', 'GH_Hy_si', 'DDGHy', 'DrG_H2', 'DrG_H2s'])

    xtb_columns = []
    for columns in df_all.columns:
        if 'xtb' in columns.lower():
            xtb_columns.append(columns)
    if xtb_columns:
        df_dft = df_all.drop(columns=xtb_columns)
    
    dft_placeholder = df_dft.copy()
    dft_placeholder.drop(columns=['Names', 'DG_H2_1', 'DG_H2_1s', 'DG_H2_alkox', 'DG_H2_solvalkox', 'DG_Hy_re', 'GH_Hy_si', 'DDGHy', 'DrG_H2', 'DrG_H2s']) 

    xtb_placeholder.rename(columns=lambda x: x.replace('xtb_', ''), inplace=True)
    dft_placeholder.rename(columns=lambda x: x.replace('dft_', ''), inplace=True)

    fig, ax = plt.subplots(ncols=5, nrows=4, figsize=(20, 15), constrained_layout=True)

    cols = []
    for col1 in dft_placeholder.columns:
        for col2 in xtb_placeholder.columns:
            if col1 == col2:
                cols.append((col1, col2))

    ax = ax.ravel()
    counter = 0
    for col1 in dft_placeholder.columns:
        for col2 in xtb_placeholder.columns:
            if col1 == col2:
                ax[counter].scatter(x=xtb_placeholder[col2], y=dft_placeholder[col1], c='forestgreen', alpha=0.75)
                ax[counter].set_xlabel('xTB')
                ax[counter].set_ylabel('DFT')
                ax[counter].set_title(f'{col1}')
                counter += 1

    for i in range(len(ax)):
        if i == 3:
            ax[i].set(xlim=(-0.7,0), ylim=(-0.7,0))
        if i == 6:
            ax[i].set(xlim=(1.01,1.04), ylim=(1.01,1.04))
        x = np.linspace(ax[i].get_xlim()[0], ax[i].get_xlim()[1], 100)
        y = x
        ax[i].plot(x, y, ls="--", c=".3")

    for i in range(counter, len(ax)):
        fig.delaxes(ax[i])

    plt.savefig('xtb_dft_comparision')
    plt.close()

xtb_dft_comparision()

def attribute_plot():
    df_xtb1 = df_xtb.copy()
    df_xtb1.hist(bins=50, figsize=(20,15), color='forestgreen', grid=False)
    plt.tight_layout(pad=2, w_pad=None, h_pad=None)
    save_fig("xtb_attribute_histogram_plots")
    plt.close()

def scaling_info():
    xtb = pd.read_csv('Presentation/training_data.csv')
    xtb = xtb.drop(columns=['Names'])
    n_features = xtb.shape[1] - 1
    palette = sns.color_palette(cc.glasbey_light, n_colors=n_features)
    df_melted = xtb.iloc[:, 1:].melt(var_name='Features', value_name='Values')
    plt.figure(figsize=(6.5,8))
    blubb = sns.boxplot(x='Features', y='Values', data=df_melted, palette=palette, whiskerprops={'linewidth':0.5}, linewidth=0.5, fliersize=1)
    blubb.set_title("Unscaled")
    plt.yscale('log')
    blubb.set_xticklabels([])
    save_fig('unscaled')

    n_features = xtb.shape[1] - 1
    palette = sns.color_palette(cc.glasbey_light, n_colors=n_features)

    # Robust scaling
    r_xtb = xtb.copy()
    rs = RobustScaler()
    rs_xtb = rs.fit_transform(r_xtb)
    rs_xtb = pd.DataFrame(rs_xtb, columns=xtb.columns)
    df_rs_melted = rs_xtb.iloc[:,1:].melt(var_name='Features', value_name='Values')

    # Standard scaling
    ss_xtb = xtb.copy()
    standard = StandardScaler()
    standard_xtb = standard.fit_transform(ss_xtb)
    standard_xtb = pd.DataFrame(standard_xtb, columns=xtb.columns)
    df_standard_melted = standard_xtb.iloc[:,1:].melt(var_name='Features', value_name='Values')

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))
    sns.boxplot(x='Features', y='Values', data=df_rs_melted, ax=ax1, palette=palette, whiskerprops={'linewidth':0.5}, linewidth=0.5, fliersize=1)
    ax1.set_title("Robust Scaler")
    ax1.set_ylim(-5, 5)
    ax1.set_xticklabels([])

    sns.boxplot(x='Features', y='Values', data=df_standard_melted, ax=ax2, palette=palette, whiskerprops={'linewidth':0.5}, linewidth=0.5, fliersize=1)
    ax2.set_title("Standard Scaler")
    ax2.set_ylim(-5, 5)
    ax2.set_yticks([])
    ax2.set_ylabel("")
    ax2.set_xticklabels([])

    legend_handles = [mpatches.Patch(color=palette[i], label=label) for i, label in enumerate(xtb.columns[1:])]
    ax2.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Features")
    plt.tight_layout()
    save_fig('scaling_info_new')
    plt.close()

def violin_plot(): 
    knn_grid = pd.read_csv('grid_result_KNN.csv')
    svr_grid = pd.read_csv('grid_result_SVR.csv')
    dt_grid = pd.read_csv('grid_result_DT1.csv')
    et_grid = pd.read_csv('grid_result_ET.csv')
    bag_grid = pd.read_csv('grid_result_BAG.csv')
    rf_grid = pd.read_csv('grid_result_RF.csv')

    # RMSE Scores
    test_knn_nmse = []
    for i in range(len(knn_grid['rank_test_neg_mean_squared_error'])):
        if knn_grid['rank_test_neg_mean_squared_error'][i] == 1:
            for j in range(10):
                test_nmse = f'split{j}_test_neg_mean_squared_error'
                test_knn_nmse.append(knn_grid[test_nmse][i])
    test_mse_knn = [abs(x) for x in test_knn_nmse]
    test_rmse_knn = [math.sqrt(x) for x in test_mse_knn]

    test_svr_nmse = []
    for i in range(len(svr_grid['rank_test_neg_mean_squared_error'])):
        if svr_grid['rank_test_neg_mean_squared_error'][i] == 1:
            for j in range(10):
                test_nmse = f'split{j}_test_neg_mean_squared_error'
                test_svr_nmse.append(svr_grid[test_nmse][i])
    test_mse_svr = [abs(x) for x in test_svr_nmse]
    test_rmse_svr = [math.sqrt(x) for x in test_mse_svr]

    test_dt_nmse = []
    for i in range(len(dt_grid['rank_test_neg_mean_squared_error'])):
        if dt_grid['rank_test_neg_mean_squared_error'][i] == 1:
            for j in range(10):
                test_nmse = f'split{j}_test_neg_mean_squared_error'
                test_dt_nmse.append(dt_grid[test_nmse][i])
    test_mse_dt = [abs(x) for x in test_dt_nmse]
    test_rmse_dt = [math.sqrt(x) for x in test_mse_dt]

    test_et_nmse = []
    for i in range(len(et_grid['rank_test_neg_mean_squared_error'])):
        if et_grid['rank_test_neg_mean_squared_error'][i] == 1:
            for j in range(10):
                test_nmse = f'split{j}_test_neg_mean_squared_error'
                test_et_nmse.append(et_grid[test_nmse][i])
    test_mse_et = [abs(x) for x in test_et_nmse]
    test_rmse_et = [math.sqrt(x) for x in test_mse_et]

    test_bag_nmse = []
    for i in range(len(bag_grid['rank_test_neg_mean_squared_error'])):
        if bag_grid['rank_test_neg_mean_squared_error'][i] == 1:
            for j in range(10):
                test_nmse = f'split{j}_test_neg_mean_squared_error'
                test_bag_nmse.append(bag_grid[test_nmse][i])
    test_mse_bag = [abs(x) for x in test_bag_nmse]
    test_rmse_bag = [math.sqrt(x) for x in test_mse_bag]

    test_rf_nmse = []
    for i in range(len(rf_grid['rank_test_neg_mean_squared_error'])):
        if rf_grid['rank_test_neg_mean_squared_error'][i] == 1:
            for j in range(10):
                test_nmse = f'split{j}_test_neg_mean_squared_error'
                test_rf_nmse.append(rf_grid[test_nmse][i])
    test_mse_rf = [abs(x) for x in test_rf_nmse]
    test_rmse_rf = [math.sqrt(x) for x in test_mse_rf]

    labels = ['K Nearest Neighbour', 'Support Vector Regression', 'Decision Tree', 'Extra Trees', 'Bagging', 'Random Forest']
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.violinplot(data=[test_rmse_knn, test_rmse_svr, test_rmse_dt, test_rmse_et, test_rmse_bag, test_rmse_rf], palette='Set2', cut=0, bw=0.15, scale='count', inner='box', linewidth=0.75, ax=ax)
    fig.subplots_adjust(right=0.75)
    ax.set_ylabel('RMSE (kcal/mol)')
    ax.set_xticklabels(labels)
    for item in ax.get_xticklabels():
        item.set_rotation(90)
    fig.suptitle('Test RMSE during Cross Validation')
    
    save_fig('test_rmse_cv_violin_')
    plt.close()