import sys
assert sys.version_info >= (3, 5) 

import sklearn
assert sklearn.__version__ >= "0.20" 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import r2_score, make_scorer, mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor

import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import os
import seaborn as sns
import math
from scipy import stats 
import statsmodels.api as sm
import pickle 
import dtreeviz

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

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

def data_set():
    descriptors_2d = pd.read_csv('2d_descriptors.csv')
    descriptors_2d = descriptors_2d.dropna(axis=1, how='any')
    descriptors_2d = descriptors_2d.drop
    ls_2d = descriptors_2d.to_dict('records')

    for row in ls_2d:
        split_name = row['Names'].split('-')
        if len(split_name) == 3:
            row['Names'] = '-'.join(split_name[:2])
        else:
            row['Names'] = split_name[0]

    descriptors_2d = pd.DataFrame(ls_2d)

    df_all = pd.read_csv('3d_descriptors.csv')

    df_merged = df_all.merge(descriptors_2d, how='inner', left_on='Names', right_on='Names') 

    # Structures Omitted according to Selection Criteria
    df_merged = df_merged[~df_merged['Names'].isin(['PPF3FAMP', 'PPFAMP-CN', 'PPh3_NH2'])] 
    df_merged.to_csv('ligands_data_2d_3d.csv', index=False)

df_all = pd.read_csv('ligands_data_2d_3d.csv')

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

for train_index, test_index in split.split(df_xtb, df_xtb["cat_type"]):
    train_set = df_xtb.loc[train_index]
    test_set = df_xtb.loc[test_index]

for set_ in (train_set, test_set):
    set_.drop("cat_type", axis=1, inplace=True)

xtb = train_set.copy()
xtb.to_csv('training_data.csv', index=True)

df_merged_corr = pd.read_csv('training_data.csv', index_col=0) 

descriptors_3d = df_merged_corr[['Names', 'precat_xtb_homolumo', 'actcat_xtb_homolumo', 'precat_xtb_Mn_charge', 'actcat_xtb_Mn_charge', 'precat_xtb_singlet_triplet', 'actcat_xtb_singlet_triplet', 'xtb_MnH', 'xtb_HH', 'xtb_HN', 'xtb_bv_fraction', 'xtb_Hy_sasa', 'xtb_mol_sasa', 'xtb_mol_sasa_vol', 'xtb_angle_sr', 'xtb_angle_ca', 'xtb_angle_G', 'xtb_sterimol_L', 'xtb_sterimol_B_1', 'xtb_sterimol_B_5']].copy()

descriptors_2d = df_merged_corr.drop(columns=['DG_H2_1s', 'DG_H2_alkox', 'DG_H2_solvalkox', 'DG_Hy_re', 'GH_Hy_si', 'DDGHy', 'DrG_H2', 'DrG_H2s', 'precat_xtb_homolumo', 'actcat_xtb_homolumo', 'precat_xtb_Mn_charge', 'actcat_xtb_Mn_charge', 'precat_xtb_singlet_triplet', 'actcat_xtb_singlet_triplet', 'xtb_MnH', 'xtb_HH', 'xtb_HN', 'xtb_bv_fraction', 'xtb_Hy_sasa', 'xtb_mol_sasa', 'xtb_mol_sasa_vol', 'xtb_angle_sr', 'xtb_angle_ca', 'xtb_angle_G', 'xtb_sterimol_L', 'xtb_sterimol_B_1', 'xtb_sterimol_B_5'])

df_placeholder = descriptors_2d.copy()

for column in df_placeholder:
    if df_placeholder[column].sum() == 0:
        descriptors_2d.drop(columns=[column], inplace=True)

df_merged_corr1 = descriptors_2d.copy()

while True:
    corr_matrix = df_merged_corr1.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    indices = np.where((corr_matrix.abs() <= 0.5) & mask)

    if indices[0].size == 0:
        break
    
    df_merged_corr1.drop(corr_matrix.columns[indices[1][0]], axis=1, inplace=True)

while True:
    corr_matrix = df_merged_corr1.corr(numeric_only=True)

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    indices = np.where((corr_matrix.abs() >= 0.8) & mask)

    list = []
    for i, j in zip(indices[0], indices[1]):
        if i < j:
            corr_ij = corr_matrix.iloc[i, j]
            corr_i = corr_matrix.iloc[i, 0]
            corr_j = corr_matrix.iloc[j, 0]
            feature_i = corr_matrix.columns[i]
            feature_j = corr_matrix.columns[j]
            list.append((feature_i, feature_j, abs(corr_ij)))

    print(len(list))
    if len(list) == 0:
        break

    sorted_list = sorted(list, key=lambda x: x[2], reverse=True)
    feature_i = sorted_list[0][0]
    feature_j = sorted_list[0][1]

    corr_i = corr_matrix.loc[feature_i][0]
    corr_j = corr_matrix.loc[feature_j][0]

    if corr_i > corr_j:
        df_merged_corr1.drop(columns=[feature_j], inplace=True)
    else:
        df_merged_corr1.drop(columns=[feature_i], inplace=True)

df_merged = descriptors_3d.merge(df_merged_corr1, how='inner', left_on='Names', right_on='Names')

df_merged.to_csv('ligand_data_train.csv', index=True)

def corr_matrix():
    df_corr = df_merged.corr(numeric_only=True)
    df_corr.to_csv('train_data_corr.csv', index=True)

    palette = sns.color_palette('light:g', as_cmap=True)

    fig, ax = plt.subplots(figsize=(10,10))
    ax = sns.heatmap(df_corr, vmin=0, vmax=1, square=True, cmap=palette, linecolor='black', linewidths=0.1)
    plt.title('Pearson Correlation Coefficient')
    plt.savefig('corr_heatmap', bbox_inches='tight')

xtb_train_new = pd.read_csv('ligand_data_train.csv', index_col=0)

feature_names = xtb_train_new.drop(columns=['Names', 'DG_H2_1']).columns 

data_train = xtb_train_new.reset_index(drop=True)

X_train = data_train.drop(columns=['Names', 'DG_H2_1']) 
y_train = abs(data_train['DG_H2_1'].values)

data_test = test_set.copy()

for column in test_set:
    if column not in X_train.columns and column != 'DG_H2_1' and column != 'Names':
        data_test.drop(columns=[column], inplace=True)

X_test = data_test.drop(columns=['Names', 'DG_H2_1'])
y_test = abs(data_test['DG_H2_1'].values)

placeholder = data.copy()

for column in placeholder:
    if column not in X_train.columns and column != 'DG_H2_1' and column != 'Names':
        data.drop(columns=[column], inplace=True)

X = data.drop(columns=['Names', 'DG_H2_1'])

y = abs(data['DG_H2_1'].values)

mol_names_train = train_set['Names'].values
mol_names_test = test_set['Names'].values
mol_names = data['Names'].values

scores = {}
scores_non_linear = {}

def linear_grid():
    pipe = Pipeline([
        ('scaler', RobustScaler()),
        ('regressor', LinearRegression())
    ])
    
    scoring = {
        'r2': make_scorer(r2_score, greater_is_better=True),
        'neg_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
    }

    prep_train = pipe.fit(X_train, y_train)

    param_grid = {
        'regressor': [LinearRegression()]
    }

    # Grid Search
    grid = GridSearchCV(pipe, param_grid, cv=10, return_train_score=True, scoring=scoring, refit='neg_mean_squared_error')
    grid.fit(X_train, y_train)

    # Grid Results
    grid_res = grid.cv_results_
    for mean_score, params in zip(grid_res['mean_test_score'], grid_res['params']):
        print(np.sqrt(-mean_score), params)
    pd.DataFrame(grid.cv_results_).to_csv('grid_results_linear.csv', index=True)

    # Predictions
    train_pred = grid.best_estimator_.predict(X_train)
    df = pd.DataFrame(columns=['Actual','Predicted'], data={'Actual': y_train, 'Predicted': train_pred})
    df.insert(0, 'Names', mol_names_train)
    df.to_csv('actual_vs_predicted_train_linear.csv', index=False)

    test_pred = grid.best_estimator_.predict(X_test)
    df = pd.DataFrame(columns=['Actual','Predicted'], data={'Actual': y_test, 'Predicted': test_pred})
    df.insert(0, 'Names', mol_names_test)
    df.to_csv('actual_vs_predicted_test_linear.csv', index=False)

    y_pred = grid.best_estimator_.predict(X)
    df = pd.DataFrame(columns=['Actual','Predicted'], data={'Actual': y, 'Predicted': y_pred})
    df.insert(0, 'Names', mol_names)
    df.to_csv('TRIAL/actual_vs_predicted_linear.csv', index=False)

    # Coefficients
    coef_regressor = grid.best_estimator_.named_steps['regressor']
    coef = coef_regressor.coef_
    df = pd.DataFrame({"Feature": feature_names, "Coefficient": coef})
    df = df.sort_values(by="Coefficient", key=abs, ascending=False)
    df.to_csv("coef_linear.csv", index=False)

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(y_test, test_pred, label='Test', color='forestgreen', marker='.')
    ax.scatter(y_train, train_pred, label='Training', color='darkorange', marker='x')
    ax.plot([0, 20], [0, 20], c='grey', linestyle='--')
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    ax.set_xlabel('Calculated Barrier Height [kcal/mol]')
    ax.set_ylabel('Predicted Barrier Height [kcal/mol]')
    save_fig('predictions_graph_linear')
    exit()

def lasso_regression():
    pipe = Pipeline([
        ('scaler', RobustScaler()),
        ('pca', None),
        ('regressor', Lasso())
    ])

    scoring = {
        'r2': make_scorer(r2_score, greater_is_better=True),
        'neg_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
    }

    prep_train = pipe.fit(X_train, y_train)

    param_grid = {
    'pca': [None, *[PCA(n_components=i) for i in range(1, X_train.shape[1]+1)]], 
    'regressor': [Lasso()],
    'regressor__alpha': [0.0001, 0.001, 0.01, 0.1, 1]
    }

    # Grid Search
    grid = GridSearchCV(pipe, param_grid, cv=10, return_train_score=True, scoring=scoring, refit='neg_mean_squared_error')
    grid.fit(X_train, y_train)

    # Grid Search
    grid_res = grid.cv_results_
    for mean_score, params in zip(grid_res['mean_test_score'], grid_res['params']):
        print(np.sqrt(-mean_score), params)
    pd.DataFrame(grid.cv_results_).to_csv('grid_results_lasso.csv', index=True)

    # Predictions
    train_pred = grid.best_estimator_.predict(X_train)
    df = pd.DataFrame(columns=['Actual','Predicted'], data={'Actual': y_train, 'Predicted': train_pred})
    df.insert(0, 'Names', mol_names_train)
    df.to_csv('actual_vs_predicted_train_lasso.csv', index=False)

    test_pred = grid.best_estimator_.predict(X_test)
    df = pd.DataFrame(columns=['Actual','Predicted'], data={'Actual': y_test, 'Predicted': test_pred})
    df.insert(0, 'Names', mol_names_test)
    df.to_csv('actual_vs_predicted_test_lasso.csv', index=False)

    y_pred = grid.best_estimator_.predict(X)
    # print('Predictions:', y_pred)
    df = pd.DataFrame(columns=['Actual','Predicted'], data={'Actual': y, 'Predicted': y_pred})
    df.insert(0, 'Names', mol_names)
    df.to_csv('actual_vs_predicted_lasso', index=False)

    # Coefficients
    coef_regressor = grid.best_estimator_.named_steps['regressor']
    coef = coef_regressor.coef_
    df = pd.DataFrame({"Feature": feature_names, "Coefficient": coef})
    df = df.sort_values(by="Coefficient", key=abs, ascending=False)
    df.to_csv("coefficients_lasso.csv", index=False)

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(y_test, test_pred, label='Test', color='forestgreen', marker='.')
    ax.scatter(y_train, train_pred, label='Training', color='darkorange', marker='x')
    ax.plot([0, 20], [0, 20], c='grey', linestyle='--')
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    ax.set_xlabel('Calculated Barrier Height [kcal/mol]')
    ax.set_ylabel('Predicted Barrier Height [kcal/mol]')
    save_fig('predictions_graph_lasso')
    exit()

def ridge_regression(): 
    pipe = Pipeline([
    ('scaler', RobustScaler()),
    ('pca', None),
    ('regressor', Ridge())
    ])
    
    scoring = {
    'r2': make_scorer(r2_score, greater_is_better=True),
    'neg_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
    }

    prep_train = pipe.fit(X_train, y_train)

    # Grid
    param_grid = {
       'regressor': [Ridge()],
       'pca': [None, *[PCA(n_components=i) for i in range(1, X_train.shape[1]+1)]],
       'regressor__alpha': [0.0001, 0.001, 0.01, 0.1, 1],
       'regressor__solver': ['auto', 'svd', 'cholesky']
    }

    # Grid Search
    grid = GridSearchCV(pipe, param_grid, cv=10, return_train_score=True, scoring=scoring, refit='neg_mean_squared_error')
    grid.fit(X_train, y_train)

    # Grid Results
    grid_res = grid.cv_results_
    for mean_score, params in zip(grid_res['mean_test_score'], grid_res['params']):
        print(np.sqrt(-mean_score), params)
    pd.DataFrame(grid.cv_results_).to_csv('grid_results_ridge.csv', index=True)

    # Prediction
    train_pred = grid.best_estimator_.predict(X_train)
    df = pd.DataFrame(columns=['Actual','Predicted'], data={'Actual': y_train, 'Predicted': train_pred})
    mol_names = train_set['Names'].values
    df.insert(0, 'Names', mol_names)
    df.to_csv('actual_vs_predicted_train_ridge.csv', index=False)

    test_pred = grid.best_estimator_.predict(X_test)
    df = pd.DataFrame(columns=['Actual','Predicted'], data={'Actual': y_test, 'Predicted': test_pred})
    mol_names = test_set['Names'].values
    df.insert(0, 'Names', mol_names)
    df.to_csv('actual_vs_predicted_test_ridge.csv', index=False)

    y_pred = grid.best_estimator_.predict(X)
    df = pd.DataFrame(columns=['Actual','Predicted'], data={'Actual': y, 'Predicted': y_pred})
    mol_names = data['Names'].values
    df.insert(0, 'Names', mol_names)
    df.to_csv('actual_vs_predicted_ridge.csv', index=False)

    # Coefficients - in terms of PCA
    pca_components = grid.best_estimator_.named_steps['pca']
    pca_components = pca_components.components_
    pca_df = pd.DataFrame(pca_components, columns=feature_names, index=[f'PC{i+1}' for i in range(pca_components.shape[0])])
    pca_df.to_csv('pca_components_ridge.csv', index=True)

    most_important = [np.abs(pca_components[i]).argmax() for i in range(pca_components.shape[0])]
    most_important_names = [feature_names[most_important[i]] for i in range(pca_components.shape[0])]
    most_important_df = pd.DataFrame(most_important_names, columns=['Most Important Feature'], index=[f'PC{i+1}' for i in range(pca_components.shape[0])])
    most_important_df.to_csv('most_important_features_ridge.csv', index=True)

    explained_variance = grid.best_estimator_.named_steps['pca']
    explained_variance = explained_variance.explained_variance_ratio_
    explained_variance_df = pd.DataFrame(explained_variance, columns=['Explained Variance'], index=[f'PC{i+1}' for i in range(len(explained_variance))])
    cumulative_sum = np.cumsum(explained_variance)
    cumulative_sum_df = pd.DataFrame(cumulative_sum, columns=['Cumulative Sum'], index=[f'PC{i+1}' for i in range(len(cumulative_sum))])
    explained_variance_df = explained_variance_df.join(cumulative_sum_df)
    explained_variance_df.to_csv('explained_variance_ridge.csv', index=True)

    coef_regressor = grid.best_estimator_.named_steps['regressor']
    coef = coef_regressor.coef_
    coef_df = pd.DataFrame(coef, columns=['Coefficient'], index=[f'PC{i+1}' for i in range(len(coef))])
    coef_df.to_csv('coefficients_ridge.csv', index=True)

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(y_test, test_pred, label='Test', color='forestgreen', marker='.')
    ax.scatter(y_train, train_pred, label='Training', color='darkorange', marker='x')
    ax.plot([0, 20], [0, 20], c='grey', linestyle='--')
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    ax.set_xlabel('Calculated Barrier Height [kcal/mol]')
    ax.set_ylabel('Predicted Barrier Height [kcal/mol]')
    save_fig('predictions_graph_ridge')
    exit()

def elastic_net(): 
    pipe = Pipeline([
    ('scaler', RobustScaler()),
    ('pca', None),
    ('regressor', ElasticNet())
    ])

    scoring = {
    'r2': make_scorer(r2_score, greater_is_better=True),
    'neg_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
    }

    prep_train = pipe.fit(X_train, y_train)

    # Grid
    param_grid = {
        'pca': [None, *[PCA(n_components=i) for i in range(1, X_train.shape[1]+1)]],
        'regressor': [ElasticNet()],
        'regressor__alpha': [0.01, 0.1, 1],
        'regressor__l1_ratio': [0.1, 0.5, 0.9, 0.95, 0.99, 1]
    }

    # Grid Search
    grid = GridSearchCV(pipe, param_grid, cv=10, return_train_score=True, scoring=scoring, refit='neg_mean_squared_error')
    grid.fit(X_train, y_train)

    # Grid Results
    grid_res = grid.cv_results_
    for mean_score, params in zip(grid_res['mean_test_score'], grid_res['params']):
        print(np.sqrt(-mean_score), params)
    pd.DataFrame(grid.cv_results_).to_csv('grid_results_elastic.csv', index=True)

    # Predictions
    train_pred = grid.best_estimator_.predict(X_train)
    df = pd.DataFrame(columns=['Actual','Predicted'], data={'Actual': y_train, 'Predicted': train_pred})
    df.insert(0, 'Names', mol_names_train)
    df.to_csv('actual_vs_predicted_train_elastic.csv', index=False)

    test_pred = grid.best_estimator_.predict(X_test)
    df = pd.DataFrame(columns=['Actual','Predicted'], data={'Actual': y_test, 'Predicted': test_pred})
    df.insert(0, 'Names', mol_names_test)
    df.to_csv('actual_vs_predicted_test_elastic.csv', index=False)

    y_pred = grid.best_estimator_.predict(X)
    df = pd.DataFrame(columns=['Actual','Predicted'], data={'Actual': y, 'Predicted': y_pred})
    df.insert(0, 'Names', mol_names)
    df.to_csv('actual_vs_predicted_elastic.csv', index=False)

    # Coefficients
    coef_regressor = grid.best_estimator_.named_steps['regressor']
    coef = coef_regressor.coef_
    df = pd.DataFrame({"Feature": feature_names, "Coefficient": coef})
    df = df.sort_values(by="Coefficient", key=abs, ascending=False)
    df.to_csv('coefficients_elastic.csv', index=False)

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(y_test, test_pred, label='Test', color='forestgreen', marker='.')
    ax.scatter(y_train, train_pred, label='Training', color='darkorange', marker='x')
    ax.plot([0, 20], [0, 20], c='grey', linestyle='--')
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    ax.set_xlabel('Calculated Barrier Height [kcal/mol]')
    ax.set_ylabel('Predicted Barrier Height [kcal/mol]')
    save_fig('predictions_graph_elastic')
    exit()

def nearest_neighbours():
    k_max = len(data)
    k_max = (k_max * 0.80) * 0.90
    k_max = math.floor(k_max)

    pipe = Pipeline([
    ('scaler', RobustScaler()),
    ('pca', None),
    ('regressor', KNeighborsRegressor())
    ])

    scoring = {
    'r2': make_scorer(r2_score, greater_is_better=True),
    'neg_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
    }

    prep_train = pipe.fit(X_train, y_train)

    # Grid 
    param_grid = {
        'scaler': [RobustScaler()], 
        'pca': [None, *[PCA(n_components=i) for i in range(1, X_train.shape[1]+1)]], 
        'regressor': [KNeighborsRegressor()], 
        'regressor__n_neighbors': range(1, k_max), 
        'regressor__weights': ['uniform', 'distance'],
        'regressor__algorithm': ['auto']
    }

    # Grid Search
    grid = GridSearchCV(pipe, param_grid, cv=10, return_train_score=True, scoring=scoring, refit='neg_mean_squared_error')
    grid.fit(X_train, y_train)

    # Grid Results
    grid_res = grid.cv_results_
    for mean_score, params in zip(grid_res['mean_test_neg_mean_squared_error'], grid_res['params']):
        print(np.sqrt(-mean_score), params)
    pd.DataFrame(grid.cv_results_).to_csv('grid_result_KNN.csv', index=True)

    # Predictions
    train_pred = grid.best_estimator_.predict(X_train)
    df = pd.DataFrame(columns=['Actual','Predicted'], data={'Actual': y_train, 'Predicted': train_pred})
    df.insert(0, 'Names', mol_names_train)
    df.to_csv('actual_vs_predicted_train_KNN.csv', index=False)

    test_pred = grid.best_estimator_.predict(X_test)
    df = pd.DataFrame(columns=['Actual','Predicted'], data={'Actual': y_test, 'Predicted': test_pred})
    df.insert(0, 'Names', mol_names_test)
    df.to_csv('actual_vs_predicted_test_kNN.csv', index=False)

    y_pred = grid.best_estimator_.predict(X)
    df = pd.DataFrame(columns=['Actual','Predicted'], data={'Actual': y, 'Predicted': y_pred})
    df.insert(0, 'Names', mol_names)
    df.to_csv('actual_vs_predicted_KNN.csv', index=False)

    # Feature Importance - in terms of PCA
    pca_components = grid.best_estimator_.named_steps['pca']
    pca_components = pca_components.components_
    pca_df = pd.DataFrame(pca_components, columns=feature_names, index=[f'PC{i+1}' for i in range(pca_components.shape[0])])
    pca_df.to_csv('pca_components_KNN.csv', index=True)

    most_important = [np.abs(pca_components[i]).argmax() for i in range(pca_components.shape[0])]
    most_important_names = [feature_names[most_important[i]] for i in range(pca_components.shape[0])]
    most_important_df = pd.DataFrame(most_important_names, columns=['Most Important Feature'], index=[f'PC{i+1}' for i in range(pca_components.shape[0])])
    most_important_df.to_csv('most_important_features_KNN.csv', index=True)

    explained_variance = grid.best_estimator_.named_steps['pca']
    explained_variance = explained_variance.explained_variance_ratio_
    explained_variance_df = pd.DataFrame(explained_variance, columns=['Explained Variance'], index=[f'PC{i+1}' for i in range(len(explained_variance))])
    cumulative_sum = np.cumsum(explained_variance)
    cumulative_sum_df = pd.DataFrame(cumulative_sum, columns=['Cumulative Sum'], index=[f'PC{i+1}' for i in range(len(cumulative_sum))])
    explained_variance_df = explained_variance_df.join(cumulative_sum_df)
    explained_variance_df.to_csv('explained_variance_KNN.csv', index=True)

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(y_test, test_pred, label='Test', color='forestgreen', marker='.')
    ax.scatter(y_train, train_pred, label='Training', color='darkorange', marker='x')
    ax.plot([0, 20], [0, 20], c='grey', linestyle='--', lw=0.1)
    fit = np.polyfit(y_test, test_pred, 1)
    ax.plot(y_test, fit[0] * y_test + fit[1], color='forestgreen')
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    ax.set_xlabel('Calculated Barrier Height [kcal/mol]')
    ax.set_ylabel('Predicted Barrier Height [kcal/mol]')
    save_fig('predictions_graph_KNN')
    exit()

def svm_regression():
    pipe = Pipeline([
    ('scaler', RobustScaler()),
    ('pca', None),
    ('regressor', SVR())
    ])

    scoring = {
    'r2': make_scorer(r2_score, greater_is_better=True),
    'neg_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
    }

    prep_train = pipe.fit(X_train, y_train)

    # Grid 
    param_grid = {
        'scaler': [RobustScaler()], 
        'pca': [None, *[PCA(n_components=i) for i in range(1, X_train.shape[1]+1)]], 
        'regressor': [SVR()], 
        'regressor__kernel': ['linear', 'poly', 'rbf'], 
        'regressor__C': [1.5, 10, 15, 100], 
        'regressor__gamma': ['auto'], 
        'regressor__epsilon': [0.1, 0.2, 0.3, 0.4, 0.5],
        'regressor__degree': [1, 2, 3, 4, 5]
    }

    # Grid Search
    grid = GridSearchCV(pipe, param_grid, cv=10, return_train_score=True, scoring=scoring, refit='neg_mean_squared_error')
    grid.fit(X_train, y_train)

    # Grid Results
    grid_res = grid.cv_results_
    for mean_score, params in zip(grid_res['mean_test_neg_mean_squared_error'], grid_res['params']):
        print(np.sqrt(-mean_score), params)
    pd.DataFrame(grid.cv_results_).to_csv('grid_result_SVR.csv', index=True)

    # Predictions
    train_pred = grid.best_estimator_.predict(X_train)
    df = pd.DataFrame(columns=['Actual','Predicted'], data={'Actual': y_train, 'Predicted': train_pred})
    df.insert(0, 'Names', mol_names_train)
    df.to_csv('actual_vs_predicted_train_SVR.csv', index=False)

    test_pred = grid.best_estimator_.predict(X_test)
    df = pd.DataFrame(columns=['Actual','Predicted'], data={'Actual': y_test, 'Predicted': test_pred})
    df.insert(0, 'Names', mol_names_test)
    df.to_csv('actual_vs_predicted_test_SVR.csv', index=False)

    y_pred = grid.best_estimator_.predict(X)
    df = pd.DataFrame(columns=['Actual','Predicted'], data={'Actual': y, 'Predicted': y_pred})
    df.insert(0, 'Names', mol_names)
    df.to_csv('actual_vs_predicted_SVR.csv', index=False)

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(y_test, test_pred, label='Test', color='forestgreen', marker='.')
    ax.scatter(y_train, train_pred, label='Training', color='darkorange', marker='x')
    ax.plot([0, 20], [0, 20], c='grey', linestyle='--', lw=0.1)
    fit = np.polyfit(y_test, test_pred, 1)
    ax.plot(y_test, fit[0] * y_test + fit[1], color='forestgreen')
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    ax.set_xlabel('Calculated Barrier Height [kcal/mol]')
    ax.set_ylabel('Predicted Barrier Height [kcal/mol]')
    save_fig('predictions_graph_SVR')
    exit()

def d_tree():
    pipe = Pipeline([
        ('scaler', RobustScaler()),
        ('dt', DecisionTreeRegressor())
    ])

    scoring = {
    'r2': make_scorer(r2_score, greater_is_better=True),
    'neg_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
    }

    prep_train = pipe.fit(X_train, y_train)

    # Grid
    param_grid = {
        'scaler': [RobustScaler()], 
        'dt': [DecisionTreeRegressor()], 
        'dt__max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10], 
        'dt__splitter': ['best', 'random'],
        'dt__min_weight_fraction_leaf': [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    }

    # Grid Search
    grid = GridSearchCV(pipe, param_grid, cv=10, return_train_score=True, scoring=scoring, refit='neg_mean_squared_error')
    grid.fit(X_train, y_train)

    # Grid Results
    grid_res = grid.cv_results_
    for mean_score, params in zip(grid_res['mean_test_neg_mean_squared_error'], grid_res['params']):
        print(np.sqrt(-mean_score), params)
    pd.DataFrame(grid.cv_results_).to_csv('grid_result_DT.csv', index=True)

    # Predictions
    train_pred = grid.best_estimator_.predict(X_train)
    df = pd.DataFrame(columns=['Actual','Predicted'], data={'Actual': y_train, 'Predicted': train_pred})
    df.insert(0, 'Names', mol_names_train)
    df.to_csv('actual_vs_predicted_train_DT.csv', index=False)

    test_pred = grid.best_estimator_.predict(X_test)
    df = pd.DataFrame(columns=['Actual','Predicted'], data={'Actual': y_test, 'Predicted': test_pred})
    df.insert(0, 'Names', mol_names_test)
    df.to_csv('actual_vs_predicted_test_DT.csv', index=False)

    y_pred = grid.best_estimator_.predict(X)
    df = pd.DataFrame(columns=['Actual','Predicted'], data={'Actual': y, 'Predicted': y_pred})
    df.insert(0, 'Names', mol_names)
    df.to_csv('actual_vs_predicted_DT.csv', index=False)

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(y_test, test_pred, label='Test', color='forestgreen', marker='.')
    ax.scatter(y_train, train_pred, label='Training', color='darkorange', marker='x')
    ax.plot([0, 20], [0, 20], c='grey', linestyle='--', lw=0.1)
    fit = np.polyfit(y_test, test_pred, 1)
    ax.plot(y_test, fit[0] * y_test + fit[1], color='forestgreen')
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    ax.set_xlabel('Calculated Barrier Height [kcal/mol]')
    ax.set_ylabel('Predicted Barrier Height [kcal/mol]')
    save_fig('predictions_graph_DT')
    
    # Tree Visualisation
    d_tree = grid.best_estimator_['dt']
    d_tree.fit(X.values, y)
    viz = dtreeviz.model(model=d_tree, X_train=X, y_train=y, feature_names=feature_names, target_name='DG_H2_1') 
    viz.save('tree_viz.svg')
    exit()

def extra_tree():
    pipe = Pipeline([
        ('scaler', RobustScaler()),
        ('et', ExtraTreesRegressor())
    ])

    scoring = {
    'r2': make_scorer(r2_score, greater_is_better=True),
    'neg_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
    }

    prep_train = pipe.fit(X_train, y_train)

    # Grid
    param_grid = {
        'scaler': [RobustScaler()], 
        'et': [ExtraTreesRegressor()], 
        'et__max_depth': [3, 4, 5, 6], 
        'et__n_estimators': [50, 100, 200],
        'et__random_state': [42],
        'et__min_samples_split': [2, 3, 4]
    }

    # Grid Search
    grid = GridSearchCV(pipe, param_grid, cv=10, return_train_score=True, scoring=scoring, refit='neg_mean_squared_error')
    grid.fit(X_train, y_train)

    # Grid Results
    grid_res = grid.cv_results_
    for mean_score, params in zip(grid_res['mean_test_neg_mean_squared_error'], grid_res['params']):
        print(np.sqrt(-mean_score), params)
    pd.DataFrame(grid.cv_results_).to_csv('grid_result_ET.csv', index=True)

    # Predictions
    train_pred = grid.best_estimator_.predict(X_train)
    df = pd.DataFrame(columns=['Actual','Predicted'], data={'Actual': y_train, 'Predicted': train_pred})
    df.insert(0, 'Names', mol_names_train)
    df.to_csv('actual_vs_predicted_train_ET.csv', index=False)

    test_pred = grid.best_estimator_.predict(X_test)
    df = pd.DataFrame(columns=['Actual','Predicted'], data={'Actual': y_test, 'Predicted': test_pred})
    df.insert(0, 'Names', mol_names_test)
    df.to_csv('actual_vs_predicted_test_ET.csv', index=False)

    y_pred = grid.best_estimator_.predict(X)
    df = pd.DataFrame(columns=['Actual','Predicted'], data={'Actual': y, 'Predicted': y_pred})
    df.insert(0, 'Names', mol_names)
    df.to_csv('actual_vs_predicted_ET.csv', index=False)

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(y_test, test_pred, label='Test', color='forestgreen', marker='.')
    ax.scatter(y_train, train_pred, label='Training', color='darkorange', marker='x')
    ax.plot([0, 20], [0, 20], c='grey', linestyle='--', lw=0.1)
    fit = np.polyfit(y_test, test_pred, 1)
    ax.plot(y_test, fit[0] * y_test + fit[1], color='forestgreen')
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    ax.set_xlabel('Calculated Barrier Height [kcal/mol]')
    ax.set_ylabel('Predicted Barrier Height [kcal/mol]')
    save_fig('predictions_graph_ET')
    exit()

def bagging():
    pipe = Pipeline([
        ('scaler', RobustScaler()),
        ('bag', BaggingRegressor())
    ])

    scoring = {
    'r2': make_scorer(r2_score, greater_is_better=True),
    'neg_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
    }

    prep_train = pipe.fit(X_train, y_train)

    # Grid
    param_grid = {
        'scaler': [RobustScaler()], 
        'bag': [BaggingRegressor()], 
        'bag__n_estimators': [50, 100, 200],
    }

    # Grid Search
    grid = GridSearchCV(pipe, param_grid, cv=10, return_train_score=True, scoring=scoring, refit='neg_mean_squared_error')
    grid.fit(X_train, y_train)

    # Grid Results
    grid_res = grid.cv_results_
    for mean_score, params in zip(grid_res['mean_test_neg_mean_squared_error'], grid_res['params']):
        print(np.sqrt(-mean_score), params)
    pd.DataFrame(grid.cv_results_).to_csv('grid_result_BAG.csv', index=True)

    # Predictions
    train_pred = grid.best_estimator_.predict(X_train)
    df = pd.DataFrame(columns=['Actual','Predicted'], data={'Actual': y_train, 'Predicted': train_pred})
    df.insert(0, 'Names', mol_names_train)
    df.to_csv('actual_vs_predicted_train_BAG.csv', index=False)

    test_pred = grid.best_estimator_.predict(X_test)
    df = pd.DataFrame(columns=['Actual','Predicted'], data={'Actual': y_test, 'Predicted': test_pred})
    df.insert(0, 'Names', mol_names_test)
    df.to_csv('actual_vs_predicted_test_BAG.csv', index=False)

    y_pred = grid.best_estimator_.predict(X)
    df = pd.DataFrame(columns=['Actual','Predicted'], data={'Actual': y, 'Predicted': y_pred})
    df.insert(0, 'Names', mol_names)
    df.to_csv('actual_vs_predicted_BAG.csv', index=False)

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(y_test, test_pred, label='Test', color='forestgreen', marker='.')
    ax.scatter(y_train, train_pred, label='Training', color='darkorange', marker='x')
    ax.plot([0, 20], [0, 20], c='grey', linestyle='--', lw=0.1)
    fit = np.polyfit(y_test, test_pred, 1)
    ax.plot(y_test, fit[0] * y_test + fit[1], color='forestgreen')
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    ax.set_xlabel('Calculated Barrier Height [kcal/mol]')
    ax.set_ylabel('Predicted Barrier Height [kcal/mol]')
    save_fig('predictions_graph_BAG')
    exit()

def random_forest():
    pipe = Pipeline([
        ('scaler', RobustScaler())
        ('rf', RandomForestRegressor())
    ])

    scoring = {
    'r2': make_scorer(r2_score, greater_is_better=True),
    'neg_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
    }

    prep_train = pipe.fit(X_train, y_train)

    # Grid
    param_grid = {
        'rf__max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
        'rf__max_depth': [None, *[i for i in range(1, 10)]],
        'rf__min_samples_leaf': [1, 2, 4],
        'rf__n_estimators': [5000],
        'rf__max_features': [None, 'auto', 'sqrt', 'log2']
    }

    # Grid Search
    grid = GridSearchCV(pipe, param_grid, cv=10, return_train_score=True, scoring=scoring, refit='neg_mean_squared_error')
    grid.fit(X_train, y_train)

    # Grid Results
    grid_res = grid.cv_results_
    for mean_score, params in zip(grid_res['mean_test_neg_mean_squared_error'], grid_res['params']):
        print(np.sqrt(-mean_score), params)
    pd.DataFrame(grid.cv_results_).to_csv('grid_result_RF.csv', index=True)

    # Predictions
    train_pred = grid.best_estimator_.predict(X_train)
    df = pd.DataFrame(columns=['Actual','Predicted'], data={'Actual': y_train, 'Predicted': train_pred})
    df.insert(0, 'Names', mol_names_train)
    df.to_csv('actual_vs_predicted_train_RF.csv', index=False)

    test_pred = grid.best_estimator_.predict(X_test)
    df = pd.DataFrame(columns=['Actual','Predicted'], data={'Actual': y_test, 'Predicted': test_pred})
    df.insert(0, 'Names', mol_names_test)
    df.to_csv('actual_vs_predicted_test_RF.csv', index=False)

    y_pred = grid.best_estimator_.predict(X)
    df = pd.DataFrame(columns=['Actual','Predicted'], data={'Actual': y, 'Predicted': y_pred})
    df.insert(0, 'Names', mol_names)
    df.to_csv('actual_vs_predicted_RF.csv', index=False)

    # Feature Importance
    feature_importance = grid.feature_importances_
    df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
    df = df.sort_values(by="Importance", key=abs, ascending=False)
    df.to_csv("importance_FOREST.csv", index=False)

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(y_test, test_pred, label='Test', color='forestgreen', marker='.')
    ax.scatter(y_train, train_pred, label='Training', color='darkorange', marker='x')
    # Ideal line
    ax.plot([0, 20], [0, 20], color='grey', linestyle='--')
    fit = np.polyfit(y_test, test_pred, 1)
    ax.plot(y_test, fit[0] * y_test + fit[1], color='forestgreen')
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    ax.set_xlabel('Calculated Barrier Height [kcal/mol]')
    ax.set_ylabel('Predicted Barrier Height [kcal/mol]')
    save_fig('predictions_graph_FOREST')
    exit()
