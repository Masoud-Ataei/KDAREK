#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmarking Regression Models on UCI Datasets
Models included: MLP, DAREK, KDAREK, Ensemble-KAN
Author: Masoud Ataei
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import logging
import json_tricks as json

from KDAREK import KDAREK

# -------------------------------
# Logging setup
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# -------------------------------
# Reproducibility
# -------------------------------
np.random.seed(0)
torch.manual_seed(0)

# -------------------------------
# Configuration
# -------------------------------
config = {
    'datasets': ['Concrete'],    
    'device': 'cpu'
}


# -------------------------------
# Utility Functions
# -------------------------------
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

def test_violation(pred, uncertainty, true):
    error = np.abs(pred - true)
    return (error > uncertainty).sum() / len(true)

def to_kan_dataset(Xtr, Xte, ytr, yte, device='cpu', noise=1e-4):
    Xtr = Xtr + np.random.uniform(-1, 1, Xtr.shape) * noise * np.max(np.abs(Xtr))
    Xte = Xte + np.random.uniform(-1, 1, Xte.shape) * noise * np.max(np.abs(Xte))
    dataset = {
        'train_input': torch.tensor(Xtr, dtype=torch.float32).to(device),
        'test_input': torch.tensor(Xte, dtype=torch.float32).to(device),
        'train_label': torch.tensor(ytr, dtype=torch.float32).to(device),
        'test_label': torch.tensor(yte, dtype=torch.float32).to(device)
    }
    return dataset

def lipschitz_bruteforce(X, Y, eps=1e-12):
    # X: (n,d), Y: (n,m)
    n = X.shape[0]
    max_ratio = np.array(0.0)
    for i in range(n):
        xi = X[i]
        yi = Y[i]
        diffs_x = X[i+1:] - xi        # (n-i-1, d)
        diffs_y = Y[i+1:] - yi        # (n-i-1, m)
        dx = np.linalg.norm(diffs_x, axis=1)
        dy = np.linalg.norm(diffs_y, axis=1)
        valid = dx > eps
        ratios = dy[valid] / dx[valid]
        if ratios.size:
            max_ratio = max(max_ratio, ratios.max())
    return max_ratio.item()
    
# -------------------------------
# Dataset Loaders
# -------------------------------
def load_concrete(d=None):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
    filename = "concrete.xls"
    if not os.path.exists(filename):
        logging.info(f"Downloading {filename}")
        import urllib.request
        urllib.request.urlretrieve(url, filename)
    df = pd.read_excel(filename)
    X, y = df.iloc[:, :-1].values.astype(np.float32), df.iloc[:, -1].values.astype(np.float32).reshape(-1,1)
    return X[:,:d], y

# Additional loaders can follow the same style...
dataset_loaders = {
    'Concrete': load_concrete,
    # 'WineRed': load_wine_red,
    # etc...
}

# -------------------------------
# Main Benchmark Loop
# -------------------------------
results = {}

for name in config['datasets']:
    logging.info(f"Running benchmark on {name}")
    loader = dataset_loaders[name]
    X, y = loader(d=None)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Standardize
    Xs = StandardScaler().fit(X_train)
    ys = StandardScaler().fit(y_train)
    X_train, X_test = Xs.transform(X_train), Xs.transform(X_test)
    y_train, y_test = ys.transform(y_train), ys.transform(y_test)
    
    lip = lipschitz_bruteforce(X_train, y_train)
    lip1 = lip
    lipk = (lip ** 3) / 6
    
    # Convert to Torch dataset
    dataset = to_kan_dataset(X_train, X_test, y_train, y_test, device=config['device'])
    print(lip1, np.sqrt(lip1))
    # Example: KDAREK
    kdk = KDAREK(
        mlp_width=[X.shape[1],5],
        kan_width=[5,1],
        kan_grid=8,
        kan_k=3,
        L_l=np.sqrt(lip1),
        kan_base_fun='silu',
        symbolic_enabled = False, 
        kan_extend=True,
        device=config['device']
    )
    kdk.fit(dataset, opt="Adam", steps=1000, lr=0.1, nonfixknot=True, rand_method = 'Kmean', scheduler="dec", step_sch=200, gamma=0.9)
    y_pred, y_var = kdk.predict(dataset['test_input'], L_mlp=np.sqrt(lip1), L_1=np.sqrt(lip1), L_k=lipk, oknot=2)
    
    rmse = evaluate(y_test, y_pred.detach().cpu().numpy())
    vio = test_violation(y_pred.detach().cpu().numpy(), y_var.detach().cpu().numpy(), y_test)
    print('dataset', name, 'RMSE', rmse, 'VIO', vio)
    

logging.info("Benchmark complete.")
