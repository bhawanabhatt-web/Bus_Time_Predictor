import pandas as pd
import numpy as np
import warnings
import os
import sys
import subprocess
import joblib
warnings.filterwarnings('ignore')

from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 1. DATA LOADING & EXPLORATION

def load_bus_data(file_path='bus_data.csv'):
    """Load the bus dwell times dataset."""
    print("STEP 1: LOADING DATA")
    print(f"Loading bus data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Successfully loaded {len(df):,} records")
        print(f"✓ Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"✗ File not found: {file_path}")
        return None


