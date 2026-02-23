import pandas as pd
import numpy as np
import warnings
import os
import sys
import subprocess
import joblib
warnings.filterwarnings('ignore')
from datetime import datetime

# sklearn – preprocessing
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, OneHotEncoder, PolynomialFeatures
)
from sklearn.compose   import ColumnTransformer
from sklearn.pipeline  import Pipeline

# sklearn – model selection
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold
)

# sklearn – linear model
from sklearn.linear_model import LinearRegression

# sklearn – metrics
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)

from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 1. DATA LOADING & EXPLORATION
def load_bus_data(file_path='bus_data.csv'):
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

def explore_data(df):
    """Explore the dataset structure and characteristics."""
    print("STEP 2: DATA EXPLORATION")
    print(f"\n Dataset Overview:")
    print(f"   Total records: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    print("\n Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"   {col}: {dtype}")
    print("\n Missing Values:")
    missing = df.isnull().sum()
    for col, count in missing.items():
        status = f"{count} ({count/len(df)*100:.2f}%)" if count > 0 else "No missing values"
        print(f"   {col}: {status}")
    print("\n Unique Value Counts:")
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].nunique() < 50:
            print(f"   {col}: {df[col].nunique()} unique values")
    print("\n Sample Data (First 5 rows):")
    print(df.head())
    print("\n Statistical Summary:")
    print(df.describe())

# 2. DATA PREPROCESSING
def preprocess_data(df):
    print("\n" + "=" * 70)
    print("STEP 3: DATA PREPROCESSING")
    print("=" * 70)

    df_processed = df.copy()

    # 3.1 Missing Values
    print("\n 3.1: Handling Missing Values")
    for col in df_processed.select_dtypes(include=[np.number]).columns:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
    for col in df_processed.select_dtypes(include=['object']).columns:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    print(f"   Remaining missing cells: {df_processed.isnull().sum().sum()}")

    # 3.2 Outlier Removal (IQR x3 on dwell_time_in_seconds)
    print("\n 3.2: Removing Outliers (IQR x 3 on dwell_time_in_seconds)")
    Q1, Q3 = df_processed['dwell_time_in_seconds'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 3 * IQR, Q3 + 3 * IQR
    before = len(df_processed)
    df_processed = df_processed[
        df_processed['dwell_time_in_seconds'].between(lower, upper)
    ]
    print(f"   Removed {before - len(df_processed):,} rows. Remaining: {len(df_processed):,}")

    # 3.3 Datetime conversion
    print("\n 3.3: Converting Datetime Columns")
    df_processed['arrival_datetime'] = pd.to_datetime(
        df_processed['date'] + ' ' + df_processed['arrival_time'], errors='coerce')
    df_processed['departure_datetime'] = pd.to_datetime(
        df_processed['date'] + ' ' + df_processed['departure_time'], errors='coerce')
    df_processed.dropna(subset=['arrival_datetime', 'departure_datetime'], inplace=True)
    df_processed.sort_values(['trip_id', 'arrival_datetime'], inplace=True)
    df_processed.reset_index(drop=True, inplace=True)
    print(f"   Valid datetime rows: {len(df_processed):,}")

    # Time-based features
    h = df_processed['arrival_datetime'].dt.hour
    df_processed['arrival_hour']   = h
    df_processed['arrival_minute'] = df_processed['arrival_datetime'].dt.minute
    df_processed['day_of_week']    = df_processed['arrival_datetime'].dt.dayofweek
    df_processed['month']          = df_processed['arrival_datetime'].dt.month
    df_processed['day']            = df_processed['arrival_datetime'].dt.day

    # Cyclical encoding -- wraps hour-23 next to hour-0
    df_processed['sin_hour'] = np.sin(2 * np.pi * h / 24)
    df_processed['cos_hour'] = np.cos(2 * np.pi * h / 24)
    print("   arrival_hour, arrival_minute, day_of_week, month, day")
    print("   sin_hour, cos_hour (cyclical encoding)")

    # 3.5 Time period categories & binary flags
    print("\n 3.5: Time Period Categories & Flags")
    def categorize_time(hr):
        if   5 <= hr <  9: return 'morning_peak'
        elif 9 <= hr < 12: return 'morning'
        elif 12 <= hr < 15: return 'afternoon'
        elif 15 <= hr < 19: return 'evening_peak'
        elif 19 <= hr < 22: return 'evening'
        else:               return 'night'

    df_processed['time_period']  = h.apply(categorize_time)
    df_processed['is_peak_hour'] = df_processed['time_period'].isin(
        ['morning_peak', 'evening_peak']).astype(int)
    df_processed['is_weekend']   = df_processed['day_of_week'].isin([5, 6]).astype(int)
    # is_weekday = 1 - is_weekend (perfectly collinear --> NOT created)
    print("   time_period, is_peak_hour, is_weekend created")
    print("   is_weekday DROPPED (is_weekday = 1 - is_weekend, perfectly collinear)")

    # 3.6 Derived features
    print("\n 3.6: Derived Features")
    df_processed['actual_dwell_time'] = (
        df_processed['departure_datetime'] - df_processed['arrival_datetime']
    ).dt.total_seconds()

    df_processed['travel_time_to_next_stop'] = (
        df_processed.groupby('trip_id')['arrival_datetime']
        .shift(-1) - df_processed['arrival_datetime']
    ).dt.total_seconds() / 60
    
    print("   actual_dwell_time (s), travel_time_to_next_stop (min)")

    # 3.7 Lag features
    print("\n 3.7: Lag Features")
    df_processed['prev_travel_time'] = df_processed.groupby('trip_id')[
        'travel_time_to_next_stop'].shift(1)
    df_processed['prev_dwell_time']  = df_processed.groupby('trip_id')[
        'actual_dwell_time'].shift(1)
    print("   prev_travel_time, prev_dwell_time")

    # 3.8 Clean target variable
    print("\n 3.8: Cleaning Target Variable")
    df_processed.dropna(subset=['travel_time_to_next_stop'], inplace=True)
    df_processed['prev_travel_time'].fillna(
        df_processed['travel_time_to_next_stop'].median(), inplace=True)
    df_processed['prev_dwell_time'].fillna(
        df_processed['actual_dwell_time'].median(), inplace=True)
    df_processed = df_processed[df_processed['travel_time_to_next_stop'] > 0]
    print(f"   Final records: {len(df_processed):,}")

    # 3.9 Interaction features
    print("\n 3.9: Interaction Features")
    df_processed['hour_x_peak']    = df_processed['arrival_hour'] * df_processed['is_peak_hour']
    df_processed['dow_x_weekend']  = df_processed['day_of_week']  * df_processed['is_weekend']
    df_processed['prev_tt_x_peak'] = df_processed['prev_travel_time'] * df_processed['is_peak_hour']
    print("   hour_x_peak, dow_x_weekend, prev_tt_x_peak")

    # 3.10 Drop raw string / datetime columns
    print("\n 3.10: Dropping Raw String / Datetime Columns")
    drop_cols = ['date', 'arrival_time', 'departure_time',
                 'arrival_datetime', 'departure_datetime']
    drop_cols = [c for c in drop_cols if c in df_processed.columns]
    df_processed.drop(columns=drop_cols, inplace=True)
    print(f"   Dropped: {drop_cols}")

    print(f"\n Preprocessing Complete! Final shape: {df_processed.shape}")
    label_encoders = {}  
    return df_processed, label_encoders

def save_cleaned_dataset(df_processed, path='bus_data_cleaned.csv'):
    """Save the cleaned and feature-engineered dataset to CSV after preprocessing."""
    print("\n" + "=" * 70)
    print("SAVING CLEANED DATASET")
    print("=" * 70)

    df_to_save = df_processed.copy()
    df_to_save.to_csv(path, index=False)

    print(f"  ✓ Cleaned dataset saved → '{path}'")
    print(f"  ✓ Shape   : {df_to_save.shape[0]:,} rows × {df_to_save.shape[1]} columns")
    print(f"  ✓ Columns : {list(df_to_save.columns)}")
    print(f"  ✓ Missing : {df_to_save.isnull().sum().sum()} values remaining")

# 3. MULTICOLLINEARITY REMOVAL  

def remove_multicollinearity(X_num, threshold_corr=0.85, threshold_vif=10.0):

    print("\n" + "=" * 70)
    print("STEP 3b: MULTICOLLINEARITY REMOVAL")
    print("=" * 70)

    cols = list(X_num.columns)

    # Pass 1: correlation filter
    print(f"\n Pass 1 -- Correlation filter (|r| > {threshold_corr})")
    corr   = X_num[cols].corr().abs()
    upper  = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_c = [c for c in upper.columns if any(upper[c] > threshold_corr)]
    for c in drop_c:
        if c in cols:
            cols.remove(c)
    print(f"   Dropped (correlation): {drop_c}")
    print(f"   Remaining: {len(cols)}")

    # Pass 2: VIF filter
    print(f"\n Pass 2 -- VIF filter (VIF > {threshold_vif})")
    dropped_vif = []
    while True:
        data = X_num[cols].values
        vif_df = pd.DataFrame({
            'feature': cols,
            'VIF': [variance_inflation_factor(data, i) for i in range(len(cols))]
        }).sort_values('VIF', ascending=False)

        max_row = vif_df.iloc[0]
        if max_row['VIF'] > threshold_vif:
            print(f"   Removing '{max_row['feature']}' (VIF = {max_row['VIF']:.2f})")
            cols.remove(max_row['feature'])
            dropped_vif.append(max_row['feature'])
        else:
            break

    print(f"   Dropped (VIF): {dropped_vif}")
    print(f"   Final numeric features after multicollinearity removal: {len(cols)}")
    return cols

# 4. FEATURE PREPARATION
def prepare_features_for_ml(df_processed):
    print("\n" + "=" * 70)
    print("STEP 4: PREPARING FEATURES FOR MACHINE LEARNING")
    print("=" * 70)

    # Categorical columns (retain as strings for OHE inside Pipeline)
    ohe_cols = [c for c in ['bus_stop', 'deviceid', 'time_period']
                if c in df_processed.columns
                and df_processed[c].dtype == object]

    # direction: numeric (1 or 2) -- treat as numeric, not categorical
    candidate_num = [
        'direction',
        'arrival_hour', 'arrival_minute', 'day_of_week', 'month', 'day',
        'sin_hour', 'cos_hour',
        'is_peak_hour', 'is_weekend',
        'actual_dwell_time', 'dwell_time_in_seconds',
        'prev_travel_time', 'prev_dwell_time',
        'hour_x_peak', 'dow_x_weekend', 'prev_tt_x_peak',
    ]
    candidate_num = [c for c in candidate_num if c in df_processed.columns]

    # Remove multicollinearity from numeric features
    num_cols = remove_multicollinearity(df_processed[candidate_num])

    # Polynomial expansion on three high-signal numeric features
    poly_base = ['prev_travel_time', 'prev_dwell_time', 'arrival_hour']
    poly_cols  = [c for c in poly_base if c in num_cols]

    X = df_processed[ohe_cols + num_cols].copy()
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    y_raw = df_processed['travel_time_to_next_stop'].copy()
    y_log = np.log1p(y_raw)

    print(f"\n   OHE categorical cols   : {ohe_cols}")
    print(f"   Numeric cols (post-VIF): {num_cols}")
    print(f"   Poly-expanded cols     : {poly_cols}")
    print(f"\n   Feature matrix : {X.shape}")
    print(f"   Target (raw)   : min={y_raw.min():.2f}  max={y_raw.max():.2f}  "
          f"mean={y_raw.mean():.2f}")
    print(f"   Target (log1p) : min={y_log.min():.2f}  max={y_log.max():.2f}  "
          f"mean={y_log.mean():.2f}")

    return ohe_cols, num_cols, poly_cols, X, y_raw, y_log


# 5. MODEL TRAINING

def compute_regression_metrics(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    if label:
        print(f"\n   {label}:")
    print(f"      R²   : {r2:>10.4f}")
    print(f"      MAE  : {mae:>10.4f} min")
    print(f"      RMSE : {rmse:>10.4f} min")
    print(f"      MAPE : {mape:>10.2f} %")

    return {'r2': r2, 'mae': mae, 'rmse': rmse, 'mape': mape}

def build_pipeline(ohe_cols, num_cols, poly_cols, estimator):
    other_num = [c for c in num_cols if c not in poly_cols]

    poly_branch = Pipeline([
        ('scaler', StandardScaler()),
        ('poly',   PolynomialFeatures(degree=2, interaction_only=True,
                                      include_bias=False)),
    ])

    transformers = []
    if ohe_cols:
        transformers.append((
            'ohe',
            OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False),
            ohe_cols
        ))
    if poly_cols:
        transformers.append(('poly_num', poly_branch, poly_cols))
    if other_num:
        transformers.append(('std_num', StandardScaler(), other_num))

    ct = ColumnTransformer(transformers=transformers, remainder='drop')
    return Pipeline([('preprocessor', ct), ('model', estimator)])


def train_model(ohe_cols, num_cols, poly_cols,
                X, y_raw, y_log,
                test_size=0.2, random_state=42):

    print("\n" + "=" * 70)
    print("STEP 5: MODEL TRAINING & EVALUATION")
    print("=" * 70)

    print("\n 5.1: Train-Test Split (80 / 20)")
    X_tr, X_te, ylog_tr, ylog_te, yraw_tr, yraw_te = train_test_split(
        X, y_log, y_raw, test_size=test_size, random_state=random_state
    )
    print(f"   Train: {len(X_tr):,}  |  Test: {len(X_te):,}")

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    print(f"\n{'─'*60}")
    print(f"  MODEL: Multiple Linear Regression")
    print(f"{'─'*60}")

    pipe = build_pipeline(ohe_cols, num_cols, poly_cols, LinearRegression())
    pipe.fit(X_tr, ylog_tr)

    # Back-transform predictions: expm1 reverses log1p
    y_pred_tr = np.clip(np.expm1(pipe.predict(X_tr)), 0, None)
    y_pred_te = np.clip(np.expm1(pipe.predict(X_te)), 0, None)

    train_metrics = compute_regression_metrics(yraw_tr, y_pred_tr, "Train Set")
    test_metrics  = compute_regression_metrics(yraw_te, y_pred_te, "Test  Set")

    # Cross-validation R² on log-space
    cv_r2 = cross_val_score(pipe, X_tr, ylog_tr,
                             cv=cv, scoring='r2', n_jobs=-1)
    print(f"\n   5-Fold CV R²: mean={cv_r2.mean():.4f}  std={cv_r2.std():.4f}")

    results = {
        "Multiple Linear Regression": {
            'pipeline'    : pipe,
            'best_params' : {},
            'X_tr': X_tr, 'X_te': X_te,
            'yraw_tr': yraw_tr, 'yraw_te': yraw_te,
            'y_pred_tr': y_pred_tr, 'y_pred_te': y_pred_te,
            'train_metrics': train_metrics,
            'test_metrics' : test_metrics,
            'cv_r2_mean'   : cv_r2.mean(),
            'cv_r2_std'    : cv_r2.std(),
        }
    }

    # Model performance summary
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE -- TEST SET (original scale)")
    print("=" * 70)
    tm = test_metrics
    print(f"  R²   : {tm['r2']:>10.4f}")
    print(f"  MAE  : {tm['mae']:>10.4f} min")
    print(f"  RMSE : {tm['rmse']:>10.4f} min")
    print(f"  MAPE : {tm['mape']:>10.2f} %")
    print(f"  CV-R²: {cv_r2.mean():>10.4f} ± {cv_r2.std():.4f}")

    # Feature importances from MLR Pipeline
    try:
        feat_names = pipe.named_steps['preprocessor'].get_feature_names_out()
        coefs = pipe.named_steps['model'].coef_
        feat_imp = pd.DataFrame({
            'feature'    : feat_names,
            'coefficient': coefs,
            'abs_coef'   : np.abs(coefs)
        }).sort_values('abs_coef', ascending=False)

        print("\nTop 10 Features (MLR -- absolute coefficient after OHE + Poly):")
        for _, row in feat_imp.head(10).iterrows():
            print(f"   {str(row['feature']):<45}: {row['coefficient']:+.4f}")
        results['feature_importance'] = feat_imp
    except Exception as e:
        print(f"   Could not extract feature names: {e}")
        results['feature_importance'] = pd.DataFrame()

    results['best_model_name']    = "Multiple Linear Regression"
    results['primary_model_name'] = "Multiple Linear Regression"
    return results

# 6. PREDICTION FUNCTION

def predict_arrival_time(results, features, user_location_stop,
                          destination_stop, df_raw, ohe_cols, num_cols):

    print("\n" + "=" * 70)
    print("STEP 6: PREDICTION")
    print("=" * 70)
    print(f"\nJourney: {user_location_stop} -> {destination_stop}")

    pipe     = results["Multiple Linear Regression"]['pipeline']
    all_cols = ohe_cols + num_cols
    feat_df  = pd.DataFrame([{c: features.get(c, np.nan) for c in all_cols}])

    for c in num_cols:
        if feat_df[c].isnull().any():
            feat_df[c] = df_raw[c].median() if c in df_raw.columns else 0.0

    y_log_pred = pipe.predict(feat_df)[0]
    predicted  = max(0.0, float(np.expm1(y_log_pred)))

    minutes = int(predicted)
    seconds = int((predicted - minutes) * 60)
    print(f"\nEstimated Travel Time: {minutes} min {seconds} sec")

    return {
        'predicted_time_minutes': predicted,
        'from_stop': user_location_stop,
        'to_stop'  : destination_stop,
    }

# 7. VISUALIZATION
def plot_all_visualizations(results, df_original):
    print("\n Generating visualisations...")

    primary = results["Multiple Linear Regression"]

    # Fig 1: MLR evaluation
    fig1, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig1.suptitle("Multiple Linear Regression -- Model Evaluation",
                  fontsize=16, fontweight='bold')

    axes[0, 0].scatter(primary['yraw_te'], primary['y_pred_te'], alpha=0.3, s=5)
    lims = [primary['yraw_te'].min(), primary['yraw_te'].max()]
    axes[0, 0].plot(lims, lims, 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Travel Time (min)')
    axes[0, 0].set_ylabel('Predicted Travel Time (min)')
    axes[0, 0].set_title(f"Actual vs Predicted\nR2 = {primary['test_metrics']['r2']:.4f}")
    axes[0, 0].grid(True, alpha=0.3)

    residuals = primary['yraw_te'].values - primary['y_pred_te']
    axes[0, 1].scatter(primary['y_pred_te'], residuals, alpha=0.3, s=5)
    axes[0, 1].axhline(0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted (min)')
    axes[0, 1].set_ylabel('Residuals (min)')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)

    if not results['feature_importance'].empty:
        top_f = results['feature_importance'].head(10)
        axes[1, 0].barh(range(len(top_f)), top_f['abs_coef'], color='steelblue')
        axes[1, 0].set_yticks(range(len(top_f)))
        axes[1, 0].set_yticklabels(top_f['feature'].astype(str), fontsize=8)
        axes[1, 0].set_xlabel('|Coefficient|')
        axes[1, 0].set_title('Top 10 Feature Importances')
        axes[1, 0].grid(True, alpha=0.3, axis='x')

    axes[1, 1].hist(residuals, bins=60, edgecolor='black', alpha=0.7, color='coral')
    axes[1, 1].axvline(0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Prediction Error (min)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f"Error Distribution\nMAE = {primary['test_metrics']['mae']:.3f} min")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=150, bbox_inches='tight')
    print("   Saved: model_evaluation.png")
    plt.close(fig1)

    # Fig 2: Data distributions
    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
    fig2.suptitle("Bus Dataset -- Data Distributions", fontsize=14, fontweight='bold')

    df_orig = df_original.copy()
    df_orig['arrival_datetime'] = pd.to_datetime(
        df_orig['date'].astype(str) + ' ' + df_orig['arrival_time'], errors='coerce')
    df_orig['arrival_hour'] = df_orig['arrival_datetime'].dt.hour

    axes2[0, 0].hist(df_orig['dwell_time_in_seconds'].clip(0, 300),
                     bins=50, color='steelblue', edgecolor='black', alpha=0.8)
    axes2[0, 0].set_title('Dwell Time Distribution (s)')
    axes2[0, 0].set_xlabel('Dwell Time (s)')
    axes2[0, 0].grid(True, alpha=0.3)

    hc = df_orig['arrival_hour'].value_counts().sort_index()
    axes2[0, 1].bar(hc.index, hc.values, color='coral')
    axes2[0, 1].set_title('Arrivals by Hour')
    axes2[0, 1].set_xlabel('Hour')
    axes2[0, 1].grid(True, alpha=0.3, axis='y')

    sc = df_orig['bus_stop'].value_counts().head(15)
    axes2[0, 2].bar(sc.index.astype(str), sc.values, color='green', alpha=0.8)
    axes2[0, 2].set_title('Top 15 Bus Stops')
    axes2[0, 2].tick_params(axis='x', rotation=45)
    axes2[0, 2].grid(True, alpha=0.3, axis='y')

    dc = df_orig['direction'].value_counts()
    axes2[1, 0].pie(dc.values,
                    labels=[f'Direction {d}' for d in dc.index],
                    autopct='%1.1f%%', colors=['#2196F3', '#FF9800'])
    axes2[1, 0].set_title('Direction Distribution')

    day_c = df_orig['arrival_datetime'].dt.dayofweek.value_counts().sort_index()
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes2[1, 1].bar([day_labels[i] for i in day_c.index], day_c.values,
                    color='purple', alpha=0.8)
    axes2[1, 1].set_title('Arrivals by Day of Week')
    axes2[1, 1].grid(True, alpha=0.3, axis='y')

    axes2[1, 2].hist(df_orig['dwell_time_in_seconds'].clip(0, 300),
                     bins=50, cumulative=True, density=True,
                     histtype='step', color='red', linewidth=2)
    axes2[1, 2].set_title('CDF of Dwell Time')
    axes2[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi=150, bbox_inches='tight')
    print("   Saved: data_distribution.png")
    plt.close(fig2)

    # Fig 3: Correlation heatmap
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_original.select_dtypes(include=[np.number]).corr(),
                annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax3, cbar_kws={'shrink': 0.8})
    ax3.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
    print("   Saved: correlation_heatmap.png")
    plt.close(fig3)

    print("   All visualisations saved successfully.")

# 8. UI LAUNCHER

def launch_bus_tracker_ui():
    """Launch the Bus Tracker UI application."""
    print("\n" + "=" * 70)
    print("LAUNCHING BUS TRACKER UI")
    print("=" * 70)
    ui_candidates = ['bus_time_tracker_UI.py']
    ui_file = None
    for f in ui_candidates:
        if os.path.exists(f):
            ui_file = f
            break
    if ui_file:
        print(f"\n✓ Found UI file: {ui_file}")
        try:
            subprocess.run([sys.executable, ui_file])
        except Exception as e:
            print(f"❌ Error launching UI: {e}")
    else:
        print("⚠ UI file not found. Please run 'bus_time_tracker_UI.py' manually.")


# 9. MAIN PIPELINE

def main():
    print("\n" + "=" * 70)
    print("BUS ARRIVAL TIME PREDICTION SYSTEM")
    print("    Multiple Linear Regression | ML Pipeline")
    print("=" * 70)

    # Load
    df = load_bus_data('bus_data.csv')
    if df is None:
        return

    # Explore
    explore_data(df)

    # Preprocess
    df_processed, label_encoders = preprocess_data(df)

    # Save cleaned dataset
    save_cleaned_dataset(df_processed)

    # Feature preparation (includes multicollinearity removal)
    ohe_cols, num_cols, poly_cols, X, y_raw, y_log = prepare_features_for_ml(df_processed)

    # Train model
    results = train_model(ohe_cols, num_cols, poly_cols, X, y_raw, y_log)

    # Visualise
    plot_all_visualizations(results, df)

    # Example prediction
    print("EXAMPLE PREDICTION")
    sample = df_processed.iloc[100]
    sample_features = {c: sample[c]
                       for c in (ohe_cols + num_cols)
                       if c in df_processed.columns}
    predict_arrival_time(
        results=results,
        features=sample_features,
        user_location_stop=str(sample.get('bus_stop', 'Unknown')),
        destination_stop='Next Stop',
        df_raw=df_processed,
        ohe_cols=ohe_cols,
        num_cols=num_cols,
    )

    primary_pipeline = results["Multiple Linear Regression"]["pipeline"]
    model_path = 'bus_arrival_mlr_model.pkl'
    joblib.dump({
        'model'          : primary_pipeline,
        'scaler'         : primary_pipeline.named_steps['preprocessor'],
        'feature_columns': ohe_cols + num_cols,
        'label_encoders' : label_encoders,
        'performance'    : results["Multiple Linear Regression"]['test_metrics'],
        'ohe_cols'       : ohe_cols,
        'num_cols'       : num_cols,
        'poly_cols'      : poly_cols,
    }, model_path)
    print(f"Model saved to: {model_path}")

    # Final summary
    tm = results["Multiple Linear Regression"]['test_metrics']
    print("\n" + "=" * 70)
    print("FINAL MODEL PERFORMANCE -- Multiple Linear Regression (log1p target)")
    print("=" * 70)
    print(f"   R²   : {tm['r2']:.4f}")
    print(f"   MAE  : {tm['mae']:.4f} minutes")
    print(f"   RMSE : {tm['rmse']:.4f} minutes")
    print(f"   MAPE : {tm['mape']:.2f} %")

    # Launch UI
    launch_bus_tracker_ui()

if __name__ == "__main__":
    main()