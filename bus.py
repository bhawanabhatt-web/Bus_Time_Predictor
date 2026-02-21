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
    print("STEP 2: DATA EXPLORATION")
    print(f"\n📊 Dataset Overview:")
    print(f"   Total records: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    print("\n📋 Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"   {col}: {dtype}")
    print("\n🔍 Missing Values:")
    missing = df.isnull().sum()
    for col, count in missing.items():
        status = f"{count} ({count/len(df)*100:.2f}%)" if count > 0 else "No missing values"
        print(f"   {col}: {status}")
    print("\n📈 Unique Value Counts:")
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].nunique() < 50:
            print(f"   {col}: {df[col].nunique()} unique values")
    print("\n📄 Sample Data (First 5 rows):")
    print(df.head())
    print("\n📊 Statistical Summary:")
    print(df.describe())

# 2. DATA PREPROCESSING
def preprocess_data(df):
    print("STEP 3: DATA PREPROCESSING")

    df_processed = df.copy()
    initial_count = len(df_processed)

    # 3.1 Handle Missing Values
    print("\n🔧 3.1: Handling Missing Values")
    missing_before = df_processed.isnull().sum().sum()
    print(f"   Total missing cells before cleaning: {missing_before}")

    # Fill numeric columns with median
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_processed[col].isnull().sum() > 0:
            median_val = df_processed[col].median()
            df_processed[col].fillna(median_val, inplace=True)
            print(f"   ✓ {col}: filled NaN with median = {median_val:.2f}")

    # Fill object/string columns with mode
    object_cols = df_processed.select_dtypes(include=['object']).columns
    for col in object_cols:
        if df_processed[col].isnull().sum() > 0:
            mode_val = df_processed[col].mode()[0]
            df_processed[col].fillna(mode_val, inplace=True)
            print(f"   ✓ {col}: filled NaN with mode = {mode_val}")

    missing_after = df_processed.isnull().sum().sum()
    print(f"   Total missing cells after cleaning: {missing_after}")

    # 3.2 Remove Outliers in dwell_time_in_seconds
    print("\n🔧 3.2: Removing Outliers (IQR method on dwell_time_in_seconds)")
    Q1 = df_processed['dwell_time_in_seconds'].quantile(0.25)
    Q3 = df_processed['dwell_time_in_seconds'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR
    upper = Q3 + 3 * IQR
    before = len(df_processed)
    df_processed = df_processed[
        (df_processed['dwell_time_in_seconds'] >= lower) &
        (df_processed['dwell_time_in_seconds'] <= upper)
    ]
    removed = before - len(df_processed)
    print(f"   IQR bounds: [{lower:.1f}, {upper:.1f}]")
    print(f"   ✓ Removed {removed:,} outlier rows ({removed/before*100:.2f}%)")
    print(f"   ✓ Remaining records: {len(df_processed):,}")

    # 3.3 Convert Datetime Columns
    print("\n🔧 3.3: Converting Date and Time Columns")
    df_processed['arrival_datetime'] = pd.to_datetime(
        df_processed['date'] + ' ' + df_processed['arrival_time'], errors='coerce'
    )
    df_processed['departure_datetime'] = pd.to_datetime(
        df_processed['date'] + ' ' + df_processed['departure_time'], errors='coerce'
    )
    df_processed = df_processed.dropna(subset=['arrival_datetime', 'departure_datetime'])
    print(f"   ✓ Valid datetime rows: {len(df_processed):,}")

    # 3.4 Sort Data
    df_processed = df_processed.sort_values(['trip_id', 'arrival_datetime']).reset_index(drop=True)
    print("\n🔧 3.4: Sorted data by trip_id and arrival_datetime ✓")

    # 3.5 Extract Time-Based Features
    print("\n🔧 3.5: Extracting Time-Based Features")
    df_processed['arrival_hour']          = df_processed['arrival_datetime'].dt.hour
    df_processed['arrival_minute']        = df_processed['arrival_datetime'].dt.minute
    df_processed['day_of_week']           = df_processed['arrival_datetime'].dt.dayofweek
    df_processed['month']                 = df_processed['arrival_datetime'].dt.month
    df_processed['day']                   = df_processed['arrival_datetime'].dt.day
    df_processed['seconds_since_midnight'] = (
        df_processed['arrival_hour'] * 3600 + df_processed['arrival_minute'] * 60
    )
    print("   ✓ arrival_hour, arrival_minute, day_of_week, month, day, seconds_since_midnight")

    # 3.6 Time Period Categories
    print("\n🔧 3.6: Creating Time Period Categories")
    def categorize_time_period(hour):
        if 5 <= hour < 9:   return 'morning_peak'
        elif 9 <= hour < 12:  return 'morning'
        elif 12 <= hour < 15: return 'afternoon'
        elif 15 <= hour < 19: return 'evening_peak'
        elif 19 <= hour < 22: return 'evening'
        else:                 return 'night'

    df_processed['time_period'] = df_processed['arrival_hour'].apply(categorize_time_period)
    df_processed['is_peak_hour'] = df_processed['time_period'].isin(
        ['morning_peak', 'evening_peak']).astype(int)
    df_processed['is_weekend']   = df_processed['day_of_week'].isin([5, 6]).astype(int)
    df_processed['is_weekday']   = (~df_processed['day_of_week'].isin([5, 6])).astype(int)
    print("   ✓ time_period, is_peak_hour, is_weekend, is_weekday created")

    # 3.7 Derived Features
    print("\n🔧 3.7: Calculating Derived Features")
    df_processed['actual_dwell_time'] = (
        df_processed['departure_datetime'] - df_processed['arrival_datetime']
    ).dt.total_seconds()

    df_processed['travel_time_to_next_stop'] = (
        df_processed.groupby('trip_id')['arrival_datetime']
        .shift(-1) - df_processed['arrival_datetime']
    ).dt.total_seconds() / 60
    print("   ✓ actual_dwell_time (seconds), travel_time_to_next_stop (minutes)")

    # 3.8 Encode Categorical Variables
    print("\n🔧 3.8: Label Encoding Categorical Variables")
    print("   Method: Label Encoding converts each unique category to an integer.")
    categorical_features = ['bus_stop', 'deviceid', 'time_period']
    label_encoders = {}
    for feature in categorical_features:
        if feature in df_processed.columns:
            le = LabelEncoder()
            df_processed[f'{feature}_encoded'] = le.fit_transform(df_processed[feature].astype(str))
            label_encoders[feature] = le
            print(f"   ✓ {feature}: {len(le.classes_)} unique classes encoded")

    df_processed['direction_encoded'] = df_processed['direction'].astype(int)
    print("   ✓ direction: already numeric (1 or 2), kept as-is")

    # 3.9 Lag Feature
    print("\n🔧 3.9: Creating Lag Features")
    df_processed['prev_travel_time'] = df_processed.groupby('trip_id')[
        'travel_time_to_next_stop'].shift(1)
    df_processed['prev_dwell_time']  = df_processed.groupby('trip_id')[
        'actual_dwell_time'].shift(1)
    print("   ✓ prev_travel_time, prev_dwell_time")

    # 3.10 Remove Rows with Missing Target
    print("\n🔧 3.10: Removing Rows with Missing Target Variable")
    df_processed = df_processed.dropna(subset=['travel_time_to_next_stop'])
    df_processed['prev_travel_time'].fillna(df_processed['travel_time_to_next_stop'].median(), inplace=True)
    df_processed['prev_dwell_time'].fillna(df_processed['actual_dwell_time'].median(), inplace=True)

    # Remove negative travel times (impossible)
    df_processed = df_processed[df_processed['travel_time_to_next_stop'] > 0]
    print(f"   ✓ Final dataset after cleaning: {len(df_processed):,} records")

    # 3.11 Min-Max Normalization of Numeric Features
    print("\n🔧 3.11: Min-Max Normalization of Continuous Features")
    print("   Formula: x_norm = (x - x_min) / (x_max - x_min)")
    norm_cols = ['actual_dwell_time', 'prev_travel_time', 'prev_dwell_time',
                 'seconds_since_midnight', 'dwell_time_in_seconds']
    minmax_scaler = MinMaxScaler()
    df_processed[[f'{c}_norm' for c in norm_cols]] = minmax_scaler.fit_transform(
        df_processed[norm_cols]
    )
    print(f"   ✓ Normalized: {norm_cols}")

    # 3.12 Feature Selection / Drop Low-Value Cols
    print("\n🔧 3.12: Feature Selection — Dropping Raw String/Datetime Columns")
    cols_to_drop = ['date', 'arrival_time', 'departure_time',
                    'arrival_datetime', 'departure_datetime',
                    'time_period', 'bus_stop', 'deviceid']
    cols_to_drop = [c for c in cols_to_drop if c in df_processed.columns]
    df_processed = df_processed.drop(columns=cols_to_drop)
    print(f"   ✓ Dropped: {cols_to_drop}")
    print(f"   ✓ Remaining features: {len(df_processed.columns)}")

    print(f"\n✅ Preprocessing Complete! Final shape: {df_processed.shape}")
    return df_processed, label_encoders, minmax_scaler

# 3. FEATURE PREPARATION
def prepare_features_for_ml(df_processed):
    print("\n" + "=" * 70)
    print("STEP 4: PREPARING FEATURES FOR MACHINE LEARNING")
    print("=" * 70)

    feature_columns = [
        'bus_stop_encoded', 'deviceid_encoded', 'time_period_encoded',
        'direction_encoded',
        'arrival_hour', 'arrival_minute', 'day_of_week', 'month', 'day',
        'seconds_since_midnight',
        'is_peak_hour', 'is_weekend', 'is_weekday',
        'actual_dwell_time', 'prev_travel_time', 'prev_dwell_time',
        'actual_dwell_time_norm', 'prev_travel_time_norm', 'prev_dwell_time_norm',
    ]

    feature_columns = [c for c in feature_columns if c in df_processed.columns]

    print(f"\n📋 Selected {len(feature_columns)} Features:")
    for i, f in enumerate(feature_columns, 1):
        print(f"   {i:2d}. {f}")

    X = df_processed[feature_columns].copy()
    y = df_processed['travel_time_to_next_stop'].copy()

    # Ensure no NaN in features
    X = X.fillna(X.median())

    print(f"\n✓ Feature matrix: {X.shape}, Target vector: {y.shape}")
    return feature_columns, X, y


# 4. MODEL TRAINING & COMPARATIVE ANALYSIS

def compute_regression_metrics(y_true, y_pred, label=""):
    mae   = mean_absolute_error(y_true, y_pred)
    mse   = mean_squared_error(y_true, y_pred)
    rmse  = np.sqrt(mse)
    r2    = r2_score(y_true, y_pred)
    mape  = mean_absolute_percentage_error(y_true, y_pred) * 100

    # Within-threshold accuracy 
    threshold = 2.0
    within_thresh = np.mean(np.abs(y_true - y_pred) <= threshold) * 100

    if label:
        print(f"\n   📊 {label}:")
    print(f"      MAE   (Mean Absolute Error)        : {mae:.4f} min")
    print(f"      MSE   (Mean Squared Error)          : {mse:.4f} min²")
    print(f"      RMSE  (Root Mean Squared Error)     : {rmse:.4f} min")
    print(f"      R²    (Coefficient of Determination): {r2:.4f}")
    print(f"      MAPE  (Mean Abs Percentage Error)   : {mape:.2f}%")
    print(f"      Accuracy within ±{threshold} min      : {within_thresh:.2f}%")

    return {
        'mae': mae, 'mse': mse, 'rmse': rmse,
        'r2': r2, 'mape': mape, 'within_threshold': within_thresh
    }

def train_and_compare_models(X, y, test_size=0.2, random_state=42):
    print("\n" + "=" * 70)
    print("STEP 5: MODEL TRAINING, EVALUATION & COMPARATIVE ANALYSIS")
    print("=" * 70)

    # ── Train-Test Split ──
    print("\n🔧 5.1: Train-Test Split (80% train / 20% test)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    print(f"   Training: {len(X_train):,} | Testing: {len(X_test):,}")

    # ── Feature Scaling (StandardScaler) ──
    print("\n🔧 5.2: Feature Scaling (StandardScaler: mean=0, std=1)")
    print("   Formula: z = (x - μ) / σ")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    print("   ✓ Features standardized")

    # ── Define Models ──
    models = {
        "Multiple Linear Regression": LinearRegression(),
        "Ridge Regression (α=1.0)": Ridge(alpha=1.0),
        "Decision Tree Regressor": DecisionTreeRegressor(
            max_depth=8, min_samples_split=20, random_state=random_state),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_split=20,
            n_jobs=-1, random_state=random_state),
    }

    print("\n🔧 5.3: Training and Evaluating All Models")
    results = {}

    for model_name, model in models.items():
        print(f"\n{'─'*55}")
        print(f"  MODEL: {model_name}")
        print(f"{'─'*55}")

        # Train
        model.fit(X_train_scaled, y_train)

        # Predict
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred  = model.predict(X_test_scaled)

        # Metrics
        train_metrics = compute_regression_metrics(y_train, y_train_pred, "Train Set")
        test_metrics  = compute_regression_metrics(y_test,  y_test_pred,  "Test Set")

        # Cross-Validation (5-Fold, MAE)
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(model, X_train_scaled, y_train,
                                     cv=cv, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores
        print(f"\n   📊 5-Fold Cross-Validation MAE:")
        print(f"      Mean: {cv_mae.mean():.4f} | Std: {cv_mae.std():.4f}")
        print(f"      Scores: {np.round(cv_mae, 3)}")

        results[model_name] = {
            'model': model,
            'scaler': scaler,
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'y_train_pred': y_train_pred, 'y_test_pred': y_test_pred,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_mae_mean': cv_mae.mean(),
            'cv_mae_std': cv_mae.std(),
        }

    # ── Comparative Summary ──
    print("\n" + "=" * 70)
    print("📊 COMPARATIVE MODEL PERFORMANCE SUMMARY (Test Set)")
    print("=" * 70)
    header = f"{'Model':<35} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'Acc±2min':>10} {'CV-MAE':>10}"
    print(header)
    print("-" * 80)
    best_model_name = None
    best_r2 = -np.inf
    for name, res in results.items():
        tm = res['test_metrics']
        row = (f"{name:<35} {tm['mae']:>8.4f} {tm['rmse']:>8.4f} "
               f"{tm['r2']:>8.4f} {tm['within_threshold']:>9.2f}% "
               f"{res['cv_mae_mean']:>10.4f}")
        print(row)
        if tm['r2'] > best_r2:
            best_r2 = tm['r2']
            best_model_name = name

    print(f"\n🏆 Best Model (highest R²): {best_model_name} (R² = {best_r2:.4f})")

    # ── Feature Importance for Linear Regression ──
    lr_model = results["Multiple Linear Regression"]['model']
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': lr_model.coef_,
        'abs_coefficient': np.abs(lr_model.coef_)
    }).sort_values('abs_coefficient', ascending=False)

    print("\n📊 Top 10 Feature Importances (MLR — by absolute coefficient):")
    for _, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:35s}: {row['coefficient']:+.4f}")

    results['feature_importance'] = feature_importance
    results['best_model_name'] = best_model_name
    results['primary_model_name'] = "Multiple Linear Regression"

    return results

# 5. PREDICTION FUNCTION

def predict_arrival_time(results, features, user_location_stop, destination_stop,
                          df_raw, feature_columns):
    """Predict bus arrival time using the primary model."""
    print("\n" + "=" * 70)
    print("STEP 6: MAKING PREDICTION FOR USER QUERY")
    print("=" * 70)
    print(f"\n📍 Journey: {user_location_stop} → {destination_stop}")

    model_data = results["Multiple Linear Regression"]
    feature_vector = pd.DataFrame([features], columns=feature_columns)
    feature_vector = feature_vector.fillna(feature_vector.median())
    feature_vector_scaled = model_data['scaler'].transform(feature_vector)
    predicted_time = model_data['model'].predict(feature_vector_scaled)[0]

    minutes = int(predicted_time)
    seconds = int((predicted_time - minutes) * 60)
    print(f"\n🔮 Estimated Travel Time: {minutes} min {seconds} sec")

    return {
        'predicted_time_minutes': predicted_time,
        'from_stop': user_location_stop,
        'to_stop': destination_stop
    }

# 6. VISUALIZATION
# ====================================================

def plot_all_visualizations(results, df_original):
    """Generate all evaluation and data visualizations."""
    print("\n📊 Generating visualizations...")

    primary = results["Multiple Linear Regression"]

    # ── Fig 1: Model Evaluation (MLR) ──
    fig1, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig1.suptitle("Multiple Linear Regression — Model Evaluation", fontsize=16, fontweight='bold')

    # Actual vs Predicted
    axes[0, 0].scatter(primary['y_test'], primary['y_test_pred'], alpha=0.3, s=5)
    lims = [primary['y_test'].min(), primary['y_test'].max()]
    axes[0, 0].plot(lims, lims, 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Travel Time (min)')
    axes[0, 0].set_ylabel('Predicted Travel Time (min)')
    axes[0, 0].set_title(f'Actual vs Predicted\nR² = {primary["test_metrics"]["r2"]:.4f}')
    axes[0, 0].grid(True, alpha=0.3)

    # Residuals
    residuals = primary['y_test'].values - primary['y_test_pred']
    axes[0, 1].scatter(primary['y_test_pred'], residuals, alpha=0.3, s=5)
    axes[0, 1].axhline(0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Travel Time (min)')
    axes[0, 1].set_ylabel('Residuals (min)')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)

    # Feature Importance
    top_f = results['feature_importance'].head(10)
    axes[1, 0].barh(range(len(top_f)), top_f['abs_coefficient'], color='steelblue')
    axes[1, 0].set_yticks(range(len(top_f)))
    axes[1, 0].set_yticklabels(top_f['feature'], fontsize=9)
    axes[1, 0].set_xlabel('|Coefficient|')
    axes[1, 0].set_title('Top 10 Feature Importances')
    axes[1, 0].grid(True, alpha=0.3, axis='x')

    # Error Distribution
    axes[1, 1].hist(residuals, bins=60, edgecolor='black', alpha=0.7, color='coral')
    axes[1, 1].axvline(0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Prediction Error (min)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Error Distribution\nMAE = {primary["test_metrics"]["mae"]:.3f} min')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: model_evaluation.png")
    plt.close(fig1)

    # ── Fig 2: Comparative Bar Chart ──
    model_names = [n for n in results.keys()
                   if n not in ('feature_importance', 'best_model_name', 'primary_model_name')]
    mae_vals  = [results[n]['test_metrics']['mae']  for n in model_names]
    rmse_vals = [results[n]['test_metrics']['rmse'] for n in model_names]
    r2_vals   = [results[n]['test_metrics']['r2']   for n in model_names]

    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    fig2.suptitle("Comparative Model Performance", fontsize=14, fontweight='bold')

    short_names = [n.replace(" Regressor", "").replace(" Regression", "") for n in model_names]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

    for ax, vals, title, ylabel in zip(
        axes2,
        [mae_vals, rmse_vals, r2_vals],
        ['MAE (lower is better)', 'RMSE (lower is better)', 'R² Score (higher is better)'],
        ['Minutes', 'Minutes', 'R² Score']
    ):
        bars = ax.bar(short_names, vals, color=colors[:len(model_names)])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(short_names, rotation=15, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: model_comparison.png")
    plt.close(fig2)

    # ── Fig 3: Data Distribution ──
    fig3, axes3 = plt.subplots(2, 3, figsize=(16, 10))
    fig3.suptitle("Bus Dataset — Data Distributions", fontsize=14, fontweight='bold')

    df_orig = df_original.copy()
    df_orig['date'] = pd.to_datetime(df_orig['date'], errors='coerce')
    df_orig['arrival_datetime'] = pd.to_datetime(
        df_orig['date'].astype(str) + ' ' + df_orig['arrival_time'], errors='coerce')
    df_orig['arrival_hour'] = df_orig['arrival_datetime'].dt.hour

    axes3[0, 0].hist(df_orig['dwell_time_in_seconds'].clip(0, 300), bins=50,
                     color='steelblue', edgecolor='black', alpha=0.8)
    axes3[0, 0].set_title('Dwell Time Distribution (seconds)')
    axes3[0, 0].set_xlabel('Dwell Time (s)')
    axes3[0, 0].set_ylabel('Frequency')
    axes3[0, 0].grid(True, alpha=0.3)

    hour_counts = df_orig['arrival_hour'].value_counts().sort_index()
    axes3[0, 1].bar(hour_counts.index, hour_counts.values, color='coral')
    axes3[0, 1].set_title('Bus Arrivals by Hour of Day')
    axes3[0, 1].set_xlabel('Hour')
    axes3[0, 1].set_ylabel('Count')
    axes3[0, 1].grid(True, alpha=0.3, axis='y')

    stop_counts = df_orig['bus_stop'].value_counts().head(15)
    axes3[0, 2].bar(stop_counts.index.astype(str), stop_counts.values, color='green', alpha=0.8)
    axes3[0, 2].set_title('Top 15 Bus Stops by Record Count')
    axes3[0, 2].set_xlabel('Stop ID')
    axes3[0, 2].set_ylabel('Count')
    axes3[0, 2].tick_params(axis='x', rotation=45)
    axes3[0, 2].grid(True, alpha=0.3, axis='y')

    direction_counts = df_orig['direction'].value_counts()
    axes3[1, 0].pie(direction_counts.values, labels=[f'Direction {d}' for d in direction_counts.index],
                    autopct='%1.1f%%', colors=['#2196F3', '#FF9800'])
    axes3[1, 0].set_title('Direction Distribution')

    day_counts = df_orig['arrival_datetime'].dt.dayofweek.value_counts().sort_index()
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes3[1, 1].bar([day_labels[i] for i in day_counts.index], day_counts.values, color='purple', alpha=0.8)
    axes3[1, 1].set_title('Arrivals by Day of Week')
    axes3[1, 1].set_xlabel('Day')
    axes3[1, 1].set_ylabel('Count')
    axes3[1, 1].grid(True, alpha=0.3, axis='y')

    axes3[1, 2].hist(df_orig['dwell_time_in_seconds'].clip(0, 300), bins=50,
                     cumulative=True, density=True, histtype='step', color='red', linewidth=2)
    axes3[1, 2].set_title('Cumulative Distribution of Dwell Time')
    axes3[1, 2].set_xlabel('Dwell Time (s)')
    axes3[1, 2].set_ylabel('Cumulative Probability')
    axes3[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: data_distribution.png")
    plt.close(fig3)

    # ── Fig 4: Correlation Heatmap ──
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    numeric_df = df_original.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax4, cbar_kws={'shrink': 0.8})
    ax4.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: correlation_heatmap.png")
    plt.close(fig4)

    print("   ✓ All visualizations saved successfully.")


# 8. MAIN PIPELINE

def main():
    print("\n" + "=" * 70)
    print("🚌  BUS ARRIVAL TIME PREDICTION SYSTEM")
    print("    Multiple Linear Regression | Comparative Analysis")
    print("=" * 70)

    # Load
    df = load_bus_data('bus_data.csv')
    if df is None:
        return

    # Explore
    explore_data(df)

    # Preprocess
    df_processed, label_encoders, minmax_scaler = preprocess_data(df)

    # Prepare Features
    feature_columns, X, y = prepare_features_for_ml(df_processed)

    # Train + Compare
    results = train_and_compare_models(X, y)

    # Visualize
    plot_all_visualizations(results, df)

    # Example Prediction
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTION")
    print("=" * 70)
    sample = df_processed.iloc[100]
    example_features = {col: sample[col] for col in feature_columns if col in df_processed.columns}
    prediction = predict_arrival_time(
        results=results,
        features=example_features,
        user_location_stop=str(int(sample.get('bus_stop_encoded', 0))),
        destination_stop='Next Stop',
        df_raw=df,
        feature_columns=feature_columns
    )

    # Save primary model
    print("\n" + "=" * 70)
    print("SAVING PRIMARY MODEL")
    print("=" * 70)
    primary = results["Multiple Linear Regression"]
    model_path = 'bus_arrival_mlr_model.pkl'
    joblib.dump({
        'model': primary['model'],
        'scaler': primary['scaler'],
        'feature_columns': feature_columns,
        'label_encoders': label_encoders,
        'performance': primary['test_metrics']
    }, model_path)
    print(f"✓ Model saved to: {model_path}")

    # Final Summary
    tm = primary['test_metrics']
    print("\n" + "=" * 70)
    print("✅ FINAL MODEL PERFORMANCE — Multiple Linear Regression")
    print("=" * 70)
    print(f"   MAE  : {tm['mae']:.4f} minutes")
    print(f"   RMSE : {tm['rmse']:.4f} minutes")
    print(f"   R²   : {tm['r2']:.4f}")
    print(f"   MAPE : {tm['mape']:.2f}%")
    print(f"   Accuracy within ±2 min: {tm['within_threshold']:.2f}%")
    print(f"\n   Best overall model: {results['best_model_name']}")

    
if __name__ == "__main__":
    main()