# utils/modeling.py

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# =========================================
# HITUNG MAPE
# =========================================
def calculate_mape(y_true, y_pred):

    mask = y_true != 0

    mape = np.mean(
        np.abs(
            (y_true[mask] - y_pred[mask])
            / y_true[mask]
        )
    ) * 100

    return mape


# =========================================
# TRAIN XGBOOST
# =========================================
def train_xgboost(feature_df):

    # =========================
    # FEATURE & TARGET
    # =========================
    X = feature_df.drop(
        columns=['minggu', 'Total']
    )

    y = feature_df['Total']

    # =========================
    # SPLIT DATA
    # =========================
    split = int(len(feature_df) * 0.8)

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    # =========================
    # PARAMETER GRID
    # =========================
    param_grid = {

        'n_estimators': [200, 400, 600, 800],

        'learning_rate': [
            0.01,
            0.03,
            0.05,
            0.1
        ],

        'max_depth': [3, 4, 5, 6, 8],

        'subsample': [
            0.7,
            0.8,
            0.9,
            1
        ],

        'colsample_bytree': [
            0.7,
            0.8,
            0.9,
            1
        ]
    }

    # =========================
    # MODEL
    # =========================
    xgb = XGBRegressor(
        random_state=42,
        n_jobs=1
    )

    # =========================
    # RANDOM SEARCH
    # =========================
    search = RandomizedSearchCV(

        estimator=xgb,

        param_distributions=param_grid,

        n_iter=20,

        scoring='neg_mean_absolute_error',

        cv=3,

        verbose=1,

        random_state=42,

        n_jobs=-1
    )

    # =========================
    # TRAINING 80%
    # =========================
    search.fit(X_train, y_train)

    best_params = search.best_params_

    best_model = search.best_estimator_

    # =========================
    # PREDICTION TEST
    # =========================
    pred = best_model.predict(X_test)

    # =========================
    # EVALUASI
    # =========================
    mae = mean_absolute_error(
        y_test,
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            pred
        )
    )

    mape = calculate_mape(
        y_test.values,
        pred
    )

    # =========================
    # FEATURE IMPORTANCE
    # =========================
    importance_df = pd.DataFrame({

        'Feature': X.columns,

        'Importance':
        best_model.feature_importances_

    })

    importance_df = importance_df.sort_values(

        by='Importance',

        ascending=False
    )

    # =========================
    # FINAL TRAINING 100%
    # =========================
    final_model = XGBRegressor(

        **best_params,

        random_state=42,

        n_jobs=1
    )

    final_model.fit(X, y)

    # =========================
    # RETURN
    # =========================
    return {

        # MODEL EVALUASI
        'model': best_model,

        # MODEL FINAL 100%
        'final_model': final_model,

        'best_params': best_params,

        'X_test': X_test,

        'y_test': y_test,

        'prediction': pred,

        'mae': mae,

        'rmse': rmse,

        'mape': mape,

        'importance_df': importance_df
    }


# =========================================
# TRAIN RANDOM FOREST
# =========================================
def train_random_forest(feature_df):

    # =========================
    # FEATURE & TARGET
    # =========================
    X = feature_df.drop(
        columns=['minggu', 'Total']
    )

    y = feature_df['Total']

    # =========================
    # SPLIT DATA
    # =========================
    split = int(len(feature_df) * 0.8)

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    # =========================
    # PARAMETER GRID
    # =========================
    param_grid = {

        'n_estimators': [200, 400, 600, 800],

        'max_depth': [
            5,
            10,
            15,
            20,
            None
        ],

        'min_samples_split': [
            2,
            5,
            10
        ],

        'min_samples_leaf': [
            1,
            2,
            4
        ],

        'max_features': [
            'sqrt',
            'log2'
        ]
    }

    # =========================
    # MODEL
    # =========================
    rf = RandomForestRegressor(
        random_state=42,
        n_jobs=1
    )

    # =========================
    # RANDOM SEARCH
    # =========================
    search = RandomizedSearchCV(

        estimator=rf,

        param_distributions=param_grid,

        n_iter=25,

        scoring='neg_mean_absolute_error',

        cv=3,

        random_state=42,

        n_jobs=-1,

        verbose=1
    )

    # =========================
    # TRAINING 80%
    # =========================
    search.fit(X_train, y_train)

    best_params = search.best_params_

    best_model = search.best_estimator_

    # =========================
    # PREDICTION TEST
    # =========================
    pred = best_model.predict(X_test)

    # =========================
    # EVALUASI
    # =========================
    mae = mean_absolute_error(
        y_test,
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            pred
        )
    )

    mape = calculate_mape(
        y_test.values,
        pred
    )

    # =========================
    # FEATURE IMPORTANCE
    # =========================
    importance_df = pd.DataFrame({

        'Feature': X.columns,

        'Importance':
        best_model.feature_importances_

    })

    importance_df = importance_df.sort_values(

        by='Importance',

        ascending=False
    )

    # =========================
    # FINAL TRAINING 100%
    # =========================
    final_model = RandomForestRegressor(

        **best_params,

        random_state=42,

        n_jobs=1
    )

    final_model.fit(X, y)

    # =========================
    # RETURN
    # =========================
    return {

        # MODEL EVALUASI
        'model': best_model,

        # MODEL FINAL 100%
        'final_model': final_model,

        'best_params': best_params,

        'X_test': X_test,

        'y_test': y_test,

        'prediction': pred,

        'mae': mae,

        'rmse': rmse,

        'mape': mape,

        'importance_df': importance_df
    }