import pandas as pd
import numpy as np


def create_features(df_weekly, selected_kopi):

    # =========================
    # AMBIL TARGET KOPI
    # =========================
    df = df_weekly[
        ['minggu', selected_kopi]
    ].copy()

    # Rename target
    df = df.rename(columns={
        selected_kopi: 'Total'
    })

    # =========================
    # FEATURE ENGINEERING
    # =========================

    # Lag Features
    df['Lag_1'] = df['Total'].shift(1)
    df['Lag_2'] = df['Total'].shift(2)
    df['Lag_3'] = df['Total'].shift(3)
    df['Lag_4'] = df['Total'].shift(4)
    df['Lag_8'] = df['Total'].shift(8)

    # SMA
    df['SMA_4'] = (
        df['Total']
        .rolling(4)
        .mean()
    )

    df['SMA_8'] = (
        df['Total']
        .rolling(8)
        .mean()
    )

    # EMA
    df['EMA_4'] = (
        df['Total']
        .ewm(span=4, adjust=False)
        .mean()
    )

    df['EMA_8'] = (
        df['Total']
        .ewm(span=8, adjust=False)
        .mean()
    )

    # Momentum
    df['Momentum'] = (
        df['Total']
        - df['Total'].shift(1)
    )

    # MACD
    df['MACD_kopi'] = (
        df['EMA_4']
        - df['EMA_8']
    )

    # Trend
    df['Trend'] = np.arange(len(df))

    # =========================
    # HAPUS NULL
    # =========================
    df = df.dropna().reset_index(drop=True)

    return df