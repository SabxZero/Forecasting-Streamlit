import pandas as pd
import numpy as np


def forecasting_4_weeks(
    feature_df,
    model
):

    # =====================================
    # COPY HISTORY
    # =====================================
    history = feature_df.copy()

    future_predictions = []

    # =====================================
    # FORECAST 4 MINGGU
    # =====================================
    for i in range(4):

        # =============================
        # AMBIL FEATURE TERAKHIR
        # =============================
        X_last = history.drop(
            columns=['minggu', 'Total']
        ).iloc[-1:]

        # =============================
        # PREDIKSI
        # =============================
        pred = model.predict(X_last)[0]

        future_predictions.append(pred)

        # =============================
        # NEXT DATE
        # =============================
        next_date = (
            history['minggu'].iloc[-1]
            + pd.Timedelta(weeks=1)
        )

        # =============================
        # TAMBAH ROW BARU
        # =============================
        new_row = pd.DataFrame({

            'minggu': [next_date],

            'Total': [pred]
        })

        history = pd.concat(
            [
                history[['minggu', 'Total']],
                new_row
            ],
            ignore_index=True
        )

        # =============================
        # FEATURE ENGINEERING ULANG
        # =============================
        history['Lag_1'] = (
            history['Total'].shift(1)
        )

        history['Lag_2'] = (
            history['Total'].shift(2)
        )

        history['Lag_3'] = (
            history['Total'].shift(3)
        )

        history['Lag_4'] = (
            history['Total'].shift(4)
        )

        history['Lag_8'] = (
            history['Total'].shift(8)
        )

        history['SMA_4'] = (
            history['Total']
            .rolling(4)
            .mean()
        )

        history['SMA_8'] = (
            history['Total']
            .rolling(8)
            .mean()
        )

        history['EMA_4'] = (
            history['Total']
            .ewm(span=4, adjust=False)
            .mean()
        )

        history['EMA_8'] = (
            history['Total']
            .ewm(span=8, adjust=False)
            .mean()
        )

        history['Momentum'] = (
            history['Total']
            - history['Total'].shift(1)
        )

        history['MACD_kopi'] = (
            history['EMA_4']
            - history['EMA_8']
        )

        history['Trend'] = np.arange(
            len(history)
        )

    # =====================================
    # FUTURE DATE
    # =====================================
    last_date = feature_df['minggu'].iloc[-1]

    future_start = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1),
        periods=4,
        freq='W-MON'
    )

    future_end = (
        future_start
        + pd.Timedelta(days=6)
    )

    # =====================================
    # FORECAST DATAFRAME
    # =====================================
    forecast_df = pd.DataFrame({

        'Minggu_ke': range(1, 5),

        'Tanggal_Mulai': future_start,

        'Tanggal_Akhir': future_end,

        'Forecast': np.round(
            future_predictions,
            0
        )
    })

    return forecast_df