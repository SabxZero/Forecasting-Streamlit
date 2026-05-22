# utils/visualization.py

import plotly.graph_objects as go
import plotly.express as px


# =========================================
# ACTUAL VS PREDICTION CHART
# =========================================
def actual_vs_prediction_chart(
    y_test,
    prediction,
    model_name
):

    fig = go.Figure()

    # =====================================
    # ACTUAL
    # =====================================
    fig.add_trace(
        go.Scatter(
            y=y_test,
            mode='lines+markers',
            name='Actual'
        )
    )

    # =====================================
    # PREDICTION
    # =====================================
    fig.add_trace(
        go.Scatter(
            y=prediction,
            mode='lines+markers',
            name='Prediction'
        )
    )

    # =====================================
    # LAYOUT
    # =====================================
    fig.update_layout(

        title=f'Actual vs Prediction - {model_name}',

        template='plotly_white',

        height=500,

        xaxis_title='Minggu',

        yaxis_title='Gram',

        hovermode='x unified'
    )

    return fig


# =========================================
# FEATURE IMPORTANCE CHART
# =========================================
def feature_importance_chart(
    importance_df,
    model_name
):

    fig = px.bar(

        importance_df,

        x='Importance',

        y='Feature',

        orientation='h',

        title=f'Feature Importance - {model_name}'
    )

    # =====================================
    # LAYOUT
    # =====================================
    fig.update_layout(

        template='plotly_white',

        height=500,

        yaxis=dict(
            categoryorder='total ascending'
        )
    )

    return fig


# =========================================
# FORECASTING CHART
# =========================================
def forecasting_chart(

    feature_df,

    forecast_df,

    selected_result,

    model_name
):

    fig = go.Figure()

    # =====================================
    # SPLITTING
    # =====================================
    total_data = len(feature_df)

    train_size = int(
        total_data * 0.8
    )

    # =====================================
    # TRAIN DATA
    # =====================================
    train_df = feature_df.iloc[
        :train_size
    ]

    # =====================================
    # TEST DATA
    # =====================================
    test_df = feature_df.iloc[
        train_size:
    ]

    # =====================================
    # HISTORICAL TRAIN
    # =====================================
    fig.add_trace(

        go.Scatter(

            x=train_df['minggu'],

            y=train_df['Total'],

            mode='lines+markers',

            name='Historical Data',

            line=dict(
                width=3,
                color='blue'
            )
        )
    )

    # =====================================
    # ACTUAL TEST
    # =====================================
    fig.add_trace(

        go.Scatter(

            x=test_df['minggu'],

            y=selected_result[
                'y_test'
            ],

            mode='lines+markers',

            name='Actual Test',

            line=dict(
                width=3,
                color='green'
            )
        )
    )

    # =====================================
    # PREDICTION TEST
    # =====================================
    fig.add_trace(

        go.Scatter(

            x=test_df['minggu'],

            y=selected_result[
                'prediction'
            ],

            mode='lines+markers',

            name='Prediction Model',

            line=dict(
                width=3,
                color='orange'
            )
        )
    )

    # =====================================
    # FORECAST FUTURE
    # =====================================
    fig.add_trace(

        go.Scatter(

            x=forecast_df[
                'Tanggal_Mulai'
            ],

            y=forecast_df[
                'Forecast'
            ],

            mode='lines+markers',

            name='Forecast Future',

            line=dict(
                width=4,
                color='red',
                dash='dash'
            )
        )
    )

    # =====================================
    # LAYOUT
    # =====================================
    fig.update_layout(

        title=f'''

        Forecasting 4 Minggu
        - {model_name}

        ''',

        template='plotly_white',

        height=600,

        xaxis_title='Tanggal',

        yaxis_title='Gram',

        hovermode='x unified',

        legend=dict(

            orientation='h',

            yanchor='bottom',

            y=1.02,

            xanchor='right',

            x=1
        )
    )

    return fig