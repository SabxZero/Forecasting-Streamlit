# app.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from streamlit_option_menu import option_menu

# =========================================
# IMPORT UTILS
# =========================================
from utils.preprocessing import (
    preprocessing_data,
    gram_mapping
)

from utils.feature_engineering import (
    create_features
)

from utils.modeling import (
    train_xgboost,
    train_random_forest
)

from utils.visualization import (
    actual_vs_prediction_chart,
    feature_importance_chart,
    forecasting_chart
)

from utils.forecasting import (
    forecasting_4_weeks
)

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Forecasting Kopi",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# LOAD CSS
# =========================================
with open("assets/style.css") as f:

    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# =========================================
# SESSION STATE
# =========================================
if "df_weekly" not in st.session_state:

    st.session_state.df_weekly = None

if "model_results" not in st.session_state:

    st.session_state.model_results = {}

# =========================================
# LIST JENIS KOPI
# =========================================
jenis_kopi = [

    "Robusta",
    "Robusta_Highblend",
    "Gayo",
    "Ciwidey",
    "Mandailing",
    "Kintamani",
    "Flores",
    "Papua",
    "Toraja"
]

# =========================================
# HELPER UI
# =========================================
def render_premium_table(
    df,
    height=None,
    compact=False,
    product=False
):
    table_class = "premium-table"
    wrapper_class = "premium-table-wrapper"

    if compact:
        table_class += " compact-table"
        wrapper_class += " compact-table-wrapper"

    if product:
        table_class += " product-table"
        wrapper_class += " product-table-wrapper"

    height_style = (
        f"max-height:{height}px; overflow-y:auto;"
        if height else ""
    )

    table_html = df.to_html(
        index=False,
        escape=False,
        classes=table_class
    )

    st.markdown(
        f"""
        <div class="{wrapper_class}" style="{height_style}">
            {table_html}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_metric_card(icon, title, value, subtitle):
    html = f"""
<div class="metric-card-custom">
    <div class="metric-icon-custom">{icon}</div>
    <div class="metric-content-custom">
        <div class="metric-title-custom">{title}</div>
        <div class="metric-value-custom">{value}</div>
        <div class="metric-subtitle-custom">{subtitle}</div>
    </div>
</div>
"""
    st.markdown(
        html,
        unsafe_allow_html=True
    )


def render_eval_metric_card(icon, title, value, subtitle):
    html = f"""
<div class="eval-metric-card">
    <div class="eval-metric-icon">{icon}</div>
    <div class="eval-metric-body">
        <div class="eval-metric-title">{title}</div>
        <div class="eval-metric-value">{value}</div>
        <div class="eval-metric-subtitle">{subtitle}</div>
    </div>
</div>
"""
    st.markdown(
        html,
        unsafe_allow_html=True
    )


def render_info_box(icon, title, rows):
    rows_html = ""

    for label, value in rows:
        rows_html += f"""
        <div class="eval-info-row">
            <span>{label}</span>
            <b>{value}</b>
        </div>
        """

    html = f"""
<div class="eval-info-card">
    <div class="eval-info-title">
        <span>{icon}</span>
        <h4>{title}</h4>
    </div>
    <div class="eval-info-content">
        {rows_html}
    </div>
</div>
"""
    st.markdown(
        html,
        unsafe_allow_html=True
    )



def create_time_series_component_chart(feature_df):

    component_df = feature_df[
        [
            "minggu",
            "Total"
        ]
    ].copy()

    component_df = component_df.sort_values(
        "minggu"
    ).reset_index(drop=True)

    component_df["Trend"] = (
        component_df["Total"]
        .rolling(
            window=8,
            center=True,
            min_periods=1
        )
        .mean()
    )

    detrended = (
        component_df["Total"]
        - component_df["Trend"]
    )

    seasonal_group = (
        pd.Series(
            range(len(component_df))
        )
        % 4
    )

    component_df["Seasonal"] = (
        detrended
        .groupby(seasonal_group)
        .transform("mean")
    )

    component_df["Residual"] = (
        component_df["Total"]
        - component_df["Trend"]
        - component_df["Seasonal"]
    )

    fig_component = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        subplot_titles=(
            "Trend",
            "Seasonal",
            "Residual"
        )
    )

    fig_component.add_trace(
        go.Scatter(
            x=component_df["minggu"],
            y=component_df["Trend"],
            mode="lines",
            name="Trend",
            line=dict(
                color="#4e2e1e",
                width=3
            )
        ),
        row=1,
        col=1
    )

    fig_component.add_trace(
        go.Scatter(
            x=component_df["minggu"],
            y=component_df["Seasonal"],
            mode="lines",
            name="Seasonal",
            line=dict(
                color="#5f8a5f",
                width=2.6
            )
        ),
        row=2,
        col=1
    )

    fig_component.add_trace(
        go.Scatter(
            x=component_df["minggu"],
            y=component_df["Residual"],
            mode="lines",
            name="Residual",
            line=dict(
                color="#e07a24",
                width=2.4
            )
        ),
        row=3,
        col=1
    )

    fig_component.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fffaf4",
        height=720,
        showlegend=False,
        hovermode="x unified",
        font=dict(
            family="Poppins",
            color="#4e2e1e",
            size=12
        ),
        margin=dict(
            l=72,
            r=40,
            t=70,
            b=70
        )
    )

    fig_component.update_annotations(
        font=dict(
            color="#4e2e1e",
            size=14,
            family="Poppins"
        )
    )

    fig_component.update_xaxes(
        title_text="Minggu",
        title_standoff=10,
        showgrid=True,
        gridcolor="rgba(92,51,23,0.10)",
        zeroline=False,
        linecolor="rgba(92,51,23,0.22)",
        tickfont=dict(
            color="#6b4b36",
            size=11
        ),
        title_font=dict(
            color="#4e2e1e",
            size=13
        ),
        automargin=True,
        row=3,
        col=1
    )

    fig_component.update_yaxes(
        title_text="Trend",
        title_standoff=12,
        showgrid=True,
        gridcolor="rgba(92,51,23,0.10)",
        zeroline=False,
        linecolor="rgba(92,51,23,0.22)",
        tickfont=dict(
            color="#6b4b36",
            size=11
        ),
        title_font=dict(
            color="#4e2e1e",
            size=13
        ),
        automargin=True,
        row=1,
        col=1
    )

    fig_component.update_yaxes(
        title_text="Seasonal",
        title_standoff=12,
        showgrid=True,
        gridcolor="rgba(92,51,23,0.10)",
        zeroline=False,
        linecolor="rgba(92,51,23,0.22)",
        tickfont=dict(
            color="#6b4b36",
            size=11
        ),
        title_font=dict(
            color="#4e2e1e",
            size=13
        ),
        automargin=True,
        row=2,
        col=1
    )

    fig_component.update_yaxes(
        title_text="Residual",
        title_standoff=12,
        showgrid=True,
        gridcolor="rgba(92,51,23,0.10)",
        zeroline=False,
        linecolor="rgba(92,51,23,0.22)",
        tickfont=dict(
            color="#6b4b36",
            size=11
        ),
        title_font=dict(
            color="#4e2e1e",
            size=13
        ),
        automargin=True,
        row=3,
        col=1
    )

    return fig_component


# =========================================
# SIDEBAR
# =========================================
with st.sidebar:

    # =====================================
    # LOGO
    # =====================================
    st.image(
        "assets/logo.png",
        use_container_width=True
    )

    st.write("")

    # =====================================
    # NAVIGASI
    # =====================================
    st.markdown(
        """
        <div class="sidebar-title">
        NAVIGASI
        </div>
        """,
        unsafe_allow_html=True
    )

    selected = option_menu(

        menu_title=None,

        options=[
            "Informasi Data",
            "Evaluasi Model",
            "Forecasting"
        ],

        icons=[
            "database-fill",
            "bar-chart-fill",
            "graph-up-arrow"
        ],

        default_index=0,

        styles={

            "container": {

                "padding": "16px",

                "background-color":
                "#252731",

                "border":
                "1px solid rgba(255,255,255,0.04)",

                "box-shadow":
                "0 10px 26px rgba(0,0,0,0.22)",

                "margin":
                "0px",

                "border-radius":
                "0px"
            },

            "icon": {

                "color": "#f5d7b2",

                "font-size": "20px"
            },

            "nav-link": {

                "font-size": "16px",

                "font-weight": "500",

                "text-align": "left",

                "margin": "4px 0",

                "padding": "14px 18px",

                "border-radius": "18px",

                "background":
                "transparent",

                "color": "#f5f5f5",

                "transition": "0.25s",

                "display": "flex",

                "align-items": "center",

                "gap": "10px"
            },

            "nav-link-selected": {

                "background":
                "linear-gradient(90deg, rgba(140,80,35,0.85), rgba(90,45,20,0.55))",

                "border":
                "1px solid rgba(255,255,255,0.05)",

                "box-shadow":
                "0 6px 18px rgba(0,0,0,0.22)",

                "color":
                "white"
            }
        }
    )

    st.write("")

    # =====================================
    # UPLOAD DATASET
    # =====================================
    st.markdown(
        """
        <div class="sidebar-title">
        UPLOAD DATASET
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload File Excel",
        type=["xlsx"],
        label_visibility="visible"
    )

    st.write("")

    # =====================================
    # PILIH KOPI
    # =====================================
    st.markdown(
        """
        <div class="sidebar-subtitle">
        Pilih Jenis Kopi
        </div>
        """,
        unsafe_allow_html=True
    )

    selected_kopi = st.selectbox(
        "",
        jenis_kopi
    )

    st.write("")

    # =====================================
    # INFO CARD
    # =====================================
    st.markdown(
        """
        <div class="info-sidebar-card">

        ☕ Forecasting kebutuhan bahan
        baku kopi untuk membantu
        perencanaan produksi yang
        lebih optimal dan efisien.

        </div>
        """,

        unsafe_allow_html=True
    )

# =========================================
# TITLE
# =========================================
# Title bawaan Streamlit dimatikan supaya halaman memakai custom header.
# if selected != "Informasi Data":
#     st.title(selected)

# =========================================
# VALIDASI DATASET
# =========================================
if uploaded_file is None:

    st.markdown(
        """
<div class="empty-upload-state">
    <div class="empty-upload-icon">☕</div>
    <div class="empty-upload-title">Silakan Upload Dataset Terlebih Dahulu</div>
    <div class="empty-upload-subtitle">
        Website prediksi kebutuhan bahan baku kopi PT Bogor Japutra Jaya
    </div>
</div>
""",
        unsafe_allow_html=True
    )

    st.stop()

# =========================================
# PREPROCESSING DATASET
# HANYA SEKALI
# =========================================
if st.session_state.df_weekly is None:

    loading_placeholder = st.empty()

    loading_placeholder.markdown(
        """
<div class="center-loading-state">
    <div class="coffee-loader">☕</div>
    <div class="center-loading-title">Memproses Dataset</div>
    <div class="center-loading-subtitle">
        Sistem sedang melakukan preprocessing data. Mohon tunggu sebentar.
    </div>
</div>
""",
        unsafe_allow_html=True
    )

    try:

        df_weekly = preprocessing_data(
            uploaded_file
        )

        st.session_state.df_weekly = (
            df_weekly
        )

        loading_placeholder.empty()

    except Exception:

        loading_placeholder.empty()

        st.markdown(
            """
<div class="dataset-error-state">
    <div class="dataset-error-icon">⚠️</div>
    <div class="dataset-error-title">Dataset Tidak Sesuai</div>
    <div class="dataset-error-subtitle">
        Pastikan file Excel yang diupload memiliki format dan kolom yang sesuai dengan kebutuhan sistem.
    </div>
</div>
""",
            unsafe_allow_html=True
        )

        st.stop()

# =========================================
# LOAD PREPROCESSING
# =========================================
df_weekly = st.session_state.df_weekly

# =========================================
# TRAIN MODEL PER KOPI
# HANYA JIKA BELUM ADA
# =========================================
if selected_kopi not in st.session_state.model_results:

    training_placeholder = st.empty()

    training_placeholder.markdown(
        f"""
<div class="center-loading-state">
    <div class="coffee-loader">☕</div>
    <div class="center-loading-title">Training Model {selected_kopi}</div>
    <div class="center-loading-subtitle">
        Sistem sedang melatih model XGBoost dan Random Forest. Mohon tunggu sebentar.
    </div>
</div>
""",
        unsafe_allow_html=True
    )

    try:

        # =================================
        # FEATURE ENGINEERING
        # =================================
        feature_df = create_features(

            df_weekly,

            selected_kopi
        )

        # =================================
        # TRAIN XGBOOST
        # =================================
        xgb_result = train_xgboost(
            feature_df
        )

        # =================================
        # TRAIN RANDOM FOREST
        # =================================
        rf_result = train_random_forest(
            feature_df
        )

        # =================================
        # PILIH MODEL TERBAIK
        # =================================
        if (
            xgb_result['mape']
            <
            rf_result['mape']
        ):

            best_model_name = (
                "XGBoost"
            )

            best_model = (
                xgb_result['model']
            )

            final_model = (
                xgb_result['final_model']
            )

            best_mape = (
                xgb_result['mape']
            )

        else:

            best_model_name = (
                "Random Forest"
            )

            best_model = (
                rf_result['model']
            )

            final_model = (
                rf_result['final_model']
            )

            best_mape = (
                rf_result['mape']
            )

        # =================================
        # FORECASTING
        # MODEL FINAL 100%
        # =================================
        forecast_df = forecasting_4_weeks(

            feature_df,

            final_model
        )

        # =================================
        # SAVE CACHE
        # =================================
        st.session_state.model_results[
            selected_kopi
        ] = {

            'feature_df':
            feature_df,

            'xgb_result':
            xgb_result,

            'rf_result':
            rf_result,

            'forecast_df':
            forecast_df,

            'best_model_name':
            best_model_name,

            'best_model':
            best_model,

            'final_model':
            final_model,

            'best_mape':
            best_mape
        }

        training_placeholder.empty()

    except Exception:

        training_placeholder.empty()

        st.markdown(
            """
<div class="dataset-error-state">
    <div class="dataset-error-icon">⚠️</div>
    <div class="dataset-error-title">Dataset Tidak Sesuai</div>
    <div class="dataset-error-subtitle">
        Data tidak dapat digunakan untuk proses training. Pastikan format dataset dan nilai transaksi sudah sesuai.
    </div>
</div>
""",
            unsafe_allow_html=True
        )

        st.stop()

# =========================================
# LOAD CACHE
# =========================================
result = st.session_state.model_results[
    selected_kopi
]

feature_df = result['feature_df']

xgb_result = result['xgb_result']

rf_result = result['rf_result']

forecast_df = result['forecast_df']

best_model_name = (
    result['best_model_name']
)

best_model = (
    result['best_model']
)

final_model = (
    result['final_model']
)

best_mape = (
    result['best_mape']
)

# =========================================
# HALAMAN INFORMASI DATA
# =========================================
if selected == "Informasi Data":

    # =====================================
    # SUMMARY DATA
    # =====================================
    df_raw = pd.read_excel(uploaded_file)

    jumlah_data_mentah = len(df_raw)

    jumlah_minggu = len(df_weekly)

    min_year = (
        pd.to_datetime(
            df_raw['Tanggal']
        ).dt.year.min()
    )

    max_year = (
        pd.to_datetime(
            df_raw['Tanggal']
        ).dt.year.max()
    )

    rentang_tahun = (
        f"{min_year} - {max_year}"
    )

    # =====================================
    # HEADER
    # =====================================
    col_header1, col_header2 = st.columns(
        [3, 1.5]
    )

    with col_header1:

        st.markdown(
            """
            <div style="
                color:#4e2e1e;
                font-size:56px;
                font-weight:700;
                margin-bottom:4px;
                line-height:1.1;
            ">
            ☕ Selamat Datang!
            </div>

            <div style="
                color:#7b5a45;
                font-size:21px;
                font-weight:500;
                margin-bottom:25px;
            ">
            Aplikasi Forecasting
            Kebutuhan Bahan Baku Kopi
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_header2:

        st.markdown(
            f"""
            <div style="
                background:
                linear-gradient(
                    135deg,
                    #8b5a2b,
                    #5c3317
                );
                color:white;
                border-radius:16px;
                padding:14px 18px;
                font-weight:600;
                text-align:center;
                box-shadow:
                0 6px 16px rgba(0,0,0,0.18);
            ">
            ☕ Kopi berkualitas, sejak 1975
            </div>
            """,
            unsafe_allow_html=True
        )
    st.write("")

    # =====================================
    # METRIC CARDS CUSTOM
    # =====================================
    c1, c2, c3 = st.columns(3)

    with c1:
        render_metric_card(
            "☕",
            "Jumlah Data Mentah",
            f"{jumlah_data_mentah:,}".replace(",", "."),
            "Baris transaksi"
        )

    with c2:
        render_metric_card(
            "📅",
            "Jumlah Minggu",
            jumlah_minggu,
            "Periode mingguan"
        )

    with c3:
        render_metric_card(
            "📅",
            "Rentang Tahun",
            rentang_tahun,
            "Dataset"
        )

    st.write("")
    # =====================================
    # PRODUCT INFO
    # =====================================
    product_rows = []

    for product_name, gram in gram_mapping.items():

        if (
            'OPLET' in product_name or
            'OBK' in product_name or
            'OSU' in product_name or
            'GULA' in product_name
        ):

            jenis = 'Robusta'

        elif 'PUSAKA' in product_name:

            jenis = 'Robusta_Highblend'

        elif 'KERIS' in product_name:

            jenis = (
                'Blend Robusta & Gayo'
            )

        elif 'GAYO' in product_name:

            jenis = 'Gayo'

        elif 'JAVA PREANGER' in product_name:

            jenis = 'Ciwidey'

        elif 'MANDHAILING' in product_name:

            jenis = 'Mandailing'

        elif 'KINTAMANI' in product_name:

            jenis = 'Kintamani'

        elif 'FLORES' in product_name:

            jenis = 'Flores'

        elif 'PAPUA' in product_name:

            jenis = 'Papua'

        elif 'TORAJA' in product_name:

            jenis = 'Toraja'

        else:

            jenis = '-'

        product_rows.append({

            'Jenis Kopi': jenis,

            'Nama Produk':
            product_name,

            'Takaran Gram':
            f"{gram} Gram"
        })

    product_info = pd.DataFrame(
        product_rows
    )

    # =====================================
    # LAYOUT
    # =====================================
    left_col, right_col = st.columns(
        [1.35, 1]
    )

    # =====================================
    # PRODUCT TABLE
    # =====================================
    with left_col:

        st.markdown(
            """
            <div style="
                color:#4e2e1e;
                font-size:30px;
                font-weight:700;
                margin-bottom:14px;
            ">
            ☕ Informasi Produk &
            Takaran Gram
            </div>
            """,
            unsafe_allow_html=True
        )

        render_premium_table(
            product_info,
            height=530,
            product=True
        )

    # =====================================
    # HEADER & FOOTER
    # =====================================
    with right_col:

        st.markdown(
            """
            <div style="
                color:#4e2e1e;
                font-size:30px;
                font-weight:700;
                margin-bottom:14px;
            ">
            📄 Header Data
            </div>
            """,
            unsafe_allow_html=True
        )

        render_premium_table(
            df_weekly.head(),
            height=215,
            compact=True
        )

        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style="
                color:#4e2e1e;
                font-size:30px;
                font-weight:700;
                margin-bottom:14px;
            ">
            📄 Footer Data
            </div>
            """,
            unsafe_allow_html=True
        )

        render_premium_table(
            df_weekly.tail(),
            height=215,
            compact=True
        )
# =========================================
# HALAMAN EVALUASI MODEL
# =========================================
elif selected == "Evaluasi Model":

    # =====================================
    # SPLITTING INFO
    # =====================================
    total_data = len(feature_df)

    train_size = int(
        total_data * 0.8
    )

    test_size = (
        total_data - train_size
    )

    # =====================================
    # DATE INFO
    # =====================================
    train_start = (
        feature_df.iloc[0]['minggu']
    )

    train_end = (
        feature_df.iloc[
            train_size - 1
        ]['minggu']
    )

    test_start = (
        feature_df.iloc[
            train_size
        ]['minggu']
    )

    test_end = (
        feature_df.iloc[
            total_data - 1
        ]['minggu']
    )

    # =====================================
    # HEADER EVALUASI
    # =====================================
    eval_header_left, eval_header_right = st.columns(
        [2.7, 1]
    )

    with eval_header_left:
        st.markdown(
            """
<div class="eval-page-title">
    <div class="eval-page-icon">▥</div>
    <div>
        <h1>Evaluasi Model</h1>
        <p>Menampilkan performa model forecasting kebutuhan bahan baku kopi</p>
    </div>
</div>
""",
            unsafe_allow_html=True
        )

    with eval_header_right:
        st.markdown(
            """
<div class="eval-select-label">
    Pilih Model
</div>
""",
            unsafe_allow_html=True
        )

        model_option = st.selectbox(
            "Pilih Model",
            [
                "XGBoost",
                "Random Forest"
            ],
            label_visibility="collapsed",
            key="eval_model_select"
        )

    # =====================================
    # SELECT MODEL
    # =====================================
    if model_option == "XGBoost":

        selected_result = xgb_result

    else:

        selected_result = rf_result

    mae_value = round(
        selected_result['mae'],
        2
    )

    rmse_value = round(
        selected_result['rmse'],
        2
    )

    mape_value = round(
        selected_result['mape'],
        2
    )

    # =====================================
    # TOP INFORMATION CARDS
    # =====================================
    top_info_1, top_info_2, top_info_3, top_info_4 = st.columns(4)

    with top_info_1:
        render_eval_metric_card(
            "☕",
            "Jenis Kopi",
            selected_kopi,
            "Kopi yang dievaluasi"
        )

    with top_info_2:
        render_eval_metric_card(
            "🏆",
            "Model",
            model_option,
            "Model yang dipilih"
        )

    with top_info_3:
        render_eval_metric_card(
            "📊",
            "Splitting Data",
            f"{train_size} : {test_size}",
            "Training 80% - Testing 20%"
        )

    with top_info_4:
        render_eval_metric_card(
            "📅",
            "Rentang Data",
            (
                f"{train_start.strftime('%d-%m-%Y')} "
                f"s/d {test_end.strftime('%d-%m-%Y')}"
            ),
            "Periode evaluasi model"
        )

    st.write("")

    # =====================================
    # ACTUAL VS PREDICTION - FULL WIDTH
    # =====================================
    st.markdown(
        """
<div class="eval-section-title">↗ Perbandingan Aktual vs Prediksi</div>
""",
        unsafe_allow_html=True
    )

    fig_actual = actual_vs_prediction_chart(

        selected_result['y_test'],

        selected_result['prediction'],

        model_option
    )

    line_colors = [
        "#4e2e1e",
        "#5f8a5f",
        "#e07a24"
    ]

    for idx, trace in enumerate(fig_actual.data):
        trace.update(
            line=dict(
                color=line_colors[idx % len(line_colors)],
                width=2.8
            ),
            marker=dict(
                size=5,
                color=line_colors[idx % len(line_colors)]
            )
        )

    fig_actual.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fffaf4",
        font=dict(
            family="Poppins",
            color="#4e2e1e",
            size=12
        ),
        title=dict(text=""),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0)",
            font=dict(
                color="#4e2e1e",
                size=12
            )
        ),
        margin=dict(
            l=72,
            r=40,
            t=45,
            b=86
        ),
        hovermode="x unified",
        height=500
    )

    fig_actual.update_xaxes(
        title_text="Minggu",
        title_standoff=10,
        showgrid=True,
        gridcolor="rgba(92,51,23,0.10)",
        zeroline=False,
        linecolor="rgba(92,51,23,0.22)",
        tickfont=dict(
            color="#6b4b36",
            size=12
        ),
        title_font=dict(
            color="#4e2e1e",
            size=14
        ),
        automargin=True
    )

    fig_actual.update_yaxes(
        title_text="Takaran Gram",
        title_standoff=12,
        showgrid=True,
        gridcolor="rgba(92,51,23,0.10)",
        zeroline=False,
        linecolor="rgba(92,51,23,0.22)",
        tickfont=dict(
            color="#6b4b36",
            size=12
        ),
        title_font=dict(
            color="#4e2e1e",
            size=14
        ),
        automargin=True
    )

    st.plotly_chart(

        fig_actual,

        use_container_width=True
    )

    st.write("")

    # =====================================
    # EVALUATION METRIC CARDS
    # =====================================
    eval_m1, eval_m2, eval_m3 = st.columns(3)

    with eval_m1:
        render_eval_metric_card(
            "↗",
            "MAE",
            f"{mae_value:,.2f}",
            "Error rata-rata"
        )

    with eval_m2:
        render_eval_metric_card(
            "〽",
            "RMSE",
            f"{rmse_value:,.2f}",
            "Root Mean Square Error"
        )

    with eval_m3:
        render_eval_metric_card(
            "%",
            "MAPE",
            f"{mape_value:.2f}%",
            "Mean Absolute Percentage Error"
        )

    metric_explanation_html = """
<div class="eval-explanation-card">
<div class="eval-explanation-title">ℹ️ Penjelasan Metrik Evaluasi</div>
<div class="eval-explanation-grid">
<div class="eval-explanation-item">
<b>MAE</b>
<span>Mengukur rata-rata selisih absolut antara nilai aktual dan prediksi. Semakin kecil nilai MAE, semakin kecil rata-rata kesalahan prediksi model.</span>
</div>
<div class="eval-explanation-item">
<b>RMSE</b>
<span>Mengukur besar error dengan memberikan penalti lebih besar pada kesalahan prediksi yang ekstrem. Semakin kecil RMSE, semakin stabil performa model.</span>
</div>
<div class="eval-explanation-item">
<b>MAPE</b>
<span>Mengukur persentase rata-rata kesalahan prediksi terhadap data aktual. Semakin kecil MAPE, semakin baik tingkat akurasi model.</span>
</div>
</div>
</div>
"""

    st.markdown(
        metric_explanation_html,
        unsafe_allow_html=True
    )

    st.write("")

    # =====================================
    # FEATURE IMPORTANCE + PERFORMANCE TABLE
    # =====================================
    bottom_left, bottom_right = st.columns(
        [1.25, 1]
    )

    with bottom_left:

        st.markdown(
            f"""
<div class="eval-section-title">⚙ Feature Importance ({model_option})</div>
""",
            unsafe_allow_html=True
        )

        fig_importance = (

            feature_importance_chart(

                selected_result[
                    'importance_df'
                ],

                model_option
            )
        )

        fig_importance.update_traces(
            marker=dict(
                color="#8b5a2b",
                line=dict(
                    color="#5c3317",
                    width=1
                )
            )
        )

        fig_importance.update_layout(
            title=dict(text=""),
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#fffaf4",
            font=dict(
                family="Poppins",
                color="#4e2e1e",
                size=12
            ),
            margin=dict(
                l=72,
                r=32,
                t=38,
                b=78
            ),
            height=455
        )

        fig_importance.update_xaxes(
            title_text="Importance",
            title_standoff=10,
            automargin=True,
            title_font=dict(
                color="#5c3317",
                size=14
            ),
            tickfont=dict(
                color="#6b3e26",
                size=12
            ),
            showgrid=True,
            gridcolor="rgba(92,51,23,0.10)",
            zeroline=False,
            linecolor="rgba(92,51,23,0.22)"
        )

        fig_importance.update_yaxes(
            title_text="Feature",
            title_standoff=4,
            automargin=True,
            title_font=dict(
                color="#5c3317",
                size=14
            ),
            tickfont=dict(
                color="#6b3e26",
                size=12
            ),
            showgrid=False,
            zeroline=False,
            linecolor="rgba(92,51,23,0.22)"
        )

        st.plotly_chart(

            fig_importance,

            use_container_width=True
        )

        with st.expander(
            "ℹ️ Lihat Penjelasan Feature Importance",
            expanded=False
        ):

            st.markdown(
                """<div class="feature-importance-note"><div class="feature-note-item"><b>Lag</b><span>Nilai kebutuhan bahan baku pada minggu sebelumnya. Fitur ini membantu model membaca pola historis.</span></div><div class="feature-note-item"><b>SMA</b><span>Rata-rata bergerak sederhana dalam beberapa minggu terakhir untuk melihat pola kebutuhan yang lebih stabil.</span></div><div class="feature-note-item"><b>EMA</b><span>Rata-rata bergerak yang memberi bobot lebih besar pada data terbaru sehingga lebih sensitif terhadap perubahan tren.</span></div><div class="feature-note-item"><b>Momentum</b><span>Perubahan kebutuhan bahan baku dari minggu sebelumnya untuk melihat kenaikan atau penurunan yang cepat.</span></div><div class="feature-note-item"><b>MACD</b><span>Selisih antara EMA jangka pendek dan EMA jangka panjang untuk membaca arah perubahan tren.</span></div><div class="feature-note-item"><b>Trend</b><span>Urutan waktu data dari awal sampai akhir periode untuk menangkap kecenderungan naik atau turun jangka panjang.</span></div><div class="feature-note-footer">Semakin besar nilai importance suatu fitur, semakin besar pengaruh fitur tersebut terhadap hasil prediksi model.</div></div>""",
                unsafe_allow_html=True
            )

    with bottom_right:

        st.markdown(
            """
<div class="eval-section-title">▥ Perbandingan Performa Model</div>
<div class="eval-table-area">
""",
            unsafe_allow_html=True
        )

        comparison_df = pd.DataFrame({

            "Model": [
                "XGBoost",
                "Random Forest"
            ],

            "MAE": [
                round(
                    xgb_result['mae'],
                    2
                ),
                round(
                    rf_result['mae'],
                    2
                )
            ],

            "RMSE": [
                round(
                    xgb_result['rmse'],
                    2
                ),
                round(
                    rf_result['rmse'],
                    2
                )
            ],

            "MAPE": [
                f"{round(xgb_result['mape'], 2)}%",
                f"{round(rf_result['mape'], 2)}%"
            ]
        })

        render_premium_table(
            comparison_df,
            height=245,
            compact=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

        best_eval_name = (
            "XGBoost"
            if xgb_result['mape'] < rf_result['mape']
            else "Random Forest"
        )

        best_eval_mape = (
            xgb_result['mape']
            if best_eval_name == "XGBoost"
            else rf_result['mape']
        )

        recommended_model = (
            "XGBoost"
            if xgb_result['mape'] < rf_result['mape']
            else "Random Forest"
        )

        recommended_mape = (
            xgb_result['mape']
            if recommended_model == "XGBoost"
            else rf_result['mape']
        )

        comparison_model = (
            "Random Forest"
            if recommended_model == "XGBoost"
            else "XGBoost"
        )

        performance_note_html = f"""
<div class="eval-explanation-card performance-note-card">
<div class="eval-explanation-title">✅ Kesimpulan Performa Model</div>
<div class="eval-performance-text">
Berdasarkan hasil evaluasi menggunakan nilai <b>MAPE</b>, sistem merekomendasikan model <b>{recommended_model}</b> untuk memprediksi kebutuhan bahan baku kopi <b>{selected_kopi}</b>. Model ini dipilih karena menghasilkan nilai MAPE paling kecil, yaitu <b>{recommended_mape:.2f}%</b>, sehingga tingkat kesalahan persentasenya lebih rendah dibandingkan model <b>{comparison_model}</b>. Oleh karena itu, model <b>{recommended_model}</b> dapat digunakan sebagai acuan untuk forecasting kebutuhan bahan baku selama <b>4 minggu ke depan</b> agar perencanaan produksi lebih tepat dan efisien.
</div>
</div>
"""

        st.markdown(
            performance_note_html,
            unsafe_allow_html=True
        )


    # =====================================
    # PARAMETER MODEL
    # =====================================
    st.markdown(
        """
<div class="eval-section-title">⚙ Parameter Model</div>
""",
        unsafe_allow_html=True
    )

    if model_option == "XGBoost":

        parameter_grid_df = pd.DataFrame({

            "Parameter": [
                "n_estimators",
                "learning_rate",
                "max_depth",
                "subsample",
                "colsample_bytree"
            ],

            "Nilai yang Diuji": [
                "200, 400, 600, 800",
                "0.01, 0.03, 0.05, 0.1",
                "3, 4, 5, 6, 8",
                "0.7, 0.8, 0.9, 1",
                "0.7, 0.8, 0.9, 1"
            ],

            "Keterangan": [
                "Jumlah pohon/iterasi boosting",
                "Kecepatan model belajar",
                "Kedalaman maksimum tiap pohon",
                "Proporsi data yang digunakan tiap pohon",
                "Proporsi fitur yang digunakan tiap pohon"
            ]
        })

    else:

        parameter_grid_df = pd.DataFrame({

            "Parameter": [
                "n_estimators",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "max_features"
            ],

            "Nilai yang Diuji": [
                "200, 400, 600, 800",
                "5, 10, 15, 20, None",
                "2, 5, 10",
                "1, 2, 4",
                "sqrt, log2"
            ],

            "Keterangan": [
                "Jumlah pohon dalam model",
                "Kedalaman maksimum tiap pohon",
                "Jumlah minimum sampel untuk membagi node",
                "Jumlah minimum sampel pada daun",
                "Jumlah fitur yang dipertimbangkan tiap split"
            ]
        })

    parameter_order = parameter_grid_df[
        "Parameter"
    ].tolist()

    best_params_dict = selected_result[
        "best_params"
    ]

    best_parameter_rows = []

    for param in parameter_order:

        best_parameter_rows.append({

            "Parameter": param,

            "Hasil Parameter Terbaik": str(
                best_params_dict.get(
                    param,
                    "-"
                )
            )
        })

    best_parameter_df = pd.DataFrame(
        best_parameter_rows
    )

    param_left, param_right = st.columns(
        [1.25, 1]
    )

    with param_left:

        st.markdown(
            """
<div class="eval-param-card-title">Parameter yang Digunakan</div>
""",
            unsafe_allow_html=True
        )

        render_premium_table(
            parameter_grid_df,
            height=330,
            compact=True
        )

    with param_right:

        st.markdown(
            f"""
<div class="eval-param-card-title">Parameter Terbaik {model_option}</div>
""",
            unsafe_allow_html=True
        )

        render_premium_table(
            best_parameter_df,
            height=330,
            compact=True
        )

        st.markdown(
            f"""
<div class="eval-soft-note">
    ✅ Parameter terbaik dipilih otomatis menggunakan RandomizedSearchCV berdasarkan nilai error terkecil pada data training.
</div>
""",
            unsafe_allow_html=True
        )


    st.write("")


# =========================================
# HALAMAN FORECASTING
# =========================================
elif selected == "Forecasting":

    # =====================================
    # REKOMENDASI MODEL
    # Berdasarkan MAPE terkecil dari evaluasi
    # =====================================
    if xgb_result['mape'] < rf_result['mape']:

        recommended_model_name = "XGBoost"

    else:

        recommended_model_name = "Random Forest"

    # =====================================
    # PILIH MODEL FORECASTING
    # User bisa memilih XGBoost atau Random Forest
    # Rekomendasi tetap ditampilkan berdasarkan MAPE terbaik
    # =====================================
    model_forecast_option = st.session_state.get(
        "model_forecast_option",
        recommended_model_name
    )

    if model_forecast_option not in [
        "XGBoost",
        "Random Forest"
    ]:

        model_forecast_option = recommended_model_name

    selected_forecast_result = (
        xgb_result
        if model_forecast_option == "XGBoost"
        else rf_result
    )

    forecast_df = forecasting_4_weeks(
        feature_df,
        selected_forecast_result["final_model"]
    )

    # =====================================
    # SUMMARY FORECAST
    # =====================================
    forecast_period = 4

    total_forecast = (
        forecast_df['Forecast']
        .sum()
    )

    avg_forecast = (
        forecast_df['Forecast']
        .mean()
    )

    # =====================================
    # KONVERSI KEBUTUHAN KARUNG
    # Robusta & Robusta Highblend = 80 kg / karung
    # Arabika = 50 kg / karung
    # =====================================
    robusta_category = [
        "Robusta",
        "Robusta_Highblend"
    ]

    arabika_category = [
        "Gayo",
        "Ciwidey",
        "Mandailing",
        "Kintamani",
        "Flores",
        "Papua",
        "Toraja"
    ]

    if selected_kopi in robusta_category:

        kategori_kopi = "Robusta"

        karung_kg = 80

    elif selected_kopi in arabika_category:

        kategori_kopi = "Arabika"

        karung_kg = 50

    else:

        kategori_kopi = "Lainnya"

        karung_kg = 50

    total_kg = (
        total_forecast / 1000
    )

    estimasi_karung = (
        total_kg / karung_kg
    )

    estimasi_karung_bulat = int(
        -(-total_kg // karung_kg)
    )

    # =====================================
    # HEADER FORECASTING
    # =====================================
    forecast_header_left, forecast_header_right = st.columns(
        [2.1, 1.35]
    )

    with forecast_header_left:

        st.markdown(
            """
<div class="forecast-page-title">
    <div class="forecast-page-icon">↗</div>
    <div>
        <h1>Forecasting Kebutuhan</h1>
        <p>Prediksi kebutuhan bahan baku kopi untuk periode mendatang</p>
    </div>
</div>
""",
            unsafe_allow_html=True
        )

    with forecast_header_right:

        forecast_control_left, forecast_control_right = st.columns(
            [1, 1],
            gap="small"
        )

        with forecast_control_left:

            st.markdown(
                f"""
<div class="forecast-model-badge forecast-model-badge-small">
    <div class="forecast-badge-title">
        ☕ Rekomendasi Model:
    </div>
    <div class="forecast-badge-model">
        {recommended_model_name}
    </div>
</div>
""",
                unsafe_allow_html=True
            )

        with forecast_control_right:

            st.markdown(
                """
<div class="forecast-select-label forecast-select-label-small">
    Pilih Model
</div>
""",
                unsafe_allow_html=True
            )

            model_forecast_option = st.selectbox(
                label="Pilih Model Forecasting",
                options=[
                    "XGBoost",
                    "Random Forest"
                ],
                index=0 if model_forecast_option == "XGBoost" else 1,
                label_visibility="collapsed",
                key="model_forecast_option"
            )

        selected_forecast_result = (
            xgb_result
            if model_forecast_option == "XGBoost"
            else rf_result
        )

        forecast_df = forecasting_4_weeks(
            feature_df,
            selected_forecast_result["final_model"]
        )

        total_forecast = forecast_df["Forecast"].sum()
        avg_forecast = forecast_df["Forecast"].mean()

    # =====================================
    # TOP FORECAST CARDS
    # =====================================
    f1, f2, f3, f4 = st.columns(4)

    with f1:
        render_eval_metric_card(
            "☕",
            "Jenis Kopi",
            selected_kopi,
            f"Kategori {kategori_kopi}"
        )

    with f2:
        render_eval_metric_card(
            "📅",
            "Periode Forecast",
            f"{forecast_period} Minggu",
            "Mingguan"
        )

    with f3:
        render_eval_metric_card(
            "↗",
            "Total 1 Bulan",
            f"{total_forecast:,.0f}",
            "Gram"
        )

    with f4:
        render_eval_metric_card(
            "📈",
            "Rata-rata / Minggu",
            f"{avg_forecast:,.0f}",
            "Gram"
        )

    st.write("")

    # =====================================
    # FORECASTING CHART FULL WIDTH
    # =====================================
    st.markdown(
        """
<div class="forecast-section-title">↗ Forecasting 4 Minggu ke Depan</div>
""",
        unsafe_allow_html=True
    )

    fig_forecast = forecasting_chart(

        feature_df,

        forecast_df,

        selected_forecast_result,

        model_forecast_option
    )

    # =====================================
    # WARNA CHART FORECASTING
    # Actual Test = coklat muda
    # Prediction Model = hijau
    # Forecast Future = orange
    # Historical Data = coklat tua
    # =====================================
    for trace in fig_forecast.data:

        trace_name = str(trace.name).lower()

        if "historical" in trace_name or "data aktual" in trace_name:
            color = "#4e2e1e"

        elif "actual test" in trace_name or "actual" in trace_name:
            color = "#8b5a2b"

        elif "prediction model" in trace_name or "prediction" in trace_name or "prediksi" in trace_name:
            color = "#5f8a5f"

        elif "forecast future" in trace_name or "forecast" in trace_name:
            color = "#e07a24"

        else:
            color = "#8b5a2b"

        if hasattr(trace, "line"):
            trace.update(
                line=dict(
                    color=color,
                    width=2.8,
                    dash=trace.line.dash if trace.line and trace.line.dash else None
                )
            )

        if hasattr(trace, "marker"):
            trace.update(
                marker=dict(
                    size=5,
                    color=color
                )
            )

        if hasattr(trace, "fillcolor") and trace.fillcolor:
            trace.update(
                fillcolor="rgba(224,122,36,0.16)"
            )

    fig_forecast.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fffaf4",
        font=dict(
            family="Poppins",
            color="#4e2e1e",
            size=12
        ),
        title=dict(text=""),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0)",
            font=dict(
                color="#4e2e1e",
                size=12
            )
        ),
        margin=dict(
            l=72,
            r=40,
            t=45,
            b=86
        ),
        hovermode="x unified",
        height=520
    )

    fig_forecast.update_xaxes(
        title_text="Minggu",
        title_standoff=10,
        showgrid=True,
        gridcolor="rgba(92,51,23,0.10)",
        zeroline=False,
        linecolor="rgba(92,51,23,0.22)",
        tickfont=dict(
            color="#6b4b36",
            size=12
        ),
        title_font=dict(
            color="#4e2e1e",
            size=14
        ),
        automargin=True
    )

    fig_forecast.update_yaxes(
        title_text="Takaran Gram",
        title_standoff=12,
        showgrid=True,
        gridcolor="rgba(92,51,23,0.10)",
        zeroline=False,
        linecolor="rgba(92,51,23,0.22)",
        tickfont=dict(
            color="#6b4b36",
            size=12
        ),
        title_font=dict(
            color="#4e2e1e",
            size=14
        ),
        automargin=True
    )

    st.plotly_chart(
        fig_forecast,
        use_container_width=True
    )

    st.markdown(
        f"""
<div class="forecast-soft-note">
    ℹ️ Grafik menampilkan data aktual historis dan hasil prediksi kebutuhan bahan baku kopi
    menggunakan model <b>{model_forecast_option}</b> untuk 4 minggu ke depan.
</div>
""",
        unsafe_allow_html=True
    )

    # =====================================
    # TIME SERIES COMPONENT CHART
    # Ditempatkan persis di bawah grafik forecasting utama
    # =====================================
    st.markdown(
        """
<div class="forecast-section-title component-title">📈 Komponen Time Series</div>
""",
        unsafe_allow_html=True
    )

    fig_component = create_time_series_component_chart(
        feature_df
    )

    st.plotly_chart(
        fig_component,
        use_container_width=True
    )

    st.markdown(
        """
<div class="forecast-soft-note component-soft-note">
    ℹ️ Grafik komponen time series memisahkan data menjadi <b>Trend</b>,
    <b>Seasonal</b>, dan <b>Residual</b>. Trend menunjukkan arah perubahan
    kebutuhan bahan baku, Seasonal menunjukkan pola berulang, sedangkan
    Residual menunjukkan fluktuasi acak yang tidak dijelaskan oleh pola utama.
</div>
""",
        unsafe_allow_html=True
    )

    st.write("")

    # =====================================
    # FORECAST TABLE + INSIGHT
    # =====================================
    table_col, insight_col = st.columns(
        [1.15, 1]
    )

    # =====================================
    # SIAPKAN TABEL FORECASTING
    # Tabel berisi tanggal awal - akhir tiap minggu
    # =====================================
    forecast_display = forecast_df.copy()

    if "minggu" in forecast_display.columns:

        forecast_dates = pd.to_datetime(
            forecast_display["minggu"]
        )

    elif "Minggu" in forecast_display.columns:

        forecast_dates = pd.to_datetime(
            forecast_display["Minggu"]
        )

    else:

        last_week = pd.to_datetime(
            feature_df["minggu"]
        ).max()

        forecast_dates = pd.date_range(
            start=last_week + pd.Timedelta(weeks=1),
            periods=len(forecast_display),
            freq="W-MON"
        )

    forecast_display["Tanggal Awal"] = (
        forecast_dates
        .dt.strftime("%d %b %Y")
        if hasattr(forecast_dates, "dt")
        else pd.Series(forecast_dates).dt.strftime("%d %b %Y")
    )

    forecast_display["Tanggal Akhir"] = (
        (forecast_dates + pd.Timedelta(days=6))
        .dt.strftime("%d %b %Y")
        if hasattr(forecast_dates, "dt")
        else pd.Series(forecast_dates + pd.Timedelta(days=6)).dt.strftime("%d %b %Y")
    )

    forecast_display["Minggu"] = [
        f"Minggu ke-{i+1}"
        for i in range(len(forecast_display))
    ]

    if "Forecast" in forecast_display.columns:

        forecast_display["Prediksi (Gram)"] = (
            forecast_display["Forecast"]
            .round(0)
            .astype(int)
        )

    forecast_table = forecast_display[
        [
            "Minggu",
            "Tanggal Awal",
            "Tanggal Akhir",
            "Prediksi (Gram)"
        ]
    ]

    with table_col:

        st.markdown(
            """
<div class="forecast-section-title">📅 Hasil Forecasting 4 Minggu</div>
""",
            unsafe_allow_html=True
        )

        render_premium_table(
            forecast_table,
            height=330,
            compact=True
        )

        st.markdown(
            f"""
<div class="forecast-soft-note">
    ✅ Total kebutuhan bahan baku kopi <b>{selected_kopi}</b> selama 4 minggu ke depan
    diperkirakan sebesar <b>{total_forecast:,.0f} gram</b>.
</div>
""",
            unsafe_allow_html=True
        )

    with insight_col:

        st.markdown(
            """
<div class="forecast-section-title">💡 Insight Forecasting</div>
""",
            unsafe_allow_html=True
        )

        insight_html = f"""
<div class="forecast-insight-card">
<div class="forecast-insight-row">
<div class="forecast-insight-icon">🎯</div>
<div>Grafik dan tabel forecasting saat ini menampilkan model <b>{model_forecast_option}</b>. Model rekomendasi sistem adalah <b>{recommended_model_name}</b> karena memiliki nilai MAPE terbaik.</div>
</div>
<div class="forecast-insight-row">
<div class="forecast-insight-icon">📦</div>
<div>Estimasi total kebutuhan bahan baku kopi <b>{selected_kopi}</b> selama 4 minggu ke depan adalah <b>{total_forecast:,.0f} gram</b>.</div>
</div>
<div class="forecast-insight-row">
<div class="forecast-insight-icon">📊</div>
<div>Rata-rata kebutuhan bahan baku per minggu diperkirakan sebesar <b>{avg_forecast:,.0f} gram</b>.</div>
</div>
<div class="forecast-insight-row">
<div class="forecast-insight-icon">✅</div>
<div>Kebutuhan biji kopi <b>{selected_kopi}</b> selama 1 bulan diperkirakan setara dengan sekitar <b>{estimasi_karung:.1f} karung</b>. Untuk kebutuhan operasional, disarankan menyiapkan sekitar <b>{estimasi_karung_bulat} karung</b>.</div>
</div>
</div>
"""

        st.markdown(
            insight_html,
            unsafe_allow_html=True
        )

    st.markdown(
        f"""
<div class="forecast-footer-note">
    ☕ Konversi karung mengikuti kategori kopi:
    Robusta = 80 kg per karung, sedangkan Arabika = 50 kg per karung.
</div>
""",
        unsafe_allow_html=True
    )
