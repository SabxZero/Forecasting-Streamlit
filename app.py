# app.py

import streamlit as st
import pandas as pd
from datetime import datetime
import math
import hashlib
import warnings
from io import BytesIO
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from streamlit_option_menu import option_menu

# =========================================
# HIDE STREAMLIT SESSION STATE DUPLICATION WARNING
# =========================================
warnings.filterwarnings(
    "ignore",
    message=r".*was created with a default value but also had its value set via the Session State API.*"
)

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

# Menyembunyikan warning kuning Streamlit ketika key widget juga tersimpan di Session State.
try:
    st.set_option("global.disableWidgetStateDuplicationWarning", True)
except Exception:
    pass

# =========================================
# LOAD CSS
# =========================================
with open("assets/style.css", "r", encoding="utf-8") as f:

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

if "autosaved_forecast_keys" not in st.session_state:

    st.session_state.autosaved_forecast_keys = set()


def get_uploaded_file_signature(uploaded_file):
    """Membuat identitas dataset agar cache reset saat file berubah."""
    if uploaded_file is None:
        return None

    try:
        file_bytes = uploaded_file.getvalue()
    except Exception:
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)

    file_hash = hashlib.md5(file_bytes).hexdigest()
    return f"{uploaded_file.name}_{len(file_bytes)}_{file_hash}"

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




# =========================================
# DATABASE SUPABASE - RIWAYAT FORECASTING
# =========================================
# Histori forecasting disimpan penuh ke Supabase PostgreSQL.


def get_karung_kg(jenis_kopi):
    if jenis_kopi in ["Robusta", "Robusta_Highblend"]:
        return 80

    return 50


def get_lead_time(jenis_kopi):
    if jenis_kopi in ["Robusta", "Robusta_Highblend"]:
        return 3

    return 1


def calculate_inventory_metrics(feature_df, jenis_kopi):
    historical_demand = (
        feature_df["Total"]
        .astype(float)
        .dropna()
    )

    z_score = 1.65
    lead_time = get_lead_time(jenis_kopi)
    std_demand = historical_demand.std()
    avg_demand = historical_demand.mean()

    safety_stock = (
        z_score
        * std_demand
        * math.sqrt(lead_time)
    )

    minimum_stock = (
        avg_demand
        * lead_time
        + safety_stock
    )

    maximum_stock = (
        2
        * avg_demand
        * lead_time
        + safety_stock
    )

    return {
        "z_score": z_score,
        "lead_time": lead_time,
        "std_demand": float(std_demand),
        "avg_demand": float(avg_demand),
        "safety_stock": float(safety_stock),
        "minimum_stock": float(minimum_stock),
        "maximum_stock": float(maximum_stock),
    }


def convert_stock_to_gram(value, unit, karung_kg):
    if unit == "Gram":
        return float(value)

    if unit == "Kilogram":
        return float(value) * 1000

    return float(value) * karung_kg * 1000



def parse_stock_input_text(value):
    """Parse input angka format Indonesia, contoh 20.000 atau 1,5."""
    if value is None:
        return 0.0

    value_text = str(value).strip()

    if not value_text:
        return 0.0

    cleaned = (
        value_text
        .replace(" ", "")
        .replace(".", "")
        .replace(",", ".")
    )

    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def format_stock_number_id(value):
    """Format angka menjadi pemisah ribuan titik tanpa ,00 di belakang."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "0"

    if number.is_integer():
        return f"{int(number):,}".replace(",", ".")

    formatted = f"{number:,.2f}"
    formatted = formatted.replace(",", "X").replace(".", ",").replace("X", ".")
    formatted = formatted.rstrip("0").rstrip(",")
    return formatted


def normalize_stock_input_text(key):
    """Dipakai on_change text_input agar input 20000 otomatis menjadi 20.000."""
    raw_value = st.session_state.get(key, "")
    parsed_value = parse_stock_input_text(raw_value)
    st.session_state[key] = format_stock_number_id(parsed_value)


def get_coffee_sort_order(coffee_type):
    order_map = {
        "Robusta": 1,
        "Robusta_Highblend": 2,
        "Gayo": 3,
        "Ciwidey": 4,
        "Mandailing": 5,
        "Kintamani": 6,
        "Flores": 7,
        "Papua": 8,
        "Toraja": 9,
    }

    return order_map.get(str(coffee_type), 999)



def get_model_sort_order(model_name):
    order_map = {
        "XGBoost": 1,
        "Random Forest": 2,
    }
    return order_map.get(str(model_name), 999)

def format_database_coffee_name(coffee_type):
    return str(coffee_type).replace("_", " ")



def get_supabase_config():
    """
    Mengambil konfigurasi Supabase dari Streamlit secrets.
    Jika secrets belum diisi, sistem otomatis fallback ke SQLite lokal.
    """
    try:
        supabase_url = st.secrets.get("SUPABASE_URL", "")
        supabase_key = st.secrets.get("SUPABASE_KEY", "")
    except Exception:
        supabase_url = ""
        supabase_key = ""

    if supabase_url and supabase_key:
        return (
            str(supabase_url).rstrip("/"),
            str(supabase_key)
        )

    return None, None


def is_supabase_enabled():
    supabase_url, supabase_key = get_supabase_config()
    return bool(supabase_url and supabase_key)


def get_supabase_headers(prefer=None):
    _, supabase_key = get_supabase_config()

    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    if prefer:
        headers["Prefer"] = prefer

    return headers


def init_forecast_db():
    """
    Supabase tidak membutuhkan inisialisasi tabel dari aplikasi.
    Tabel forecast_summary dibuat dan dikelola melalui dashboard Supabase.
    """
    return


def require_supabase_config():
    supabase_url, supabase_key = get_supabase_config()

    if not supabase_url or not supabase_key:
        st.error(
            "Konfigurasi Supabase belum ditemukan. "
            "Pastikan file .streamlit/secrets.toml berisi SUPABASE_URL dan SUPABASE_KEY."
        )
        st.stop()

    return supabase_url, supabase_key




def is_database_connection_error(exc):
    text = str(exc).lower()
    keywords = [
        "connection",
        "connectionerror",
        "max retries",
        "failed to resolve",
        "getaddrinfo",
        "name resolution",
        "timeout",
        "timed out",
        "network",
        "supabase"
    ]
    return any(keyword in text for keyword in keywords)


def render_database_connection_error():
    st.markdown(
        """
<div class="database-error-friendly">
    <div class="database-error-icon">⚠️</div>
    <div>
        <div class="database-error-title">Terjadi error / tidak bisa terhubung ke database :(</div>
        <div class="database-error-text">Coba periksa internet anda dan coba refresh.</div>
    </div>
</div>
""",
        unsafe_allow_html=True
    )


def raise_database_connection_error():
    raise RuntimeError(
        "Terjadi error / tidak bisa terhubung ke database :( Coba periksa internet anda dan coba refresh."
    )


def safe_supabase_get(endpoint, **kwargs):
    try:
        return requests.get(endpoint, **kwargs)
    except requests.exceptions.RequestException:
        raise_database_connection_error()


def safe_supabase_post(endpoint, **kwargs):
    try:
        return requests.post(endpoint, **kwargs)
    except requests.exceptions.RequestException:
        raise_database_connection_error()


def safe_supabase_patch(endpoint, **kwargs):
    try:
        return requests.patch(endpoint, **kwargs)
    except requests.exceptions.RequestException:
        raise_database_connection_error()


def safe_supabase_delete(endpoint, **kwargs):
    try:
        return requests.delete(endpoint, **kwargs)
    except requests.exceptions.RequestException:
        raise_database_connection_error()

def get_latest_stock_for_coffee(jenis_kopi, before_forecast_date=None):
    """
    Mengambil stok terakhir untuk jenis kopi tertentu dari Supabase.
    Jika before_forecast_date diisi, maka stok yang diambil adalah stok terakhir
    sebelum tanggal forecast tersebut.
    """
    supabase_url, _ = get_supabase_config()

    if not supabase_url:
        return 0.0

    endpoint = f"{supabase_url}/rest/v1/forecast_summary"

    request_params = [
        ("select", "stock,forecast_date,created_at"),
        ("coffee_type", f"eq.{jenis_kopi}"),
        ("order", "forecast_date.desc,created_at.desc"),
        ("limit", "1")
    ]

    if before_forecast_date:
        request_params.append(("forecast_date", f"lt.{before_forecast_date}"))

    response = safe_supabase_get(
        endpoint,
        headers=get_supabase_headers(),
        params=request_params,
        timeout=20
    )

    if response.status_code != 200:
        return 0.0

    data = response.json()

    if not data:
        return 0.0

    stock_value = data[0].get("stock", 0)

    try:
        return float(stock_value) if stock_value is not None else 0.0
    except Exception:
        return 0.0


def get_existing_stock_for_forecast_period(jenis_kopi, forecast_date):
    """
    Mengecek apakah stock untuk kombinasi jenis kopi + forecast_date sudah ada.
    Ini mencegah stok berkurang dua kali ketika XGBoost dan Random Forest
    disimpan pada tanggal forecast yang sama.
    """
    supabase_url, _ = get_supabase_config()

    if not supabase_url:
        return None

    endpoint = f"{supabase_url}/rest/v1/forecast_summary"

    request_params = [
        ("select", "stock,forecast_date,created_at"),
        ("coffee_type", f"eq.{jenis_kopi}"),
        ("forecast_date", f"eq.{forecast_date}"),
        ("order", "created_at.desc"),
        ("limit", "1")
    ]

    response = safe_supabase_get(
        endpoint,
        headers=get_supabase_headers(),
        params=request_params,
        timeout=20
    )

    if response.status_code != 200:
        return None

    data = response.json()

    if not data:
        return None

    stock_value = data[0].get("stock", None)

    if stock_value is None:
        return None

    try:
        return float(stock_value)
    except Exception:
        return None


def calculate_stock_for_forecast_record(jenis_kopi, forecast_date, actual_total):
    """
    Menghitung stock yang akan disimpan pada record forecast baru.

    Aturan:
    1. Jika forecast_date untuk kopi tersebut sudah punya stock, gunakan nilai itu.
       Tujuannya agar model kedua pada tanggal yang sama tidak mengurangi stock lagi.
    2. Jika belum ada, ambil stock terakhir sebelum forecast_date.
    3. Stock baru = stock sebelumnya - total aktual terbaru.
    4. Jika belum ada stock sebelumnya, nilai stock dianggap 0 gram.
    """
    existing_stock = get_existing_stock_for_forecast_period(
        jenis_kopi,
        forecast_date
    )

    if existing_stock is not None:
        return round(existing_stock, 2)

    previous_stock = get_latest_stock_for_coffee(
        jenis_kopi,
        before_forecast_date=forecast_date
    )

    current_stock = max(
        previous_stock - float(actual_total or 0),
        0
    )

    return round(current_stock, 2)


def update_latest_stock_for_coffee(jenis_kopi, new_stock_gram, model_name=None):
    """
    Mengubah stock pada seluruh model forecasting terbaru untuk jenis kopi yang dipilih.

    Catatan:
    Stock adalah kondisi gudang, sehingga nilainya harus sama untuk XGBoost
    dan Random Forest pada tanggal forecasting terbaru. Parameter model_name
    tetap diterima agar kompatibel dengan pemanggilan lama, tetapi tidak dipakai
    sebagai filter update.
    """
    supabase_url, _ = require_supabase_config()
    endpoint = f"{supabase_url}/rest/v1/forecast_summary"

    latest_params = [
        ("select", "forecast_date"),
        ("coffee_type", f"eq.{jenis_kopi}"),
        ("order", "forecast_date.desc,created_at.desc"),
        ("limit", "1")
    ]

    latest_response = safe_supabase_get(
        endpoint,
        headers=get_supabase_headers(),
        params=latest_params,
        timeout=20
    )

    if latest_response.status_code != 200:
        raise RuntimeError(
            f"Gagal membaca stock terbaru: {latest_response.status_code} - {latest_response.text}"
        )

    latest_data = latest_response.json()

    if not latest_data:
        raise RuntimeError(
            "Belum ada data forecasting untuk jenis kopi ini. Jalankan forecasting terlebih dahulu."
        )

    latest_forecast_date = latest_data[0]["forecast_date"]

    update_params = [
        ("coffee_type", f"eq.{jenis_kopi}"),
        ("forecast_date", f"eq.{latest_forecast_date}")
    ]

    update_response = safe_supabase_patch(
        endpoint,
        headers=get_supabase_headers(prefer="return=minimal"),
        params=update_params,
        json={
            "stock": round(float(new_stock_gram), 2)
        },
        timeout=20
    )

    if update_response.status_code not in [200, 204]:
        raise RuntimeError(
            f"Gagal memperbarui stock: {update_response.status_code} - {update_response.text}"
        )

    try:
        st.cache_data.clear()
    except Exception:
        pass

    st.session_state["inventory_refresh_token"] = datetime.now().timestamp()

    return latest_forecast_date

def build_forecast_payload(jenis_kopi, model_name, forecast_df, feature_df):
    """
    Menyiapkan payload yang sama untuk SQLite maupun Supabase.
    Supabase menggunakan nama kolom: week1, week2, week3, week4.
    SQLite lokal menggunakan nama kolom: week1_gram, week2_gram, dst.
    """
    forecast_data = forecast_df.copy()

    if "minggu" in forecast_data.columns:
        forecast_dates = pd.to_datetime(forecast_data["minggu"])

    elif "Minggu" in forecast_data.columns:
        forecast_dates = pd.to_datetime(forecast_data["Minggu"])

    else:
        last_week = pd.to_datetime(feature_df["minggu"]).max()

        forecast_dates = pd.date_range(
            start=last_week + pd.Timedelta(weeks=1),
            periods=len(forecast_data),
            freq="W-MON"
        )

    forecast_values = forecast_data["Forecast"].astype(float).round(2).tolist()

    while len(forecast_values) < 4:
        forecast_values.append(0.0)

    week1_gram = float(forecast_values[0])
    week2_gram = float(forecast_values[1])
    week3_gram = float(forecast_values[2])
    week4_gram = float(forecast_values[3])

    total_week1 = week1_gram
    total_week2 = week1_gram + week2_gram
    total_week3 = week1_gram + week2_gram + week3_gram
    total_week4 = week1_gram + week2_gram + week3_gram + week4_gram

    if "Total" in feature_df.columns and len(feature_df) > 0:
        actual_total = float(feature_df["Total"].iloc[-1])
    else:
        actual_total = 0.0

    inventory_metrics = calculate_inventory_metrics(
        feature_df,
        jenis_kopi
    )

    safety_stock = inventory_metrics["safety_stock"]
    minimum_stock = inventory_metrics["minimum_stock"]
    maximum_stock = inventory_metrics["maximum_stock"]

    karung_kg = get_karung_kg(jenis_kopi)

    recommended_sacks = int(
        -(-(total_week4 / 1000) // karung_kg)
    )

    forecast_date = pd.to_datetime(
        forecast_dates.iloc[0]
        if hasattr(forecast_dates, "iloc")
        else forecast_dates[0]
    ).strftime("%Y-%m-%d")

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    stock = calculate_stock_for_forecast_record(
        jenis_kopi,
        forecast_date,
        actual_total
    )

    return {
        "forecast_date": forecast_date,
        "coffee_type": jenis_kopi,
        "model_name": model_name,
        "actual_total": round(actual_total, 2),

        "week1_gram": round(week1_gram, 2),
        "week2_gram": round(week2_gram, 2),
        "week3_gram": round(week3_gram, 2),
        "week4_gram": round(week4_gram, 2),

        "total_week1": round(total_week1, 2),
        "total_week2": round(total_week2, 2),
        "total_week3": round(total_week3, 2),
        "total_week4": round(total_week4, 2),

        "safety_stock": round(safety_stock, 2),
        "minimum_stock": round(minimum_stock, 2),
        "maximum_stock": round(maximum_stock, 2),
        "stock": round(stock, 2),

        "recommended_sacks": recommended_sacks,
        "created_at": created_at
    }


def save_forecast_to_supabase(payload):
    supabase_url, _ = get_supabase_config()

    if not supabase_url:
        return False

    supabase_payload = {
        "forecast_date": payload["forecast_date"],
        "coffee_type": payload["coffee_type"],
        "model_name": payload["model_name"],
        "actual_total": payload["actual_total"],

        "week1": payload["week1_gram"],
        "week2": payload["week2_gram"],
        "week3": payload["week3_gram"],
        "week4": payload["week4_gram"],

        "total_week1": payload["total_week1"],
        "total_week2": payload["total_week2"],
        "total_week3": payload["total_week3"],
        "total_week4": payload["total_week4"],

        "safety_stock": payload["safety_stock"],
        "minimum_stock": payload["minimum_stock"],
        "maximum_stock": payload["maximum_stock"],
        "stock": payload.get("stock", 0),

        "created_at": payload["created_at"]
    }

    endpoint = f"{supabase_url}/rest/v1/forecast_summary"

    response = safe_supabase_post(
        endpoint,
        headers=get_supabase_headers(
            prefer="resolution=merge-duplicates,return=minimal"
        ),
        params={
            "on_conflict": "forecast_date,coffee_type,model_name"
        },
        json=supabase_payload,
        timeout=20
    )

    if response.status_code not in [200, 201, 204]:
        raise RuntimeError(
            f"Gagal menyimpan data ke Supabase: {response.status_code} - {response.text}"
        )

    return True



def save_forecast_to_db(jenis_kopi, model_name, forecast_df, feature_df):
    """
    Menyimpan histori forecasting langsung ke Supabase PostgreSQL.
    SQLite lokal tidak lagi digunakan.
    """
    require_supabase_config()

    payload = build_forecast_payload(
        jenis_kopi,
        model_name,
        forecast_df,
        feature_df
    )

    save_forecast_to_supabase(payload)

    # Setelah forecast baru tersimpan, bersihkan cache supaya menu Persediaan
    # tidak menampilkan stock/forecast dari minggu sebelumnya.
    try:
        st.cache_data.clear()
    except Exception:
        pass


def load_forecast_history_from_supabase(jenis_kopi=None, tahun=None, model_name=None):
    supabase_url, _ = get_supabase_config()

    if not supabase_url:
        return pd.DataFrame()

    endpoint = f"{supabase_url}/rest/v1/forecast_summary"

    params = {
        "select": "*",
        "order": "forecast_date.asc,created_at.asc"
    }

    if jenis_kopi:
        params["coffee_type"] = f"eq.{jenis_kopi}"

    if model_name:
        params["model_name"] = f"eq.{model_name}"

    if tahun:
        params["forecast_date"] = (
            f"gte.{tahun}-01-01"
            f"&forecast_date=lte.{tahun}-12-31"
        )

    # Karena requests params tidak bisa mengirim key forecast_date dua kali
    # dengan nama yang sama secara langsung dalam dict, gunakan list tuple.
    request_params = [
        ("select", "*"),
        ("order", "forecast_date.asc,created_at.asc")
    ]

    if jenis_kopi:
        request_params.append(("coffee_type", f"eq.{jenis_kopi}"))

    if model_name:
        request_params.append(("model_name", f"eq.{model_name}"))

    if tahun:
        request_params.append(("forecast_date", f"gte.{tahun}-01-01"))
        request_params.append(("forecast_date", f"lte.{tahun}-12-31"))

    response = safe_supabase_get(
        endpoint,
        headers=get_supabase_headers(),
        params=request_params,
        timeout=20
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Gagal membaca data dari Supabase: {response.status_code} - {response.text}"
        )

    data = response.json()
    df_history = pd.DataFrame(data)

    if df_history.empty:
        return df_history

    rename_map = {
        "week1": "week1_gram",
        "week2": "week2_gram",
        "week3": "week3_gram",
        "week4": "week4_gram"
    }

    df_history = df_history.rename(columns=rename_map)

    for col in [
        "actual_total",
        "week1_gram",
        "week2_gram",
        "week3_gram",
        "week4_gram",
        "total_week1",
        "total_week2",
        "total_week3",
        "total_week4",
        "safety_stock",
        "minimum_stock",
        "maximum_stock",
        "stock"
    ]:
        if col in df_history.columns:
            df_history[col] = pd.to_numeric(
                df_history[col],
                errors="coerce"
            ).fillna(0)

    if "created_at" not in df_history.columns:
        df_history["created_at"] = ""

    return df_history



def get_total_column_by_period(periode):
    if periode == "Minggu 1":
        return "total_week1"

    if periode == "Minggu 2":
        return "total_week2"

    if periode == "Minggu 3":
        return "total_week3"

    return "total_week4"


def load_forecast_history(jenis_kopi=None, periode="Minggu 4", tahun=None, model_name=None):
    """
    Membaca histori forecasting langsung dari Supabase PostgreSQL.
    SQLite lokal tidak lagi digunakan.
    """
    require_supabase_config()

    df_history = load_forecast_history_from_supabase(
        jenis_kopi=jenis_kopi,
        tahun=tahun,
        model_name=model_name
    )

    if not df_history.empty:
        total_col = get_total_column_by_period(periode)

        df_history["selected_total"] = df_history[total_col]
        df_history["selected_safety_stock"] = df_history["safety_stock"]
        df_history["selected_minimum_stock"] = df_history["minimum_stock"]
        df_history["selected_maximum_stock"] = df_history["maximum_stock"]

    return df_history


def load_all_forecast_records():
    """
    Mengambil seluruh isi tabel forecast_summary dari Supabase
    untuk halaman Kelola Database.
    """
    supabase_url, _ = require_supabase_config()
    endpoint = f"{supabase_url}/rest/v1/forecast_summary"

    request_params = [
        ("select", "*"),
        ("order", "forecast_date.asc,coffee_type.asc,model_name.asc")
    ]

    response = safe_supabase_get(
        endpoint,
        headers=get_supabase_headers(),
        params=request_params,
        timeout=20
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Gagal membaca data database Supabase: {response.status_code} - {response.text}"
        )

    data = response.json()
    df = pd.DataFrame(data)

    if df.empty:
        return df

    for col in [
        "id",
        "actual_total",
        "week1",
        "week2",
        "week3",
        "week4",
        "total_week1",
        "total_week2",
        "total_week3",
        "total_week4",
        "safety_stock",
        "minimum_stock",
        "maximum_stock",
        "stock"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def delete_forecast_records(record_ids):
    """
    Menghapus satu atau beberapa row berdasarkan id di Supabase.
    """
    if not record_ids:
        return

    supabase_url, _ = require_supabase_config()
    endpoint = f"{supabase_url}/rest/v1/forecast_summary"

    id_values = ",".join(str(int(record_id)) for record_id in record_ids)

    response = safe_supabase_delete(
        endpoint,
        headers=get_supabase_headers(prefer="return=minimal"),
        params={
            "id": f"in.({id_values})"
        },
        timeout=20
    )

    if response.status_code not in [200, 202, 204]:
        raise RuntimeError(
            f"Gagal menghapus data Supabase: {response.status_code} - {response.text}"
        )


def dataframe_to_excel_bytes(df):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(
            writer,
            index=False,
            sheet_name="forecast_summary"
        )

    output.seek(0)
    return output.getvalue()


def prepare_database_display_df(df):
    if df.empty:
        return df

    display_df = df.copy()

    column_order = [
        "id",
        "forecast_date",
        "coffee_type",
        "model_name",
        "actual_total",
        "week1",
        "week2",
        "week3",
        "week4",
        "total_week1",
        "total_week2",
        "total_week3",
        "total_week4",
        "safety_stock",
        "minimum_stock",
        "maximum_stock",
        "stock",
        "created_at"
    ]

    existing_columns = [
        col for col in column_order
        if col in display_df.columns
    ]

    display_df = display_df[existing_columns]

    if "forecast_date" in display_df.columns:
        display_df["forecast_date"] = pd.to_datetime(
            display_df["forecast_date"],
            errors="coerce"
        ).dt.strftime("%d-%m-%Y")

    if "created_at" in display_df.columns:
        display_df["created_at"] = pd.to_datetime(
            display_df["created_at"],
            errors="coerce"
        ).dt.strftime("%d-%m-%Y %H:%M")

    if "coffee_type" in display_df.columns:
        display_df["coffee_type"] = display_df["coffee_type"].apply(format_database_coffee_name)

    numeric_columns = [
        "actual_total",
        "week1",
        "week2",
        "week3",
        "week4",
        "total_week1",
        "total_week2",
        "total_week3",
        "total_week4",
        "safety_stock",
        "minimum_stock",
        "maximum_stock",
        "stock"
    ]

    for col in numeric_columns:
        if col in display_df.columns:
            display_df[col] = (
                pd.to_numeric(display_df[col], errors="coerce")
                .fillna(0)
                .round(0)
                .astype(int)
                .map("{:,}".format)
            )

    rename_columns = {
        "id": "ID",
        "forecast_date": "Tanggal Forecast",
        "coffee_type": "Biji Kopi",
        "model_name": "Model",
        "actual_total": "Total Aktual",
        "week1": "Minggu 1",
        "week2": "Minggu 2",
        "week3": "Minggu 3",
        "week4": "Minggu 4",
        "total_week1": "Total Minggu 1",
        "total_week2": "Total Minggu 2",
        "total_week3": "Total Minggu 3",
        "total_week4": "Total Minggu 4",
        "safety_stock": "Safety Stock",
        "minimum_stock": "Minimum Stock",
        "maximum_stock": "Maximum Stock",
        "stock": "Stock",
        "created_at": "Created At"
    }

    display_df = display_df.rename(columns=rename_columns)

    return display_df


def show_delete_confirmation_dialog(record_ids):
    if hasattr(st, "dialog"):
        @st.dialog(" ")
        def delete_dialog():
            st.markdown(
                f"""
<div class="delete-dialog-content refined-delete-dialog-content">
    <div class="delete-dialog-icon refined-delete-dialog-icon">🗑️</div>
    <div class="delete-dialog-title refined-delete-dialog-title">Yakin ingin hapus data?</div>
    <div class="delete-dialog-subtitle refined-delete-dialog-subtitle">
        Sebanyak {len(record_ids)} row akan dihapus
    </div>
</div>
""",
                unsafe_allow_html=True
            )

            confirm_col, cancel_col = st.columns(2)

            with confirm_col:
                if st.button(
                    "Hapus",
                    type="secondary",
                    use_container_width=True,
                    key="confirm_delete_database_rows"
                ):
                    delete_forecast_records(record_ids)
                    st.rerun()

            with cancel_col:
                if st.button(
                    "Batal",
                    type="secondary",
                    use_container_width=True,
                    key="cancel_delete_database_rows"
                ):
                    st.rerun()

        delete_dialog()

    else:
        st.warning(
            "Versi Streamlit belum mendukung popup dialog. "
            "Gunakan tombol konfirmasi di bawah ini untuk menghapus data."
        )

        confirm_col, cancel_col = st.columns(2)

        with confirm_col:
            if st.button(
                "Hapus",
                type="secondary",
                use_container_width=True,
                key="fallback_confirm_delete_database_rows"
            ):
                delete_forecast_records(record_ids)
                st.rerun()

        with cancel_col:
            if st.button(
                "Batal",
                type="secondary",
                use_container_width=True,
                key="fallback_cancel_delete_database_rows"
            ):
                st.rerun()


def render_database_management_page():
    with st.container(key="db_header_container"):
        back_col, title_col = st.columns([0.52, 5])

        with back_col:
            with st.container(key="db_back_button_container"):
                if st.button(
                    "← Back",
                    key="back_from_database_page",
                    use_container_width=True
                ):
                    st.session_state.pop("page_override", None)
                    st.rerun()

        with title_col:
            st.markdown(
                """
<div class="database-page-title database-page-title-clean database-page-title-no-bg-icon">
    <div class="database-page-icon-clean">🗄️</div>
    <div>
        <h1>Kelola Database</h1>
        <p>Menampilkan histori forecasting dan persediaan yang tersimpan pada database Supabase PostgreSQL.</p>
    </div>
</div>
""",
                unsafe_allow_html=True
            )

    try:
        database_df = load_all_forecast_records()
    except Exception as exc:
        st.error(str(exc))
        return

    if database_df.empty:
        st.markdown(
            """
<div class="empty-upload-state">
    <div class="empty-upload-icon">🗄️</div>
    <div class="empty-upload-title">Database Masih Kosong</div>
    <div class="empty-upload-subtitle">
        Belum ada histori forecasting yang tersimpan di Supabase.
    </div>
</div>
""",
            unsafe_allow_html=True
        )
        return

    database_df = database_df.copy()
    database_df["forecast_date"] = pd.to_datetime(
        database_df["forecast_date"],
        errors="coerce"
    )
    database_df["coffee_order"] = database_df["coffee_type"].apply(get_coffee_sort_order)
    database_df["model_order"] = database_df["model_name"].apply(get_model_sort_order)
    database_df = database_df.sort_values(
        ["forecast_date", "coffee_order", "model_order", "id"],
        ascending=[True, True, True, True]
    ).reset_index(drop=True)

    available_coffees = [
        coffee for coffee in jenis_kopi
        if coffee in database_df["coffee_type"].dropna().unique().tolist()
    ]

    available_years = sorted(
        database_df["forecast_date"]
        .dt.year
        .dropna()
        .astype(int)
        .unique()
        .tolist(),
        reverse=True
    )

    table_title_col, filter_col_1, filter_col_2 = st.columns([2.1, 1, 1])

    with table_title_col:
        st.markdown(
            """
<div class="database-section-title clean-database-title">📋 Tabel Database Forecasting</div>
<div class="database-section-caption clean-database-caption">
    Data bersifat permanen agar alur histori forecasting dan stock tetap konsisten.
</div>
""",
            unsafe_allow_html=True
        )

    with filter_col_1:
        st.markdown(
            """
<div class="history-filter-label">Jenis Kopi</div>
""",
            unsafe_allow_html=True
        )

        filter_kopi = st.selectbox(
            "Jenis Kopi",
            ["All"] + available_coffees,
            index=0,
            label_visibility="collapsed",
            key="database_filter_coffee"
        )

    with filter_col_2:
        st.markdown(
            """
<div class="history-filter-label">Pilih Tahun</div>
""",
            unsafe_allow_html=True
        )

        filter_year = st.selectbox(
            "Pilih Tahun",
            ["All"] + available_years,
            index=0,
            label_visibility="collapsed",
            key="database_filter_year"
        )

    filtered_df = database_df.copy()

    if filter_kopi != "All":
        filtered_df = filtered_df[filtered_df["coffee_type"] == filter_kopi]

    if filter_year != "All":
        filtered_df = filtered_df[
            filtered_df["forecast_date"].dt.year == int(filter_year)
        ]

    filtered_df = filtered_df.sort_values(
        ["forecast_date", "coffee_order", "model_order", "id"],
        ascending=[True, True, True, True]
    ).reset_index(drop=True)

    if filtered_df.empty:
        st.warning("Tidak ada data database sesuai filter yang dipilih.")
        return

    display_database_df = prepare_database_display_df(
        filtered_df.drop(columns=["coffee_order", "model_order"], errors="ignore")
    )

    render_premium_table(
        display_database_df,
        height=520,
        compact=False
    )

    action_col1, action_col2 = st.columns([1.2, 3])

    with action_col1:
        with st.container(key="db_download_button_container"):
            st.download_button(
                "⬇️ Download Excel",
                data=dataframe_to_excel_bytes(display_database_df),
                file_name="forecast_summary_supabase.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="download_forecast_database_excel"
            )

    with action_col2:
        st.markdown(
            f"""
<div class="database-action-note">
    Total {len(filtered_df)} row ditampilkan. Data forecasting tidak disediakan fitur hapus agar histori stock tetap konsisten.
</div>
""",
            unsafe_allow_html=True
        )


def create_history_chart(history_df, selected_period="Minggu 4"):
    fig = go.Figure()

    if history_df.empty:
        return fig

    chart_df = history_df.copy()
    chart_df["forecast_date"] = pd.to_datetime(chart_df["forecast_date"])

    grouped = (
        chart_df
        .groupby("forecast_date", as_index=False)
        .agg({
            "selected_total": "sum"
        })
    )

    period_number = str(selected_period).replace("Minggu", "").strip()
    if not period_number:
        period_number = "4"

    period_number_int = int(period_number)
    grouped["tanggal_awal"] = grouped["forecast_date"].dt.strftime("%d-%m-%Y")
    grouped["tanggal_akhir"] = (
        grouped["forecast_date"]
        + pd.to_timedelta((period_number_int * 7) - 1, unit="D")
    ).dt.strftime("%d-%m-%Y")

    fig.add_trace(
        go.Scatter(
            x=grouped["forecast_date"],
            y=grouped["selected_total"],
            mode="lines+markers",
            name="Total Kebutuhan",
            line=dict(color="#8b5a2b", width=3),
            marker=dict(size=7, color="#5c3317"),
            customdata=grouped[["tanggal_awal", "tanggal_akhir"]],
            hoverlabel=dict(
                bgcolor="#7b4318",
                bordercolor="#c18a54",
                font=dict(color="#ffffff", size=12, family="Poppins")
            ),
            hovertemplate=(
                "<b>Tanggal Awal:</b> %{customdata[0]}<br>"
                "<b>Tanggal Akhir:</b> %{customdata[1]}<br><br>"
                f"<b>Total Kebutuhan {period_number} Minggu Mendatang:</b><br>"
                "%{y:,.0f} gram"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fffaf4",
        font=dict(family="Poppins", color="#4e2e1e", size=12),
        title=dict(text=""),
        xaxis_title="Tanggal Forecasting",
        yaxis_title="Total Kebutuhan (Gram)",
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#7b4318",
            bordercolor="#c18a54",
            font=dict(color="#ffffff", size=12, family="Poppins")
        ),
        height=500,
        margin=dict(l=72, r=40, t=45, b=86)
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(92,51,23,0.10)",
        zeroline=False,
        linecolor="rgba(92,51,23,0.22)",
        tickfont=dict(color="#6b4b36", size=12),
        title_font=dict(color="#4e2e1e", size=14),
        automargin=True
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(92,51,23,0.10)",
        zeroline=False,
        linecolor="rgba(92,51,23,0.22)",
        tickfont=dict(color="#6b4b36", size=12),
        title_font=dict(color="#4e2e1e", size=14),
        automargin=True
    )

    return fig





def render_stock_opname_from_history(history_df, selected_kopi):
    if history_df.empty:
        return

    latest_df = history_df.copy()
    latest_df["forecast_date"] = pd.to_datetime(latest_df["forecast_date"])
    latest_df = latest_df.sort_values([
        "forecast_date",
        "created_at"
    ])

    latest_row = latest_df.iloc[-1]

    karung_kg = get_karung_kg(selected_kopi)
    lead_time = get_lead_time(selected_kopi)

    week_values = [
        float(latest_row.get("week1_gram", 0)),
        float(latest_row.get("week2_gram", 0)),
        float(latest_row.get("week3_gram", 0)),
        float(latest_row.get("week4_gram", 0))
    ]

    lead_time_forecast = sum(
        week_values[:min(lead_time, 4)]
    )

    safety_stock = float(latest_row.get("safety_stock", 0))
    minimum_stock = float(latest_row.get("minimum_stock", 0))
    maximum_stock = float(latest_row.get("maximum_stock", 0))

    latest_forecast_date = latest_row["forecast_date"].strftime("%d-%m-%Y")
    latest_model = latest_row.get("model_name", "-")

    st.markdown(
        """
<div class="forecast-section-title opname-title">📦 Stock Opname</div>
""",
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
<div class="opname-description-card">
    Stock opname digunakan untuk membandingkan stok aktual gudang dengan hasil forecasting terbaru pada riwayat forecasting.
    Data acuan yang digunakan adalah forecasting tanggal <b>{latest_forecast_date}</b> menggunakan model <b>{latest_model}</b>.
    Perhitungan rekomendasi pembelian menggunakan kebutuhan selama lead time <b>{lead_time} minggu</b> ditambah safety stock.
</div>
""",
        unsafe_allow_html=True
    )

    opname_input_col, opname_result_col = st.columns(
        [1, 1.45]
    )

    with opname_input_col:
        # Jangan membuka <div> lalu menyisipkan widget Streamlit di dalamnya.
        # Struktur seperti itu dapat membuat HTML berikutnya terbaca sebagai teks/kode.
        st.markdown(
            """
<div class="opname-card-title standalone-opname-title">Input Stok Aktual</div>
""",
            unsafe_allow_html=True
        )

        stock_unit = st.selectbox(
            "Satuan Stok Aktual",
            ["Gram", "Kilogram", "Karung"],
            index=1,
            key="history_stock_opname_unit"
        )

        stock_value = st.number_input(
            "Jumlah Stok Aktual",
            min_value=0.0,
            value=0.0,
            step=1.0,
            key="history_stock_opname_value"
        )

        stok_aktual_gram = convert_stock_to_gram(
            stock_value,
            stock_unit,
            karung_kg
        )

        st.markdown(
            f"""
<div class="opname-conversion-box">
    Stok aktual setara dengan:<br>
    <b>{stok_aktual_gram:,.0f} gram</b><br>
    <span>{stok_aktual_gram / 1000:,.2f} kg</span>
</div>
""",
            unsafe_allow_html=True
        )

    with opname_result_col:
        kebutuhan_dengan_safety = (
            lead_time_forecast
            + safety_stock
        )

        rekomendasi_beli_gram = max(
            kebutuhan_dengan_safety - stok_aktual_gram,
            0
        )

        rekomendasi_beli_kg = (
            rekomendasi_beli_gram / 1000
        )

        rekomendasi_beli_karung = int(
            math.ceil(rekomendasi_beli_kg / karung_kg)
        ) if rekomendasi_beli_gram > 0 else 0

        if stok_aktual_gram < minimum_stock:
            status_class = "opname-status-danger"
            status_icon = "⚠️"
            status_title = "Stok di Bawah Minimum Stock"
            status_message = "Stok aktual berada di bawah batas minimum sehingga perlu dilakukan pemantauan atau pembelian bahan baku."
        elif stok_aktual_gram > maximum_stock:
            status_class = "opname-status-warning"
            status_icon = "📦"
            status_title = "Stok Melebihi Maximum Stock"
            status_message = "Stok aktual melebihi batas maksimum sehingga perlu diwaspadai agar tidak terjadi overstock."
        else:
            status_class = "opname-status-safe"
            status_icon = "✅"
            status_title = "Persediaan Aman"
            status_message = "Stok aktual berada dalam batas persediaan yang direkomendasikan."

        result_html = f"""
<div class="opname-result-card">
    <div class="opname-status-box {status_class}">
        <div class="opname-status-icon">{status_icon}</div>
        <div>
            <div class="opname-status-title">{status_title}</div>
            <div class="opname-status-message">{status_message}</div>
        </div>
    </div>
    <div class="opname-result-grid">
        <div class="opname-result-item">
            <span>Forecast Selama Lead Time</span>
            <b>{lead_time_forecast:,.0f} gram</b>
        </div>
        <div class="opname-result-item">
            <span>Safety Stock</span>
            <b>{safety_stock:,.0f} gram</b>
        </div>
        <div class="opname-result-item">
            <span>Minimum Stock</span>
            <b>{minimum_stock:,.0f} gram</b>
        </div>
        <div class="opname-result-item">
            <span>Maximum Stock</span>
            <b>{maximum_stock:,.0f} gram</b>
        </div>
    </div>
    <div class="opname-recommendation-box">
        <div class="opname-recommendation-title">Rekomendasi Pembelian</div>
        <div class="opname-recommendation-value">{rekomendasi_beli_gram:,.0f} gram</div>
        <div class="opname-recommendation-subtitle">
            Setara {rekomendasi_beli_kg:,.2f} kg atau sekitar <b>{rekomendasi_beli_karung} karung</b>
            ({karung_kg} kg/karung)
        </div>
    </div>
</div>
"""
        st.markdown(
            result_html,
            unsafe_allow_html=True
        )

def render_forecast_history_graph_table(selected_kopi, model_name=None):
    st.markdown(
        f"""
<div class="history-forecast-main-header">
    <div class="history-forecast-icon">↺</div>
    <div class="history-forecast-content">
        <h1>Riwayat Forecasting</h1>
        <p>Grafik dan tabel riwayat forecasting yang tersimpan otomatis berdasarkan jenis kopi yang dipilih.</p>
        <div class="history-forecast-badges">
            <span>☕ Biji Kopi: <b>{selected_kopi}</b></span>
            <span>🏆 Model: <b>{model_name if model_name else "Semua Model"}</b></span>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True
    )

    history_all = load_forecast_history(
        jenis_kopi=selected_kopi,
        periode="Minggu 4",
        model_name=model_name
    )

    if history_all.empty:
        st.markdown(
            f"""
<div class="empty-upload-state">
    <div class="empty-upload-icon">📁</div>
    <div class="empty-upload-title">Belum Ada Riwayat Forecasting</div>
    <div class="empty-upload-subtitle">
        Jalankan proses forecasting untuk kopi {selected_kopi} terlebih dahulu agar data tersimpan ke database.
    </div>
</div>
""",
            unsafe_allow_html=True
        )

        return

    available_years = sorted(
        pd.to_datetime(history_all["forecast_date"])
        .dt.year
        .dropna()
        .astype(int)
        .unique()
        .tolist(),
        reverse=True
    )

    title_col, filter_col1, filter_col2 = st.columns([2.2, 1, 1])

    with title_col:
        st.markdown(
            """
<div class="history-chart-title-row">
    <div class="history-chart-title">📈 Grafik Histori Forecasting</div>
    <div class="history-chart-subtitle">Pilih periode dan tahun untuk melihat perkembangan total kebutuhan bahan baku.</div>
</div>
""",
            unsafe_allow_html=True
        )

    with filter_col1:
        st.markdown(
            """
<div class="history-filter-label">Periode Forecast</div>
""",
            unsafe_allow_html=True
        )

        selected_period = st.selectbox(
            "Periode Forecast",
            ["Minggu 1", "Minggu 2", "Minggu 3", "Minggu 4"],
            index=3,
            label_visibility="collapsed",
            key=f"history_period_{selected_kopi}"
        )

    with filter_col2:
        st.markdown(
            """
<div class="history-filter-label">Tahun</div>
""",
            unsafe_allow_html=True
        )

        selected_year = st.selectbox(
            "Tahun",
            available_years,
            index=0,
            label_visibility="collapsed",
            key=f"history_year_{selected_kopi}_{model_name if model_name else 'all'}"
        )

    history_df = load_forecast_history(
        jenis_kopi=selected_kopi,
        periode=selected_period,
        tahun=selected_year,
        model_name=model_name
    )

    if history_df.empty:
        st.warning("Tidak ada data riwayat sesuai filter yang dipilih.")
        return

    fig_history = create_history_chart(history_df, selected_period)
    st.plotly_chart(fig_history, use_container_width=True)

    # =====================================
    # FILTER KHUSUS TABEL HISTORI
    # =====================================
    table_source_df = load_forecast_history(
        jenis_kopi=selected_kopi,
        periode=selected_period,
        model_name=model_name
    )

    if table_source_df.empty:
        st.warning("Tidak ada data tabel histori forecasting.")
        return

    table_source_df["forecast_date"] = pd.to_datetime(
        table_source_df["forecast_date"],
        errors="coerce"
    )

    month_options = [
        "All",
        "Januari",
        "Februari",
        "Maret",
        "April",
        "Mei",
        "Juni",
        "Juli",
        "Agustus",
        "September",
        "Oktober",
        "November",
        "Desember"
    ]

    month_mapping = {
        "Januari": 1,
        "Februari": 2,
        "Maret": 3,
        "April": 4,
        "Mei": 5,
        "Juni": 6,
        "Juli": 7,
        "Agustus": 8,
        "September": 9,
        "Oktober": 10,
        "November": 11,
        "Desember": 12
    }

    table_year_options = ["All"] + sorted(
        table_source_df["forecast_date"]
        .dt.year
        .dropna()
        .astype(int)
        .astype(str)
        .unique()
        .tolist(),
        reverse=True
    )

    table_title_col, table_filter_col1, table_filter_col2 = st.columns([2.4, 1, 1])

    with table_title_col:
        st.markdown(
            """
<div class="forecast-section-title history-table-title">📋 Tabel Histori Forecasting (Takaran Gram)</div>
""",
            unsafe_allow_html=True
        )

    with table_filter_col1:
        st.markdown(
            """
<div class="history-filter-label">Bulan</div>
""",
            unsafe_allow_html=True
        )

        selected_table_month = st.selectbox(
            "Bulan",
            month_options,
            index=0,
            label_visibility="collapsed",
            key=f"history_table_month_{selected_kopi}_{model_name if model_name else 'all'}"
        )

    with table_filter_col2:
        st.markdown(
            """
<div class="history-filter-label">Tahun</div>
""",
            unsafe_allow_html=True
        )

        selected_table_year = st.selectbox(
            "Tahun",
            table_year_options,
            index=0,
            label_visibility="collapsed",
            key=f"history_table_year_{selected_kopi}_{model_name if model_name else 'all'}"
        )

    display_df = table_source_df.copy()

    if selected_table_month != "All":
        display_df = display_df[
            display_df["forecast_date"].dt.month == month_mapping[selected_table_month]
        ]

    if selected_table_year != "All":
        display_df = display_df[
            display_df["forecast_date"].dt.year.astype(str) == selected_table_year
        ]

    if display_df.empty:
        st.markdown(
            """
<div class="empty-upload-state history-table-empty-state">
    <div class="empty-upload-icon">📋</div>
    <div class="empty-upload-title">Data Tidak Ditemukan</div>
    <div class="empty-upload-subtitle">Tidak ada riwayat forecasting sesuai filter bulan dan tahun yang dipilih.</div>
</div>
""",
            unsafe_allow_html=True
        )
        return

    display_df["Minggu Forecasting"] = pd.to_datetime(
        display_df["forecast_date"]
    ).dt.strftime("%d-%m-%Y")

    if "actual_total" not in display_df.columns:
        display_df["actual_total"] = 0

    # Tambahkan indikator kenaikan/penurunan pada Total Aktual
    # Perbandingan dilakukan terhadap Total Aktual pada baris histori sebelumnya
    # untuk jenis kopi dan model yang sedang ditampilkan.
    display_df = display_df.sort_values("forecast_date").reset_index(drop=True)
    display_df["actual_total"] = display_df["actual_total"].fillna(0).astype(float)
    display_df["actual_change"] = display_df["actual_total"].diff()

    def format_actual_with_icon(row):
        value = f"{row['actual_total']:,.0f}"

        if pd.isna(row["actual_change"]):
            return f"{value} <span class='trend-neutral'>−</span>"

        if row["actual_change"] > 0:
            return f"{value} <span class='trend-up'>▲</span>"

        if row["actual_change"] < 0:
            return f"{value} <span class='trend-down'>▼</span>"

        return f"{value} <span class='trend-neutral'>→</span>"

    display_df["Total Aktual"] = display_df.apply(format_actual_with_icon, axis=1)

    display_df["Minggu 1"] = (
        display_df["week1_gram"]
        .round(0)
        .astype(int)
        .map("{:,}".format)
    )

    display_df["Minggu 2"] = (
        display_df["week2_gram"]
        .round(0)
        .astype(int)
        .map("{:,}".format)
    )

    display_df["Minggu 3"] = (
        display_df["week3_gram"]
        .round(0)
        .astype(int)
        .map("{:,}".format)
    )

    display_df["Minggu 4"] = (
        display_df["week4_gram"]
        .round(0)
        .astype(int)
        .map("{:,}".format)
    )

    display_df["Safety Stock"] = (
        display_df["safety_stock"]
        .round(0)
        .astype(int)
        .map("{:,}".format)
    )

    display_df["Minimum Stock"] = (
        display_df["minimum_stock"]
        .round(0)
        .astype(int)
        .map("{:,}".format)
    )

    display_df["Maximum Stock"] = (
        display_df["maximum_stock"]
        .round(0)
        .astype(int)
        .map("{:,}".format)
    )

    display_df = display_df[
        [
            "Minggu Forecasting",
            "coffee_type",
            "Total Aktual",
            "model_name",
            "Minggu 1",
            "Minggu 2",
            "Minggu 3",
            "Minggu 4",
            "Safety Stock",
            "Minimum Stock",
            "Maximum Stock"
        ]
    ]

    display_df.columns = [
        "Minggu Forecasting",
        "Biji Kopi",
        "Total Aktual",
        "Model",
        "Minggu 1",
        "Minggu 2",
        "Minggu 3",
        "Minggu 4",
        "Safety Stock",
        "Minimum Stock",
        "Maximum Stock"
    ]

    history_table_height = 420 if len(display_df) > 6 else None

    render_premium_table(
        display_df,
        height=history_table_height,
        compact=False
    )

    st.markdown(
        """
<div class="database-manage-button-spacer"></div>
""",
        unsafe_allow_html=True
    )

    if st.button(
        "Kelola Database",
        type="primary",
        use_container_width=False,
        key=f"go_to_database_management_{selected_kopi}_{model_name if model_name else 'all'}"
    ):
        st.session_state["page_override"] = "Kelola Database"
        st.rerun()


def get_latest_inventory_row(selected_kopi, model_name=None):
    """
    Mengambil record persediaan paling baru langsung dari Supabase.
    Fungsi ini sengaja tidak memakai cache/session state agar perubahan stock
    setelah upload dataset minggu baru langsung tampil di menu Persediaan.
    """
    supabase_url, _ = require_supabase_config()
    endpoint = f"{supabase_url}/rest/v1/forecast_summary"

    request_params = [
        ("select", "*"),
        ("coffee_type", f"eq.{selected_kopi}"),
        ("order", "forecast_date.desc"),
        ("order", "created_at.desc"),
        ("limit", "1")
    ]

    if model_name:
        request_params.append(("model_name", f"eq.{model_name}"))

    response = safe_supabase_get(
        endpoint,
        headers=get_supabase_headers(),
        params=request_params,
        timeout=20
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Gagal membaca data persediaan terbaru dari Supabase: {response.status_code} - {response.text}"
        )

    data = response.json()

    if not data:
        return None, pd.DataFrame()

    latest_df = pd.DataFrame(data)

    rename_map = {
        "week1": "week1_gram",
        "week2": "week2_gram",
        "week3": "week3_gram",
        "week4": "week4_gram"
    }
    latest_df = latest_df.rename(columns=rename_map)

    numeric_cols = [
        "stock",
        "actual_total",
        "week1_gram",
        "week2_gram",
        "week3_gram",
        "week4_gram",
        "total_week1",
        "total_week2",
        "total_week3",
        "total_week4",
        "safety_stock",
        "minimum_stock",
        "maximum_stock"
    ]

    for col in numeric_cols:
        if col in latest_df.columns:
            latest_df[col] = pd.to_numeric(latest_df[col], errors="coerce").fillna(0)

    if "created_at" not in latest_df.columns:
        latest_df["created_at"] = ""

    return latest_df.iloc[0], latest_df


def clamp_percentage(value, max_value):
    try:
        if max_value <= 0:
            return 0.0
        return max(0.0, min(100.0, (float(value) / float(max_value)) * 100.0))
    except Exception:
        return 0.0


def render_inventory_stock_indicator(
    selected_kopi,
    latest_row,
    stock_now,
    safety_stock,
    minimum_stock,
    maximum_stock
):
    """Menampilkan indikator posisi stock terhadap stockout, safety, minimum, dan overstock."""
    stock_now = float(stock_now or 0)
    safety_stock = float(safety_stock or 0)
    minimum_stock = float(minimum_stock or 0)
    maximum_stock = float(maximum_stock or 0)

    if maximum_stock <= 0 and minimum_stock <= 0 and safety_stock <= 0:
        st.markdown(
            '<div class="inventory-indicator-card"><div class="inventory-indicator-empty">Belum ada data safety stock, minimum stock, dan maximum stock untuk jenis kopi ini. Jalankan forecasting terlebih dahulu.</div></div>',
            unsafe_allow_html=True
        )
        return

    # Skala indikator tidak mengikuti stock saat ini ketika melewati maximum stock.
    # Tujuannya agar marker stock saat ini tidak keluar dari bar indikator;
    # angka tetap menampilkan nilai asli, tetapi posisi visual dibatasi sampai ujung kanan.
    axis_max = max(
        maximum_stock * 1.08,
        minimum_stock * 1.25,
        safety_stock * 1.5,
        1
    )

    stockout_pos = 0
    safety_pos = clamp_percentage(safety_stock, axis_max)
    minimum_pos = clamp_percentage(minimum_stock, axis_max)
    stock_pos = clamp_percentage(min(stock_now, axis_max), axis_max)
    maximum_pos = clamp_percentage(maximum_stock, axis_max)

    def format_gram_label(value):
        return f"{float(value):,.0f} Gram"

    lead_time = get_lead_time(selected_kopi)
    week_values = [0.0, 0.0, 0.0, 0.0]

    if latest_row is not None:
        for idx, col in enumerate(["week1_gram", "week2_gram", "week3_gram", "week4_gram"]):
            if col in latest_row:
                week_values[idx] = float(latest_row.get(col, 0) or 0)
            else:
                alt_col = f"week{idx + 1}"
                week_values[idx] = float(latest_row.get(alt_col, 0) or 0)

    lead_time_forecast = sum(week_values[:min(lead_time, 4)])
    target_stock = max(minimum_stock, lead_time_forecast + safety_stock)

    if maximum_stock > 0:
        target_stock = min(target_stock, maximum_stock)

    recommended_order = max(target_stock - stock_now, 0)

    if maximum_stock > 0:
        recommended_order = min(recommended_order, max(maximum_stock - stock_now, 0))

    karung_kg = get_karung_kg(selected_kopi)
    recommended_order_kg = recommended_order / 1000
    recommended_order_sacks = int(math.ceil(recommended_order_kg / karung_kg)) if recommended_order > 0 else 0

    if stock_now <= 0:
        status_class = "danger"
        status_title = "Stockout / Stock Habis"
        status_message = "Stock saat ini sudah habis. PPIC perlu segera melakukan pemesanan bahan baku."
    elif safety_stock > 0 and stock_now < safety_stock:
        status_class = "danger"
        status_title = "Risiko Stockout"
        status_message = "Stock saat ini berada di bawah safety stock. Stock tidak aman untuk menghadapi fluktuasi kebutuhan."
    elif minimum_stock > 0 and stock_now <= minimum_stock:
        status_class = "warning"
        status_title = "Sudah Menyentuh Minimum Stock"
        status_message = "Stock saat ini sudah memasuki batas minimum. PPIC disarankan segera melakukan pemesanan stock."
    elif maximum_stock > 0 and stock_now > maximum_stock:
        status_class = "overstock"
        status_title = "Overstock"
        status_message = "Stock saat ini melebihi maximum stock. Waspadai risiko overstock dan biaya penyimpanan."
    else:
        status_class = "safe"
        status_title = "Stock Aman"
        status_message = "Stock saat ini berada pada rentang aman antara minimum stock dan maximum stock."

    recommendation_html = ""
    if minimum_stock > 0 and stock_now <= minimum_stock and recommended_order > 0:
        recommendation_html = (
            '<div class="inventory-recommendation-box">'
            '<div class="inventory-recommendation-title">Rekomendasi Pemesanan</div>'
            f'<div class="inventory-recommendation-value">{recommended_order:,.0f} Gram</div>'
            f'<div class="inventory-recommendation-subvalue">Setara {recommended_order_kg:,.2f} Kg atau sekitar <b>{recommended_order_sacks} Karung</b> ({karung_kg} Kg/Karung)</div>'
            '<div class="inventory-recommendation-text">'
            f'Rekomendasi ini dihitung untuk menjaga stock bertahan selama lead time <b>{lead_time} minggu</b> '
            'dengan mempertimbangkan safety stock, tanpa melewati batas maximum stock.'
            '</div></div>'
        )
    elif maximum_stock > 0 and stock_now > maximum_stock:
        recommendation_html = (
            '<div class="inventory-recommendation-box overstock-note">'
            '<div class="inventory-recommendation-title">Peringatan Overstock</div>'
            '<div class="inventory-recommendation-text">'
            'Tidak disarankan melakukan pembelian tambahan sampai stock kembali berada di bawah batas maximum stock.'
            '</div></div>'
        )

    # HTML dibuat tanpa indentasi Markdown agar tidak terbaca sebagai blok kode.
    indicator_html = (
        '<div class="inventory-indicator-card">'
        '<div class="inventory-indicator-header">'
        '<div><h3>Indikator Persediaan</h3>'
        '<p>Memantau posisi stock saat ini terhadap batas stockout, safety stock, minimum stock, dan maximum stock.</p></div>'
        f'<div class="inventory-status-pill {status_class}">{status_title}</div>'
        '</div>'
        '<div class="inventory-bar-area">'
        f'<div class="inventory-marker marker-stockout" style="left:{stockout_pos}%;"><div class="marker-value">0 Gram</div><div class="triangle down"></div></div>'
        f'<div class="inventory-marker marker-safety" style="left:{safety_pos}%;"><div class="marker-value">{format_gram_label(safety_stock)}</div><div class="triangle down"></div></div>'
        f'<div class="inventory-marker marker-minimum" style="left:{minimum_pos}%;"><div class="marker-value">{format_gram_label(minimum_stock)}</div><div class="triangle down"></div></div>'
        f'<div class="inventory-marker marker-current" style="left:{stock_pos}%;"><div class="marker-value">{format_gram_label(stock_now)}</div><div class="triangle down"></div><div class="current-dot"></div></div>'
        f'<div class="inventory-marker marker-maximum" style="left:{maximum_pos}%;"><div class="marker-value">{format_gram_label(maximum_stock)}</div><div class="triangle down"></div></div>'
        '<div class="inventory-gradient-bar"></div>'
        '<div class="inventory-bar-labels"><span>Stockout</span><span>Overstock</span></div>'
        '</div>'
        '<div class="inventory-legend-row">'
        '<span><i class="legend-color black"></i> Stockout / Habis</span>'
        '<span><i class="legend-color grey"></i> Safety Stock</span>'
        '<span><i class="legend-color yellow"></i> Minimum Stock</span>'
        '<span><i class="legend-color brown"></i> Stock Saat Ini</span>'
        '<span><i class="legend-color red"></i> Maximum / Overstock</span>'
        '</div>'
        f'<div class="inventory-status-message {status_class}">{status_message}</div>'
        f'{recommendation_html}'
        '</div>'
    )

    st.markdown(indicator_html, unsafe_allow_html=True)



def render_inventory_coverage_forecast(
    selected_kopi,
    latest_row,
    stock_now,
    safety_stock
):
    """Menampilkan kemampuan stock memenuhi prediksi 4 minggu ke depan.
    Stock yang dihitung adalah stock tersedia = stock saat ini - safety stock.
    """
    if latest_row is None:
        return

    stock_now = float(stock_now or 0)
    safety_stock = float(safety_stock or 0)
    available_stock = max(stock_now - safety_stock, 0)

    week_values = []
    for idx, col in enumerate(["week1_gram", "week2_gram", "week3_gram", "week4_gram"]):
        alt_col = f"week{idx + 1}"
        if col in latest_row:
            value = float(latest_row.get(col, 0) or 0)
        else:
            value = float(latest_row.get(alt_col, 0) or 0)
        week_values.append(value)

    cumulative_values = []
    running_total = 0.0
    for value in week_values:
        running_total += value
        cumulative_values.append(running_total)

    try:
        start_date = pd.to_datetime(latest_row.get("forecast_date"))
    except Exception:
        start_date = pd.Timestamp.today()

    def fmt_gram(value):
        return f"{float(value):,.0f} Gram"

    arrow_items = []
    full_weeks_covered = 0
    previous_total = 0.0

    for idx, (week_value, cumulative_total) in enumerate(zip(week_values, cumulative_values), start=1):
        week_start = start_date + pd.Timedelta(days=(idx - 1) * 7)
        week_end = week_start + pd.Timedelta(days=6)

        if available_stock >= cumulative_total:
            status_class = "safe"
            full_weeks_covered = idx
        elif available_stock > previous_total:
            status_class = "warning"
        else:
            status_class = "danger"

        arrow_items.append(
            f"""
<div class="coverage-arrow-item coverage-{status_class}">
    <div class="coverage-week-label">Minggu {idx}</div>
    <div class="coverage-arrow-shape">
        <div class="coverage-arrow-content">
            <span>Prediksi Minggu {idx}</span>
            <b>{fmt_gram(week_value)}</b>
            <span>Total Minggu {idx}</span>
            <b>{fmt_gram(cumulative_total)}</b>
        </div>
    </div>
    <div class="coverage-date-range">{week_start.strftime('%d-%m-%Y')} s/d {week_end.strftime('%d-%m-%Y')}</div>
</div>
"""
        )
        previous_total = cumulative_total

    if full_weeks_covered >= 4:
        note_text = "Stock saat ini bisa memenuhi kebutuhan 4 minggu ke depan."
        note_class = "safe"
    elif full_weeks_covered > 0:
        note_text = f"Stock saat ini hanya bisa memenuhi kebutuhan {full_weeks_covered} minggu ke depan."
        note_class = "warning"
    else:
        note_text = "Stock saat ini belum cukup untuk memenuhi kebutuhan minggu pertama."
        note_class = "danger"

    coverage_html = (
        '<div class="coverage-forecast-card">'
        '<div class="coverage-header">'
        '<div><h3>Proyeksi Kecukupan Stock 4 Minggu</h3>'
        '<p>Perhitungan menggunakan stock tersedia untuk prediksi, yaitu <b>stock saat ini - safety stock</b>.</p></div>'
        f'<div class="coverage-available-pill">Stock tersedia: <b>{fmt_gram(available_stock)}</b></div>'
        '</div>'
        '<div class="coverage-arrow-row">'
        + ''.join(arrow_items) +
        '</div>'
        f'<div class="coverage-note {note_class}">{note_text}</div>'
        '</div>'
    )

    st.markdown(coverage_html, unsafe_allow_html=True)



def render_inventory_history_table(selected_kopi, model_name=None):
    """Menampilkan historis persediaan dari forecast_summary."""
    try:
        history_df = load_forecast_history(
            jenis_kopi=selected_kopi,
            periode="Minggu 4",
            model_name=model_name
        )
    except Exception as exc:
        if is_database_connection_error(exc):
            render_database_connection_error()
        else:
            st.error(str(exc))
        return

    if history_df.empty:
        st.markdown(
            """
<div class="forecast-section-title inventory-history-title">📦 Tabel Historis Persediaan (Takaran Gram)</div>
""",
            unsafe_allow_html=True
        )

        st.markdown(
            """
<div class="empty-upload-state history-table-empty-state">
    <div class="empty-upload-icon">📦</div>
    <div class="empty-upload-title">Data Persediaan Belum Ada</div>
    <div class="empty-upload-subtitle">Belum ada histori persediaan untuk jenis kopi yang dipilih.</div>
</div>
""",
            unsafe_allow_html=True
        )
        return

    history_df = history_df.copy()
    history_df["forecast_date"] = pd.to_datetime(
        history_df["forecast_date"],
        errors="coerce"
    )

    month_options = [
        "All",
        "Januari",
        "Februari",
        "Maret",
        "April",
        "Mei",
        "Juni",
        "Juli",
        "Agustus",
        "September",
        "Oktober",
        "November",
        "Desember"
    ]

    month_mapping = {
        "Januari": 1,
        "Februari": 2,
        "Maret": 3,
        "April": 4,
        "Mei": 5,
        "Juni": 6,
        "Juli": 7,
        "Agustus": 8,
        "September": 9,
        "Oktober": 10,
        "November": 11,
        "Desember": 12
    }

    inventory_year_options = ["All"] + sorted(
        history_df["forecast_date"]
        .dt.year
        .dropna()
        .astype(int)
        .astype(str)
        .unique()
        .tolist(),
        reverse=True
    )

    title_col, month_filter_col, year_filter_col = st.columns([2.4, 1, 1])

    with title_col:
        st.markdown(
            """
<div class="forecast-section-title inventory-history-title">📦 Tabel Historis Persediaan (Takaran Gram)</div>
""",
            unsafe_allow_html=True
        )

    with month_filter_col:
        st.markdown(
            """
<div class="history-filter-label inventory-history-filter-label">Bulan</div>
""",
            unsafe_allow_html=True
        )

        selected_inventory_month = st.selectbox(
            "Bulan Historis Persediaan",
            month_options,
            index=0,
            label_visibility="collapsed",
            key=f"inventory_history_month_{selected_kopi}_{model_name if model_name else 'all'}"
        )

    with year_filter_col:
        st.markdown(
            """
<div class="history-filter-label inventory-history-filter-label">Tahun</div>
""",
            unsafe_allow_html=True
        )

        selected_inventory_year = st.selectbox(
            "Tahun Historis Persediaan",
            inventory_year_options,
            index=0,
            label_visibility="collapsed",
            key=f"inventory_history_year_{selected_kopi}_{model_name if model_name else 'all'}"
        )

    display_df = history_df.copy()

    if selected_inventory_month != "All":
        display_df = display_df[
            display_df["forecast_date"].dt.month == month_mapping[selected_inventory_month]
        ]

    if selected_inventory_year != "All":
        display_df = display_df[
            display_df["forecast_date"].dt.year.astype(str) == selected_inventory_year
        ]

    if display_df.empty:
        st.markdown(
            """
<div class="empty-upload-state history-table-empty-state">
    <div class="empty-upload-icon">📦</div>
    <div class="empty-upload-title">Data Tidak Ditemukan</div>
    <div class="empty-upload-subtitle">Tidak ada riwayat persediaan sesuai filter bulan dan tahun yang dipilih.</div>
</div>
""",
            unsafe_allow_html=True
        )
        return

    display_df = display_df.sort_values(["forecast_date", "created_at"]).reset_index(drop=True)

    if "actual_total" not in display_df.columns:
        display_df["actual_total"] = 0

    if "stock" not in display_df.columns:
        display_df["stock"] = 0

    display_df["actual_total"] = pd.to_numeric(
        display_df["actual_total"],
        errors="coerce"
    ).fillna(0)

    display_df["stock"] = pd.to_numeric(
        display_df["stock"],
        errors="coerce"
    ).fillna(0)

    display_df["actual_change"] = display_df["actual_total"].diff()

    def format_actual_with_icon(row):
        value = f"{row['actual_total']:,.0f}"
        if pd.isna(row["actual_change"]):
            return f"{value} <span class='trend-neutral'>−</span>"
        if row["actual_change"] > 0:
            return f"{value} <span class='trend-up'>▲</span>"
        if row["actual_change"] < 0:
            return f"{value} <span class='trend-down'>▼</span>"
        return f"{value} <span class='trend-neutral'>→</span>"

    table_df = pd.DataFrame({
        "Tanggal": display_df["forecast_date"].dt.strftime("%d-%m-%Y"),
        "Jenis Kopi": display_df["coffee_type"].apply(format_database_coffee_name),
        "Total Aktual / Kebutuhan Minggu Lalu": display_df.apply(format_actual_with_icon, axis=1),
        "Stock Saat Ini": (
            display_df["stock"]
            .round(0)
            .astype(int)
            .map("{:,}".format)
        )
    })

    table_height = 390 if len(table_df) > 6 else None

    render_premium_table(
        table_df,
        height=table_height,
        compact=False
    )

def render_persediaan_page(default_kopi, recommended_model_name=None):
    if recommended_model_name is None:
        recommended_model_name = "XGBoost"

    header_left, header_right = st.columns([2.3, 1])

    with header_left:
        st.markdown(
            """
<div class="forecast-page-title inventory-page-title">
    <div class="forecast-page-icon">📦</div>
    <div>
        <h1>Persediaan</h1>
        <p>Monitoring stock bahan baku berdasarkan histori forecasting dan data aktual terbaru.</p>
    </div>
</div>
""",
            unsafe_allow_html=True
        )

    with header_right:
        st.markdown(
            f"""
<div class="forecast-model-badge forecast-model-badge-small inventory-recommendation-model-badge">
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

    # Pilihan kopi khusus menu Persediaan dipisahkan dari key global
    # agar selectbox tidak kembali otomatis ke Robusta saat rerun.
    if (
        "inventory_selected_kopi" not in st.session_state
        or st.session_state.get("inventory_selected_kopi") not in jenis_kopi
    ):
        st.session_state["inventory_selected_kopi"] = default_kopi

    selected_inventory_kopi = st.session_state.get(
        "inventory_selected_kopi",
        default_kopi
    )

    if selected_inventory_kopi not in jenis_kopi:
        selected_inventory_kopi = default_kopi
        st.session_state["inventory_selected_kopi"] = selected_inventory_kopi

    def on_inventory_coffee_change():
        """Reload data ketika jenis kopi di menu Persediaan diganti tanpa mereset pilihan."""
        try:
            st.cache_data.clear()
        except Exception:
            pass

        selected_from_widget = st.session_state.get(
            "inventory_selected_kopi",
            default_kopi
        )

        if selected_from_widget in jenis_kopi:
            st.session_state["selected_kopi_global"] = selected_from_widget

        st.session_state["inventory_form_open"] = False
        st.session_state["inventory_refresh_token"] = datetime.now().timestamp()

    p1, p2, p3, p4, p5 = st.columns(5)

    with p1:
        kategori_kopi_awal = (
            "Robusta"
            if selected_inventory_kopi in ["Robusta", "Robusta_Highblend"]
            else "Arabika"
        )

        render_eval_metric_card(
            "☕",
            "Biji Kopi",
            selected_inventory_kopi,
            f"Kategori {kategori_kopi_awal}"
        )

        st.selectbox(
            "Pilih jenis kopi:",
            jenis_kopi,
            key="inventory_selected_kopi",
            label_visibility="visible",
            on_change=on_inventory_coffee_change
        )

    # Ambil ulang nilai setelah selectbox dirender.
    # Nilai ini memakai key khusus Persediaan agar tidak tertimpa default Robusta.
    selected_inventory_kopi = st.session_state.get(
        "inventory_selected_kopi",
        default_kopi
    )

    if selected_inventory_kopi not in jenis_kopi:
        selected_inventory_kopi = default_kopi
        st.session_state["inventory_selected_kopi"] = selected_inventory_kopi

    st.session_state["selected_kopi_global"] = selected_inventory_kopi

    try:
        latest_row, history_df = get_latest_inventory_row(
            selected_inventory_kopi,
            model_name=recommended_model_name
        )
    except Exception as exc:
        if is_database_connection_error(exc):
            render_database_connection_error()
        else:
            st.error(str(exc))
        return

    if latest_row is not None:
        stock_now = float(latest_row.get("stock", 0) or 0)
        safety_stock = float(latest_row.get("safety_stock", 0) or 0)
        minimum_stock = float(latest_row.get("minimum_stock", 0) or 0)
        maximum_stock = float(latest_row.get("maximum_stock", 0) or 0)
        latest_date = pd.to_datetime(latest_row.get("forecast_date")).strftime("%d-%m-%Y")
    else:
        stock_now = 0.0
        safety_stock = 0.0
        minimum_stock = 0.0
        maximum_stock = 0.0
        latest_date = "-"

    with p2:
        render_eval_metric_card(
            "📦",
            "Stock Saat Ini",
            f"{stock_now:,.0f}",
            f"Gram | Update {latest_date}"
        )

        stock_action_col1, stock_action_col2 = st.columns(2)

        with stock_action_col1:
            with st.container(key="inventory_mode_tambah_button"):
                if st.button(
                    "➕ Tambah",
                    use_container_width=True,
                    key=f"inventory_open_tambah_{selected_inventory_kopi}_{recommended_model_name}"
                ):
                    st.session_state["inventory_stock_mode"] = "Tambah"
                    st.session_state["inventory_form_open"] = True
                    st.rerun()

        with stock_action_col2:
            with st.container(key="inventory_mode_ubah_button"):
                if st.button(
                    "✏️ Ubah",
                    use_container_width=True,
                    key=f"inventory_open_ubah_{selected_inventory_kopi}_{recommended_model_name}"
                ):
                    st.session_state["inventory_stock_mode"] = "Ubah"
                    st.session_state["inventory_form_open"] = True
                    st.rerun()

        if st.session_state.get("inventory_form_open", False):
            action_mode = st.session_state.get("inventory_stock_mode", "Tambah")

            with st.container(key="inventory_brown_form_container"):
                st.markdown(
                    f"""
<div class="inventory-brown-form-title">
    {"➕" if action_mode == "Tambah" else "✏️"} {action_mode} Stock
</div>
<div class="inventory-brown-form-caption">
    Masukkan nominal stock, pilih satuan, lalu simpan perubahan.
</div>
""",
                    unsafe_allow_html=True
                )

                input_col, unit_col = st.columns([1.05, 1.20])

                stock_input_key = f"inventory_compact_stock_value_{selected_inventory_kopi}_{recommended_model_name}_{action_mode}"

                if (
                    stock_input_key not in st.session_state
                    or not isinstance(st.session_state[stock_input_key], str)
                ):
                    st.session_state[stock_input_key] = format_stock_number_id(
                        st.session_state.get(stock_input_key, 0)
                    )

                with input_col:
                    stock_value_text = st.text_input(
                        "Nominal",
                        key=stock_input_key,
                        on_change=normalize_stock_input_text,
                        args=(stock_input_key,),
                        placeholder="Contoh: 20.000"
                    )

                with unit_col:
                    stock_unit = st.selectbox(
                        "Satuan",
                        ["Gram", "Kilogram", "Karung"],
                        index=0,
                        key=f"inventory_compact_stock_unit_{selected_inventory_kopi}_{recommended_model_name}_{action_mode}"
                    )

                karung_kg = get_karung_kg(selected_inventory_kopi)

                stock_value = parse_stock_input_text(stock_value_text)

                input_stock_gram = convert_stock_to_gram(
                    stock_value,
                    stock_unit,
                    karung_kg
                )

                if action_mode == "Tambah":
                    new_stock = stock_now + input_stock_gram
                else:
                    new_stock = input_stock_gram

                st.markdown(
                    f"""
<div class="inventory-brown-conversion-box">
    <span>Hasil konversi</span>
    <b>{input_stock_gram:,.0f} gram</b>
    <small>Stock setelah {action_mode.lower()}: <b>{new_stock:,.0f} gram</b></small>
</div>
""",
                    unsafe_allow_html=True
                )

                save_col, cancel_col = st.columns(2)

                with save_col:
                    with st.container(key="inventory_compact_save_button_container"):
                        if st.button(
                            "Simpan",
                            use_container_width=True,
                            key=f"inventory_compact_save_stock_{selected_inventory_kopi}_{recommended_model_name}_{action_mode}"
                        ):
                            if latest_row is None:
                                st.warning(
                                    "Belum ada histori forecasting untuk jenis kopi ini. Jalankan forecasting terlebih dahulu sebelum mengubah stock."
                                )
                            else:
                                try:
                                    update_latest_stock_for_coffee(
                                        selected_inventory_kopi,
                                        new_stock,
                                        model_name=recommended_model_name
                                    )
                                    st.session_state["inventory_form_open"] = False
                                    st.rerun()
                                except Exception as exc:
                                    if is_database_connection_error(exc):
                                        render_database_connection_error()
                                    else:
                                        st.error(str(exc))

                with cancel_col:
                    with st.container(key="inventory_compact_cancel_button_container"):
                        if st.button(
                            "Batal",
                            use_container_width=True,
                            key=f"inventory_compact_cancel_stock_{selected_inventory_kopi}_{recommended_model_name}_{action_mode}"
                        ):
                            st.session_state["inventory_form_open"] = False
                            st.rerun()

    with p3:
        render_eval_metric_card(
            "🛡️",
            "Safety Stock",
            f"{safety_stock:,.0f}",
            "Gram"
        )

    with p4:
        render_eval_metric_card(
            "📉",
            "Minimum Stock",
            f"{minimum_stock:,.0f}",
            "Gram"
        )

    with p5:
        render_eval_metric_card(
            "📈",
            "Maximum Stock",
            f"{maximum_stock:,.0f}",
            "Gram"
        )

    if latest_row is None:
        st.markdown(
            f"""
<div class="empty-upload-state inventory-empty-state">
    <div class="empty-upload-icon">📦</div>
    <div class="empty-upload-title">Belum Ada Data Forecasting untuk {selected_inventory_kopi}</div>
    <div class="empty-upload-subtitle">
        Jalankan forecasting terlebih dahulu menggunakan model rekomendasi <b>{recommended_model_name}</b> agar menu persediaan dapat menampilkan indikator stock.
    </div>
</div>
""",
            unsafe_allow_html=True
        )
        return

    render_inventory_stock_indicator(
        selected_inventory_kopi,
        latest_row,
        stock_now,
        safety_stock,
        minimum_stock,
        maximum_stock
    )

    render_inventory_coverage_forecast(
        selected_inventory_kopi,
        latest_row,
        stock_now,
        safety_stock
    )

    render_inventory_history_table(
        selected_inventory_kopi,
        model_name=recommended_model_name
    )

def render_forecast_history_page(selected_kopi, recommended_model_name=None):
    render_persediaan_page(selected_kopi, recommended_model_name)


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
            "Forecasting",
            "Persediaan"
        ],

        icons=[
            "database-fill",
            "bar-chart-fill",
            "graph-up-arrow",
            "box-seam"
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

    if st.session_state.get("page_override"):
        selected = st.session_state["page_override"]

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
    # Dipindahkan ke halaman utama agar pilihan kopi
    # berada langsung pada card Jenis/Biji Kopi.
    # =====================================

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
# PILIHAN JENIS KOPI GLOBAL
# Jenis kopi tidak lagi dipilih dari sidebar.
# Nilai ini mengikuti selectbox pada halaman utama.
# =========================================
selected_kopi = st.session_state.get(
    "selected_kopi_global",
    jenis_kopi[0]
)

if selected_kopi not in jenis_kopi:
    selected_kopi = jenis_kopi[0]
    st.session_state["selected_kopi_global"] = selected_kopi

# =========================================
# HALAMAN KELOLA DATABASE
# Tidak membutuhkan upload dataset karena data dibaca langsung dari Supabase.
# =========================================
if selected == "Kelola Database":

    render_database_management_page()

    st.stop()

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
# DETEKSI PERUBAHAN DATASET
# Cache preprocessing dan training harus direset saat user upload dataset baru.
# =========================================
current_file_signature = get_uploaded_file_signature(uploaded_file)

if st.session_state.get("uploaded_file_signature") != current_file_signature:

    st.session_state["uploaded_file_signature"] = current_file_signature
    st.session_state.df_weekly = None
    st.session_state.model_results = {}
    st.session_state.autosaved_forecast_keys = set()
    st.session_state.pop("inventory_form_open", None)
    st.session_state.pop("inventory_action_mode", None)

    try:
        st.cache_data.clear()
    except Exception:
        pass


# =========================================
# PREPROCESSING DATASET
# HANYA SEKALI PER DATASET
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
# AUTOSAVE FORECAST SETELAH DATASET/TRAINING BERUBAH
# Forecast XGBoost dan Random Forest disimpan langsung ke Supabase
# sebelum halaman Persediaan dirender, sehingga stock terbaru tidak menunggu
# user pindah ke menu Forecasting terlebih dahulu.
# =========================================
autosave_signature = st.session_state.get("uploaded_file_signature", "no_dataset")
autosave_key = f"{autosave_signature}_{selected_kopi}"

if autosave_key not in st.session_state.autosaved_forecast_keys:

    try:
        forecast_xgb_autosave = forecasting_4_weeks(
            feature_df,
            xgb_result["final_model"]
        )

        forecast_rf_autosave = forecasting_4_weeks(
            feature_df,
            rf_result["final_model"]
        )

        save_forecast_to_db(
            selected_kopi,
            "XGBoost",
            forecast_xgb_autosave,
            feature_df
        )

        save_forecast_to_db(
            selected_kopi,
            "Random Forest",
            forecast_rf_autosave,
            feature_df
        )

        st.session_state.autosaved_forecast_keys.add(autosave_key)
        st.session_state["inventory_refresh_token"] = datetime.now().timestamp()

        try:
            st.cache_data.clear()
        except Exception:
            pass

        if selected == "Persediaan":
            st.rerun()

    except Exception as exc:
        if is_database_connection_error(exc):
            render_database_connection_error()
            st.stop()
        else:
            st.error(str(exc))
            st.stop()

# Rekomendasi model global untuk halaman Forecasting dan Persediaan.
recommended_model_name = best_model_name

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
        st.markdown("", unsafe_allow_html=True)

    model_option = st.session_state.get(
        "eval_model_select_card",
        "XGBoost"
    )

    if model_option not in ["XGBoost", "Random Forest"]:
        model_option = "XGBoost"

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

        st.selectbox(
            "Pilih jenis kopi:",
            jenis_kopi,
            index=jenis_kopi.index(selected_kopi),
            key="selected_kopi_global",
            label_visibility="visible"
        )

    with top_info_2:
        render_eval_metric_card(
            "🏆",
            "Model",
            model_option,
            "Model yang dipilih"
        )

        model_option = st.selectbox(
            "Pilih jenis model:",
            [
                "XGBoost",
                "Random Forest"
            ],
            index=0 if model_option == "XGBoost" else 1,
            key="eval_model_select_card",
            label_visibility="visible"
        )

        selected_result = (
            xgb_result
            if model_option == "XGBoost"
            else rf_result
        )

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
# HALAMAN RIWAYAT FORECASTING
# =========================================
elif selected == "Persediaan":

    render_forecast_history_page(selected_kopi, recommended_model_name)


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
        [2.7, 1]
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

    inventory_metrics = calculate_inventory_metrics(
        feature_df,
        selected_kopi
    )

    safety_stock = inventory_metrics["safety_stock"]
    minimum_stock = inventory_metrics["minimum_stock"]
    maximum_stock = inventory_metrics["maximum_stock"]
    lead_time = inventory_metrics["lead_time"]

    forecast_values_for_stock = (
        forecast_df["Forecast"]
        .astype(float)
        .tolist()
    )

    while len(forecast_values_for_stock) < 4:
        forecast_values_for_stock.append(0.0)

    lead_time_forecast = sum(
        forecast_values_for_stock[:lead_time]
    )

    # =====================================
    # TOP FORECAST CARDS
    # Urutan: Biji Kopi, Model, Periode Forecast, Total 1 Bulan
    # =====================================
    f1, f2, f3, f4 = st.columns(4)

    with f1:
        render_eval_metric_card(
            "☕",
            "Biji Kopi",
            selected_kopi,
            f"Kategori {kategori_kopi}"
        )

        st.selectbox(
            "Pilih jenis kopi:",
            jenis_kopi,
            index=jenis_kopi.index(selected_kopi),
            key="selected_kopi_global",
            label_visibility="visible"
        )

    with f2:
        model_card_value = (
            f"⭐ {model_forecast_option}"
            if model_forecast_option == recommended_model_name
            else model_forecast_option
        )

        render_eval_metric_card(
            "🏆",
            "Model",
            model_card_value,
            "Model yang dipilih"
        )

        model_forecast_option = st.selectbox(
            label="Pilih jenis model:",
            options=[
                "XGBoost",
                "Random Forest"
            ],
            index=0 if model_forecast_option == "XGBoost" else 1,
            label_visibility="visible",
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

    with f3:
        render_eval_metric_card(
            "📅",
            "Periode Forecast",
            f"{forecast_period} Minggu",
            "Mingguan"
        )

    with f4:
        render_eval_metric_card(
            "↗",
            "Total 1 Bulan",
            f"{total_forecast:,.0f}",
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

    st.write("")

    # Simpan hasil forecasting dari kedua model sekaligus ke database.
    # XGBoost dan Random Forest tetap tersimpan agar histori mengikuti pilihan model user.
    forecast_xgb_history = forecasting_4_weeks(
        feature_df,
        xgb_result["final_model"]
    )

    forecast_rf_history = forecasting_4_weeks(
        feature_df,
        rf_result["final_model"]
    )

    save_forecast_to_db(
        selected_kopi,
        "XGBoost",
        forecast_xgb_history,
        feature_df
    )

    save_forecast_to_db(
        selected_kopi,
        "Random Forest",
        forecast_rf_history,
        feature_df
    )

    # =====================================
    # RIWAYAT FORECASTING
    # Ditempatkan di bagian paling bawah halaman Forecasting
    # =====================================
    render_forecast_history_graph_table(selected_kopi, model_forecast_option)
