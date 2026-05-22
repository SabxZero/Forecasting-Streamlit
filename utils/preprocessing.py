# utils/preprocessing.py

import pandas as pd
import numpy as np


# =========================================
# MAPPING GRAM PRODUK
# =========================================
gram_mapping = {

    # OPLET
    "OPLET MINI CLASSIC":210,
    "OPLET MINI GIFTPACK":105,
    "OPLET BESAR CLASSIC":300,
    "OBK BUNGKUS":600,
    "OPLET PRAPATAN 225G":225,
    "KOPI + GULA CAP OPLET":160,
    "KOPI GULA STRONG (RENCENG)":90,
    "KOPI GULA ASOOYY":140,
    "KOPI GULA XMANGAT":120,
    "OSU RENCENG":30,

    # PUSAKA
    "PUSAKA 100G":100,
    "PUSAKA SPECIALTY COFFEE PRPT":250,
    "PUSAKA 500G":500,
    "PUSAKA 1KG":1000,

    # KERIS
    "KERIS 100G":100,
    "KERIS SPECIALTY COFFEE 250GR":250,
    "KERIS 500G":500,
    "KERIS 1KG":1000,

    # GAYO
    "ACEH GAYO 100G":100,
    "GAYO SPECIALTY COFFEE PRPT":250,
    "ACEH GAYO 500G":500,
    "ACEH GAYO 1KG":1000,

    # JAVA
    "JAVA PREANGER 100G":100,
    "JAVA PREANGER SPECIALTY COFFEE":250,
    "JAVA PREANGER 500G":500,
    "JAVA PREANGER 1KG":1000,

    # MANDAILING
    "MANDHAILING 100G":100,
    "MANDHAILING SPECIALTY COFFEE":250,
    "MANDHAILING 500G":500,
    "MANDHAILING 1KG":1000,

    # KINTAMANI
    "BALI KINTAMANI 100G":100,
    "BALI KINTAMANI SPECIALTY COFFEE":250,
    "BALI KINTAMANI 500G":500,
    "BALI KINTAMANI 1KG":1000,

    # FLORES
    "FLORES BAJAWA 100G":100,
    "FLORES BAJAWA SPECIALTY COFFEE":250,
    "FLORES BAJAWA 500G":500,
    "FLORES BAJAWA 1KG":1000,

    # PAPUA
    "PAPUA WAMENA 100G":100,
    "PAPUA WAMENA SPECIALTY COFFEE":250,
    "PAPUA WAMENA 500G":500,
    "PAPUA WAMENA 1KG":1000,

    # TORAJA
    "TORAJA 100G":100,
    "TORAJA KALOSSI SPECIALTY COFFEE":250,
    "TORAJA 500G":500,
    "TORAJA 1KG":1000
}


# =========================================
# PREPROCESSING UTAMA
# =========================================
def preprocessing_data(uploaded_file):

    # =============================
    # READ DATA
    # =============================
    df = pd.read_excel(uploaded_file)

    # =============================
    # HANDLE MISSING VALUE
    # =============================
    df = df.dropna(subset=['Nama Barang'])

    df['Tanggal'] = pd.to_datetime(
        df['Tanggal'],
        errors='coerce'
    )

    df = df.dropna(subset=['Tanggal'])

    # =============================
    # HAPUS DUPLIKAT
    # =============================
    df = df.drop_duplicates()

    # =============================
    # NORMALISASI NAMA PRODUK
    # =============================
    df['Nama Barang'] = (
        df['Nama Barang']
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # =============================
    # PENYESUAIAN NAMA PRODUK
    # =============================
    df['Nama Barang'] = df['Nama Barang'].replace({

        'OPLET MINI CLASSIC SUPER':
        'OPLET MINI CLASSIC'

    })

    # =============================
    # HAPUS PRODUK TIDAK DIGUNAKAN
    # =============================
    produk_dihapus = [

        'KALENDER HARIAN',
        'MUG ENAMEL',
        'BUBUK / BIJI OPLET 250 GRAM'

    ]

    df = df[
        ~df['Nama Barang'].isin(produk_dihapus)
    ]

    # =============================
    # FILTER PRODUK SESUAI MAPPING
    # =============================
    df = df[
        df['Nama Barang'].isin(
            gram_mapping.keys()
        )
    ]

    # =============================
    # MAPPING GRAM
    # =============================
    df['Gram'] = df['Nama Barang'].map(
        gram_mapping
    )

    df['Total_Gram'] = (
        df['Qty'] * df['Gram']
    )

    # =============================
    # HAPUS KOLOM TIDAK DIGUNAKAN
    # =============================
    kolom_hapus = ['Harga', 'Jumlah']

    for kolom in kolom_hapus:
        if kolom in df.columns:
            df = df.drop(columns=kolom)

    # =============================
    # MEMBUAT ATRIBUT MINGGU
    # =============================
    df['minggu'] = (
        df['Tanggal']
        .dt.to_period('W')
        .apply(lambda r: r.start_time)
    )

    # =============================
    # INISIALISASI JENIS KOPI
    # =============================
    jenis_kopi = [
        'Robusta',
        'Robusta_Highblend',
        'Gayo',
        'Ciwidey',
        'Mandailing',
        'Kintamani',
        'Flores',
        'Papua',
        'Toraja'
    ]

    for kopi in jenis_kopi:
        df[kopi] = 0.0

    # =============================
    # KLASIFIKASI PRODUK
    # =============================
    for i, row in df.iterrows():

        produk = str(row['Nama Barang'])
        gram = row['Total_Gram']

        # ROBUSTA
        if (
            'OPLET MINI CLASSIC' in produk or
            'OPLET MINI GIFTPACK' in produk or
            'OPLET BESAR CLASSIC' in produk or
            'OBK BUNGKUS' in produk or
            'OPLET PRAPATAN 225G' in produk or
            'KOPI + GULA CAP OPLET' in produk or
            'KOPI GULA STRONG (RENCENG)' in produk or
            'KOPI GULA ASOOYY' in produk or
            'KOPI GULA XMANGAT' in produk or
            'OSU RENCENG' in produk
        ):

            df.at[i, 'Robusta'] = gram

        # ROBUSTA HIGHBLEND
        elif (
            'PUSAKA 100G' in produk or
            'PUSAKA SPECIALTY COFFEE PRPT' in produk or
            'PUSAKA 500G' in produk or
            'PUSAKA 1KG' in produk
        ):

            df.at[i, 'Robusta_Highblend'] = gram

        # KERIS BLEND
        elif (
            'KERIS 100G' in produk or
            'KERIS SPECIALTY COFFEE 250GR' in produk or
            'KERIS 500G' in produk or
            'KERIS 1KG' in produk
        ):

            df.at[i, 'Robusta_Highblend'] = gram * 0.30
            df.at[i, 'Gayo'] = gram * 0.70

        # GAYO
        elif (
            'ACEH GAYO 100G' in produk or
            'GAYO SPECIALTY COFFEE PRPT' in produk or
            'ACEH GAYO 500G' in produk or
            'ACEH GAYO 1KG' in produk
        ):

            df.at[i, 'Gayo'] = gram

        # CIWIDEY
        elif (
            'JAVA PREANGER 100G' in produk or
            'JAVA PREANGER SPECIALTY COFFEE' in produk or
            'JAVA PREANGER 500G' in produk or
            'JAVA PREANGER 1KG' in produk
        ):

            df.at[i, 'Ciwidey'] = gram

        # MANDAILING
        elif (
            'MANDHAILING 100G' in produk or
            'MANDHAILING SPECIALTY COFFEE' in produk or
            'MANDHAILING 500G' in produk or
            'MANDHAILING 1KG' in produk
        ):

            df.at[i, 'Mandailing'] = gram

        # KINTAMANI
        elif (
            'BALI KINTAMANI 100G' in produk or
            'BALI KINTAMANI SPECIALTY COFFEE' in produk or
            'BALI KINTAMANI 500G' in produk or
            'BALI KINTAMANI 1KG' in produk
        ):

            df.at[i, 'Kintamani'] = gram

        # FLORES
        elif (
            'FLORES BAJAWA 100G' in produk or
            'FLORES BAJAWA SPECIALTY COFFEE' in produk or
            'FLORES BAJAWA 500G' in produk or
            'FLORES BAJAWA 1KG' in produk
        ):

            df.at[i, 'Flores'] = gram

        # PAPUA
        elif (
            'PAPUA WAMENA 100G' in produk or
            'PAPUA WAMENA SPECIALTY COFFEE' in produk or
            'PAPUA WAMENA 500G' in produk or
            'PAPUA WAMENA 1KG' in produk
        ):

            df.at[i, 'Papua'] = gram

        # TORAJA
        elif (
            'TORAJA 100G' in produk or
            'TORAJA KALOSSI SPECIALTY COFFEE' in produk or
            'TORAJA 500G' in produk or
            'TORAJA 1KG' in produk
        ):

            df.at[i, 'Toraja'] = gram

    # =============================
    # AGREGASI MINGGUAN
    # =============================
    df_weekly = df.groupby('minggu')[
        jenis_kopi
    ].sum().reset_index()

    # =============================
    # HAPUS MINGGU TERAKHIR
    # =============================
    df_weekly = df_weekly.iloc[:-1]

    # =============================
    # PEMBULATAN
    # =============================
    df_weekly = df_weekly.round(0)

    return df_weekly