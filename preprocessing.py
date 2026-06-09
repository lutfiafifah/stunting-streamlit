import pandas as pd
import re

def convert_usia_ke_bulan(usia_str):
    try:
        tahun = int(re.search(r'(\d+)\s*Tahun', usia_str).group(1))
    except:
        tahun = 0
    try:
        bulan = int(re.search(r'(\d+)\s*Bulan', usia_str).group(1))
    except:
        bulan = 0

    return tahun * 12 + bulan


def load_and_clean_data(path):
    # baca excel tanpa header
    df = pd.read_excel(path, header=None)

    print("=== DATA AWAL ===")
    print(df.head())

    # ambil header dari baris ke-1 (sesuaikan kalau beda)
    df.columns = df.iloc[1]
    df = df[2:].reset_index(drop=True)

    print("=== KOLOM ===")
    print(df.columns)

    # ============================
    # PROSES DATA
    # ============================

    # umur (convert ke bulan)
    df['umur'] = df['Usia Saat Ukur'].astype(str).apply(convert_usia_ke_bulan)

    # jenis kelamin
    df['jenis_kelamin'] = df['JK'].map({
        'L': 'laki-laki',
        'P': 'perempuan'
    })

    # tinggi badan (pakai TB Lahir / ganti kalau ada TB lain)
    df['tinggi_badan'] = pd.to_numeric(df['TB Lahir'], errors='coerce')

    # status gizi dari TB/U
    df['status_gizi'] = df['TB/U'].astype(str).str.lower().str.strip()

    # ambil kolom final
    df = df[['umur', 'jenis_kelamin', 'tinggi_badan', 'status_gizi']]

    # hapus data kosong
    df = df.dropna()

    print("=== DATA BERSIH ===")
    print(df.head())

    return df


def save_to_csv(df, output_path):
    df.to_csv(output_path, index=False)


# ============================
# MAIN PROGRAM
# ============================
if __name__ == "__main__":
    df = load_and_clean_data("data/DATA STUNTING.xls")
    save_to_csv(df, "data/data_stunting.csv")

    print("✅ DATA BERHASIL DIUBAH KE CSV")