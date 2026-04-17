import streamlit as st
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="Planogram Automation", layout="wide")
st.title("📊 Planogram Automation")

# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs([
    "1️⃣ Data Preparation & Build", 
    "2️⃣ Rules & PDT",
    "3️⃣ Planogram"
])

# =========================
# 🔥 FUNCTION: CLEAN PLU (WAJIB)
# =========================
def clean_plu(series):
    return (
        series.astype(str)
        .str.replace(r"\D", "", regex=True)
        .str.lstrip("0")
    )

# =========================
# FUNCTION: Load Master
# =========================
@st.cache_data
def load_master_filtered(plu_list):
    df = pd.read_parquet("master_plu.parquet")
    df.columns = df.columns.str.strip().str.upper()

    df["PLU"] = clean_plu(df["PLU"])
    df = df[df["PLU"].isin(plu_list)]

    return df

# =========================
# FUNCTION: Load Segmentasi
# =========================
@st.cache_data
def load_segmentasi_filtered(plu_list):
    df = pd.read_parquet("master_segmentasi.parquet")
    df["PLU"] = clean_plu(df["PLU"])
    df = df[df["PLU"].isin(plu_list)]
    return df

# =========================
# FUNCTION: Load LP3
# =========================
@st.cache_data
def load_lp3_filtered(plu_list):
    df = pd.read_parquet("LP3_Februari 2026.parquet")
    df["PLU"] = clean_plu(df["PLU"])
    df = df[df["PLU"].isin(plu_list)]
    return df

# =========================
# FUNCTION: Load Brand Rokok
# =========================
@st.cache_data
def load_brand_rokok(plu_list):
    df = pd.read_parquet("brand_rokok.parquet")
    df["PLU"] = clean_plu(df["PLU"])
    df = df[df["PLU"].isin(plu_list)]
    return df

# =========================
# TAB 1
# =========================
with tab1:
    st.header("📥 Upload PLU")
    uploaded_file = st.file_uploader("Upload file", type=["xlsx"])

    if uploaded_file is not None:
        df_input = pd.read_excel(uploaded_file)
        df_input.columns = df_input.columns.str.strip()

        if "PLU" not in df_input.columns:
            st.error("Kolom 'PLU' tidak ditemukan")
            st.stop()

        # 🔥 CLEAN PLU INPUT
        df_input["PLU"] = clean_plu(df_input["PLU"])

        plu_list = df_input["PLU"].unique().tolist()

        # =========================
        # MASTER PLU
        # =========================
        df_master = load_master_filtered(plu_list)

        df_master = df_master[[
            "PLU","DESCP","BARCODE","KODE PRINSIPAL","NAMA PRINSIPAL",
            "MEREK","DIVISI","KODE SUBDEPT","NAMA SUBDEPT",
            "KODE KATAGORI","NAMA KATAGORI",
            "KODE SUBKATAGORI","DESCP SUBKATAGORI",
            "PANJANG PCS","LEBAR PCS","TINGGI PCS"
        ]]

        df_result = df_input.merge(df_master, on="PLU", how="left")

        # =========================
        # CLEANING
        # =========================
        df_result["BARCODE"] = (
            df_result["BARCODE"]
            .astype(str)
            .str.replace(r"[^\d]", "", regex=True)
        )

        # Principal & hierarchy
        df_result["Principal"] = (
            df_result["KODE PRINSIPAL"].astype(str).str.strip()
            + " - " +
            df_result["NAMA PRINSIPAL"].astype(str).str.strip()
        )

        df_result["Subdept"] = (
            df_result["KODE SUBDEPT"].astype(str) + " - " +
            df_result["NAMA SUBDEPT"].astype(str)
        )

        df_result["Category"] = (
            df_result["KODE KATAGORI"].astype(str) + " - " +
            df_result["NAMA KATAGORI"].astype(str)
        )

        df_result["Subcategory"] = (
            df_result["KODE SUBKATAGORI"].astype(str) + " - " +
            df_result["DESCP SUBKATAGORI"].astype(str)
        )

        df_result = df_result.drop(columns=[
            "KODE PRINSIPAL","NAMA PRINSIPAL",
            "KODE SUBDEPT","NAMA SUBDEPT",
            "KODE KATAGORI","NAMA KATAGORI",
            "KODE SUBKATAGORI","DESCP SUBKATAGORI"
        ])

        df_result = df_result.rename(columns={"MEREK": "Brand"})

        # =========================
        # SEGMENTASI
        # =========================
        df_segmentasi = load_segmentasi_filtered(plu_list)

        df_segmentasi = df_segmentasi[[
            "PLU","Section","UKURAN","UOM","Packtype",
            "Packsize","Isi Kemasan","Variant/Flavor",
            "Function","Price Segment","User"
        ]]

        df_result = df_result.merge(df_segmentasi, on="PLU", how="left")

        df_result = df_result.rename(columns={"UKURAN": "Size"})

        # =========================
        # MARKET SHARE
        # =========================
        df_ms_category = pd.read_excel("MARKET SHARE CATEGORY.xlsx")
        df_ms_subcategory = pd.read_excel("MARKET SHARE SUBCATEGORY.xlsx")
        df_ms_brand = pd.read_excel("MARKET SHARE BRAND.xlsx")

        df_ms_category = df_ms_category.rename(columns={"CATEGORY": "Category"})

        df_result = df_result.merge(df_ms_category, on="Category", how="left")
        df_result = df_result.merge(df_ms_subcategory, on="Subcategory", how="left")
        df_result = df_result.merge(df_ms_brand, on="Brand", how="left")

        # =========================
        # LP3 PERFORMANCE
        # =========================
        df_lp3 = load_lp3_filtered(plu_list)

        df_lp3 = df_lp3[[
            "PLU","PERFORMANCE","Avg NS/Bln","NS/JTD",
            "STD (QTY/JTJ)","STD (QTY/JTD)",
            "JTD","JTJ","JTA","%(JTJ/JTD)","GM Value"
        ]]

        df_lp3 = df_lp3.rename(columns={
            "%(JTJ/JTD)": "PENETRASI"
        })

        df_result = df_result.merge(df_lp3, on="PLU", how="left")

        # =========================
        # BRAND ROKOK
        # =========================
        df_brand = load_brand_rokok(plu_list)

        df_brand = df_brand[["PLU","MERK","BRAND ROKOK"]]

        df_result = df_result.merge(df_brand, on="PLU", how="left")

        # =========================
        # SAVE SESSION
        # =========================
        st.session_state["df_result"] = df_result

        # =========================
        # OUTPUT
        # =========================
        st.header("📊 Data Preparation & Build")
        st.dataframe(df_result)

        # =========================
        # DOWNLOAD
        # =========================
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_result.to_excel(writer, index=False)

        st.download_button(
            label="📥 Download Excel",
            data=output.getvalue(),
            file_name="Data_Preparation.xlsx"
        )

# =========================
# TAB 2
# =========================
with tab2:
    st.header("🌳 Setting Purchase Decision Tree 🌳")

    # =========================
    # INPUT: SECTION
    # =========================
    section_options = [
        "Pilih Section",
        "Backwall",
        "Chiller",
        "Snack",
        "Meja Kasir",
        "Bisc & Confect",
        "House Hold",
        "Medicine & Baby Care",
        "Milk & Diapers",
        "Misc & Stationary"
    ]

    selected_section = st.selectbox(
        "Pilih Section",
        section_options
    )

    # =========================
    # VALIDASI
    # =========================
    if selected_section == "Pilih Section":
        st.warning("⚠️ Pilih Section terlebih dahulu")
    else:
        st.success(f"Section dipilih: {selected_section}")
        st.session_state["section"] = selected_section

    # =========================
    # PREVIEW DATA
    # =========================
    if "df_result" not in st.session_state:
        st.warning("Silakan load data di Tab 1 terlebih dahulu")

    elif selected_section == "Pilih Section":
        st.warning("⚠️ Pilih Section terlebih dahulu")

    else:
        df_result = st.session_state["df_result"]
        st.subheader("Master Item Preview")
        st.dataframe(df_result)
        
        # =========================
        # INPUT: STRUKTUR RAK
        # =========================
        st.subheader("🧱 Konfigurasi Rak")

        jumlah_rak = st.number_input(
            "Jumlah Rak",
            min_value=1,
            max_value=20,
            value=1,
            step=1
        )

        st.session_state["jumlah_rak"] = jumlah_rak

        # =========================
        # INPUT DETAIL PER RAK
        # =========================
        rack_config = []

        st.write("### Detail Rak")

        for i in range(1, jumlah_rak + 1):
            st.markdown(f"#### Rak {i}")

            col1, col2, col3 = st.columns(3)

            with col1:
                jumlah_shelf = st.number_input(
                    f"Jumlah Shelving (Rak {i})",
                    min_value=1,
                    max_value=10,
                    value=3,
                    key=f"shelf_{i}"
                )

            with col2:
                tinggi_rak = st.number_input(
                    f"Tinggi Rak (cm) - Rak {i}",
                    min_value=10.0,
                    max_value=300.0,
                    value=150.0,
                    step=1.0,
                    key=f"tinggi_{i}"
                )

            with col3:
                lebar_shelf = st.number_input(
                    f"Lebar Shelving (cm) - Rak {i}",
                    min_value=10.0,
                    max_value=200.0,
                    value=100.0,
                    step=1.0,
                    key=f"lebar_{i}"
                )

            rack_config.append({
                "Rak": f"Rak_{i}",
                "Jumlah Shelving": jumlah_shelf,
                "Tinggi Rak (cm)": tinggi_rak,
                "Lebar Shelving (cm)": lebar_shelf
            })

        # =========================
        # SIMPAN KE SESSION
        # =========================
        df_rack_config = pd.DataFrame(rack_config)
        st.session_state["rack_config"] = df_rack_config

        # =========================
        # PREVIEW
        # =========================
        st.subheader("Preview Konfigurasi Rak")
        st.dataframe(df_rack_config)
        
        if st.button("🚀 Generate Planogram"):

            df = st.session_state["df_result"].copy()
            # df = df[df["Section"] == "Backwall"]

            df_rack = st.session_state["rack_config"]
            num_shelf = int(df_rack.iloc[0]["Jumlah Shelving"])

            total_slot = 18

            # =========================
            # SPLIT ELEKTRIK
            # =========================
            df_shelf4 = df[df["Category"] == "2415 - SIGARET ELEKTRIK"]
            df_main = df[df["Category"] != "2415 - SIGARET ELEKTRIK"]

            # =========================
            # MARKET SHARE
            # =========================
            principal_ms = (
                df_main.groupby("Principal")["YTD %CONT NS2026 Category"]
                .mean()
                .reset_index()
                .rename(columns={"YTD %CONT NS2026 Category": "MS_Principal"})
            )

            brand_ms = (
                df_main.groupby(["Principal", "BRAND ROKOK"])["YTD %CONT NS2026 Category"]
                .mean()
                .reset_index()
                .rename(columns={"YTD %CONT NS2026 Category": "MS_Brand"})
            )

            df_main = df_main.merge(principal_ms, on="Principal", how="left")
            df_main = df_main.merge(brand_ms, on=["Principal", "BRAND ROKOK"], how="left")

            # =========================
            # SORT SKU (PERFORMANCE)
            # =========================
            df_main = df_main.sort_values("NS/JTD", ascending=False)

            # =========================
            # CAPACITY CHECK
            # =========================
            normal_shelf = num_shelf - 1 if num_shelf >= 4 else num_shelf
            capacity = normal_shelf * total_slot

            if len(df_main) > capacity:
                df_display = df_main.iloc[:capacity]
                df_not_display = df_main.iloc[capacity:]
            else:
                df_display = df_main.copy()
                df_not_display = pd.DataFrame()

            # =========================
            # RANK PRINCIPAL
            # =========================
            principal_order = (
                principal_ms.sort_values("MS_Principal", ascending=False)["Principal"]
                .tolist()
            )

            from collections import defaultdict

            principal_brand_map = defaultdict(list)

            brand_rank = (
                brand_ms.sort_values(
                    ["Principal", "MS_Brand"],
                    ascending=[True, False]
                )
            )

            for _, row in brand_rank.iterrows():
                principal_brand_map[row["Principal"]].append(row["BRAND ROKOK"])

            # =========================
            # SLOT ALLOCATION
            # =========================
            principal_ms_series = principal_ms.set_index("Principal")["MS_Principal"]

            principal_slot = {}
            for p in principal_order:
                share = principal_ms_series[p] / principal_ms_series.sum()
                principal_slot[p] = max(1, int(round(share * total_slot)))

            column_assignment = []

            for p in principal_order:
                brands = principal_brand_map[p]
                n_col = principal_slot[p]

                per_brand = max(1, n_col // len(brands))

                for b in brands:
                    for _ in range(per_brand):
                        column_assignment.append((p, b))

            while len(column_assignment) < total_slot:
                column_assignment.append(column_assignment[-1])

            column_assignment = column_assignment[:total_slot]

            # =========================
            # GRID
            # =========================
            grid = [[None for _ in range(total_slot)] for _ in range(num_shelf)]

            # =========================
            # FILL GRID (FLEXIBLE BLOCKING)
            # =========================
            used_index = set()

            for col_idx, (p, b) in enumerate(column_assignment):

                df_brand = df_display[
                    (df_display["Principal"] == p) &
                    (df_display["BRAND ROKOK"] == b)
                ]

                skus = df_brand.to_dict("records")

                for row in range(num_shelf):

                    if row + 1 == 4:
                        continue

                    # ambil SKU yang belum dipakai
                    sku = None
                    for i, s in enumerate(skus):
                        if i not in used_index:
                            sku = s
                            used_index.add(i)
                            break

                    # kalau habis → ambil dari global terbaik
                    if sku is None:
                        for i, s in enumerate(df_display.to_dict("records")):
                            if i not in used_index:
                                sku = s
                                used_index.add(i)
                                break

                    grid[row][col_idx] = sku

            # =========================
            # SHELF 4 (NO LIMIT)
            # =========================
            if num_shelf >= 4:
                for col_idx in range(len(df_shelf4)):
                    if col_idx < total_slot:
                        grid[3][col_idx] = df_shelf4.iloc[col_idx].to_dict()

            # =========================
            # BUILD PLANOGRAM
            # =========================
            planogram = []

            for row_idx in range(num_shelf):
                for col_idx in range(total_slot):

                    item = grid[row_idx][col_idx]

                    if item is None:
                        continue

                    planogram.append({
                        "Rak": 1,
                        "Shelving": row_idx + 1,
                        "No Urut": col_idx + 1,
                        "Posisi": "A",
                        "PLU": item["PLU"],
                        "Descp": item["DESCP"],
                        "Principal": item["Principal"],
                        "Brand": item["BRAND ROKOK"],
                        "Tier Kiri-Kanan": 1
                    })

            df_planogram = pd.DataFrame(planogram)

            # =========================
            # SAVE
            # =========================
            st.session_state["planogram"] = df_planogram
            st.session_state["not_display"] = df_not_display

            st.success("✅ Planogram berhasil dibuat!")
        

# =========================
# TAB 3
# =========================
with tab3:
    if "planogram" in st.session_state:
        st.subheader("📦 Planogram")
        st.dataframe(st.session_state["planogram"])

    if "not_display" in st.session_state:
        st.subheader("⚠️ Item Tidak Masuk Planogram")
        st.dataframe(st.session_state["not_display"])