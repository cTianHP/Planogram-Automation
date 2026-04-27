import streamlit as st
import pandas as pd
from io import BytesIO
# from utils.ui_styles import render_fixed_header
from utils.ui_styles import apply_custom_theme

st.set_page_config(page_title="Planogram Automation", layout="wide")
apply_custom_theme()
st.title("Planogram Automation")

# ==================
# INIT SESSION STATE
# ==================
if "df_result" not in st.session_state:
    st.session_state["df_result"] = pd.DataFrame()

if "rack_config" not in st.session_state:
    st.session_state["rack_config"] = pd.DataFrame()

if "df_tier" not in st.session_state:
    st.session_state["df_tier"] = pd.DataFrame()

if "df_affinity" not in st.session_state:
    st.session_state["df_affinity"] = pd.DataFrame()

if "df_category_rank" not in st.session_state:
    st.session_state["df_category_rank"] = pd.DataFrame()

if "packtype_priority_map" not in st.session_state:
    st.session_state["packtype_priority_map"] = {}

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
        
        # =========================
        # SAFE CONVERT FUNCTION
        # =========================
        def safe_int_to_str(series):
            return (
                pd.to_numeric(series, errors="coerce")
                .fillna(0)
                .astype(int)
                .astype(str)
            )
        # =========================
        # Principal & hierarchy (FIXED)
        # =========================

        df_result["Principal"] = (
            df_result["KODE PRINSIPAL"].astype(str).str.strip()
            + " - " +
            df_result["NAMA PRINSIPAL"].astype(str).str.strip()
        )

        df_result["Subdept"] = (
            safe_int_to_str(df_result["KODE SUBDEPT"]) + " - " +
            df_result["NAMA SUBDEPT"].astype(str)
        )

        df_result["Category"] = (
            safe_int_to_str(df_result["KODE KATAGORI"]) + " - " +
            df_result["NAMA KATAGORI"].astype(str)
        )
        
        df_result["Subcategory"] = (
            df_result["KODE SUBKATAGORI"].astype(str).str.strip()
            + " - " +
            df_result["DESCP SUBKATAGORI"].astype(str).str.strip()
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
        st.dataframe(df_result, hide_index=True)

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
        # INPUT: SECTION
        # =========================
        section_options = [
            "Pilih Section",
            "Backwall",
            "Chiller",
            "Noodle",
            "Breakfast",
            "Snack",
            "Meja Kasir",
            "Bisc & Confect",
            "Milk & Diapers",
            "Misc & Stationary",
            "House Hold",
            "Medicine & Baby Care",
            "Personal Care",
            "Sanitary Adult"
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
        # LOAD AFFINITY
        # =========================
        def load_affinity(section, valid_categories):
            try:
                file_path = f"affinity/{section}.xlsx"
                df_aff = pd.read_excel(file_path)

                df_aff.columns = df_aff.columns.str.strip()

                # 🔥 FILTER ROW (Category utama)
                df_aff = df_aff[df_aff["Category"].isin(valid_categories)]

                # 🔥 FILTER COLUMN (Category pair)
                valid_cols = ["Category"] + [
                    col for col in df_aff.columns if col in valid_categories
                ]
                df_aff = df_aff[valid_cols]

                return df_aff

            except Exception:
                st.warning(f"File affinity untuk section '{section}' tidak ditemukan")
                return pd.DataFrame()
        valid_categories = st.session_state["df_result"]["Category"].dropna().unique().tolist()
        df_affinity = load_affinity(selected_section, valid_categories)

        # simpan ke session
        st.session_state["df_affinity"] = df_affinity

        # =========================
        # PREVIEW DATA
        # =========================
        if "df_result" not in st.session_state:
            st.warning("Silakan load data di Tab 1 terlebih dahulu")
        elif selected_section == "Pilih Section":
            st.warning("⚠️ Pilih Section terlebih dahulu")
        else:
            df_result = st.session_state["df_result"]
            
            # =========================
            # INPUT: STRUKTUR RAK
            # =========================
            with st.expander("⚙️ Konfigurasi Rak", expanded=True):
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

                    # =========================
                    # DEFAULT MAX HEIGHT
                    # =========================
                    selected_section = st.session_state.get("section", "")
                    max_height_shelf = [40] * jumlah_shelf
                    if selected_section != "Chiller":
                        max_height_shelf[0] = 100 

                    # Pilih shelving yang ingin diatur
                    selected_shelves = st.multiselect(
                        f"Pilih Shelving yang ingin diatur (Rak {i})",
                        options=list(range(1, jumlah_shelf + 1)),
                        key=f"select_shelf_{i}"
                    )

                    # Input tinggi untuk shelving terpilih
                    for s in selected_shelves:
                        height_input = st.number_input(
                            f"Tinggi Maksimal Item Shelving {s} (cm) - Rak {i}",
                            min_value=5.0,
                            max_value=100.0,
                            value=30.0,
                            step=1.0,
                            key=f"max_height_r{i}_s{s}"
                        )
                        
                        # Replace default value
                        max_height_shelf[s - 1] = height_input
                    rack_config.append({
                        "Rak": f"Rak_{i}",
                        "Jumlah Shelving": jumlah_shelf,
                        "Tinggi Rak (cm)": tinggi_rak,
                        "Lebar Shelving (cm)": lebar_shelf,
                        "Max Tinggi per Shelving": max_height_shelf
                    })

                # =========================
                # SIMPAN KE SESSION
                # =========================
                df_rack_config = pd.DataFrame(rack_config)
                st.session_state["rack_config"] = df_rack_config

                # =========================
                # DATA CARD RAK
                # =========================
                df_rack = st.session_state["rack_config"]

                df_rack["Total Lebar (cm)"] = (
                    df_rack["Jumlah Shelving"] * df_rack["Lebar Shelving (cm)"]
                )

                total_shelf = df_rack["Jumlah Shelving"].sum()
                total_lebar = df_rack["Total Lebar (cm)"].sum()
                total_rak = len(df_rack)

                col1, col2, col3 = st.columns(3)

                col1.metric("Jumlah Rak", total_rak)
                col2.metric("Total Shelving", total_shelf)
                col3.metric("Total Lebar Display (cm)", f"{total_lebar:,.0f}")
                
                # =========================
                # PREVIEW
                # =========================
                st.subheader("Preview Konfigurasi Rak")
                st.dataframe(df_rack_config, hide_index=True)
            
            # =========================
            # SETTING TIER PLU
            # =========================
            with st.expander("⚙️ Setting Tier (Kiri-Kanan / Facing PLU)", expanded=True):

                df_master_item = st.session_state["df_result"][["PLU", "DESCP"]].drop_duplicates()

                # jumlah baris input
                num_rows = st.number_input(
                    "Jumlah PLU yang ingin diatur",
                    min_value=0,
                    max_value=100,
                    value=0,
                    step=1
                )

                tier_data = []

                for i in range(num_rows):
                    st.markdown(f"#### Input PLU {i+1}")

                    col1, col2, col3 = st.columns([2, 3, 1])

                    # =========================
                    # INPUT PLU (MANUAL)
                    # =========================
                    with col1:
                        input_plu = st.text_input(
                            f"PLU {i+1}",
                            key=f"plu_input_{i}"
                        )

                    # =========================
                    # AUTO DESC + VALIDASI
                    # =========================
                    with col2:
                        desc = ""
                        found = False

                        if input_plu:
                            result = df_master_item[
                                df_master_item["PLU"] == input_plu
                            ]

                            if not result.empty:
                                desc = result["DESCP"].values[0]
                                found = True
                                st.success(desc)
                            else:
                                desc = "PLU Tidak Ditemukan"
                                st.error(desc)

                        else:
                            st.info("Masukkan PLU")

                    # =========================
                    # INPUT TIER
                    # =========================
                    with col3:
                        tier = st.number_input(
                            f"Tier {i+1}",
                            min_value=1,
                            max_value=20,
                            value=1,
                            step=1,
                            key=f"tier_{i}"
                        )

                    tier_data.append({
                        "PLU": input_plu,
                        "DESCP": desc,
                        "Tier": tier,
                        "Valid": found
                    })

                # =========================
                # SAVE KE SESSION (SAFE VERSION)
                # =========================
                df_tier = pd.DataFrame(tier_data)

                # pastikan kolom selalu ada
                expected_cols = ["PLU", "DESCP", "Tier", "Valid"]
                for col in expected_cols:
                    if col not in df_tier.columns:
                        df_tier[col] = None

                # filter valid (anti error)
                if not df_tier.empty and "Valid" in df_tier.columns:
                    df_tier_valid = df_tier[df_tier["Valid"] == True].copy()
                else:
                    df_tier_valid = pd.DataFrame(columns=["PLU", "DESCP", "Tier"])

                # simpan
                st.session_state["df_tier"] = df_tier_valid

                # preview
                st.subheader("Preview Tier Setting")
                st.dataframe(df_tier_valid, hide_index=True)

# =========================
# TAB 2
# =========================
with tab2:
    if st.session_state["df_result"].empty:
        st.warning("⚠️ Silakan upload data terlebih dahulu di Tab 1")
        st.stop()
    else:
        # =========================
        # CATEGORY PERFORMANCE
        # =========================
        df = st.session_state["df_result"].copy()

        df_category_perf = (
            df[["Category", "YTD ∑NS 2026 Category"]]
            .drop_duplicates()
        )

        df_category_perf = df_category_perf.sort_values(
            by="YTD ∑NS 2026 Category", ascending=False
        ).reset_index(drop=True)
        
        # =========================
        # DETAIL PER CATEGORY
        # =========================
        df_category_detail = (
            df.groupby("Category")
            .agg(
                Jumlah_PLU=("PLU", "nunique"),
                Total_Lebar=("LEBAR PCS", "sum"),
                Avg_Lebar=("LEBAR PCS", "mean"),
                # Max_Lebar=("LEBAR PCS", "max"),
                # Avg_Tinggi=("TINGGI PCS", "mean"),
                Max_Tinggi=("TINGGI PCS", "max")
            )
            .reset_index()
        )
        
        df_final_category = df_category_perf.merge(
            df_category_detail,
            on="Category",
            how="left"
        )
        
        df_category_extra = (
                    df[[
                        "Category",
                        "YTD ∑NS 2025 Category",
                        "Subdept"
                    ]]
                    .drop_duplicates()
                )
        df_final_category = df_category_perf.merge(
            df_category_detail,
            on="Category",
            how="left"
        ).merge(
            df_category_extra,
            on="Category",
            how="left"
        )
        # =========================
        # CONTRIBUTION CATEGORY
        # =========================
        total_ns = df_final_category["YTD ∑NS 2026 Category"].sum()

        df_final_category["Contribution_Category (%)"] = (
            df_final_category["YTD ∑NS 2026 Category"] / total_ns
        ) * 100
        # =========================
        # GROWTH CATEGORY
        # =========================
        if "YTD %GROWTH NS Category" in df.columns:
            df_growth = (
                df[["Category", "YTD %GROWTH NS Category"]]
                .drop_duplicates()
            )

            df_final_category = df_final_category.merge(
                df_growth,
                on="Category",
                how="left"
            )
        # =========================
        # PREPARE AFFINITY (LONG FORMAT)
        # =========================
        df_affinity = st.session_state.get("df_affinity", pd.DataFrame())

        if not df_affinity.empty:
            df_aff_long = df_affinity.melt(
                id_vars="Category",
                var_name="Category_Pair",
                value_name="Affinity"
            )

            # optional: buang self affinity (A-A)
            df_aff_long = df_aff_long[
                df_aff_long["Category"] != df_aff_long["Category_Pair"]
            ]
        else:
            df_aff_long = pd.DataFrame()
        if not df_aff_long.empty:
            df_aff_top5 = (
                df_aff_long.sort_values(["Category", "Affinity"], ascending=[True, False])
                .groupby("Category")
                .head(5)
                .reset_index(drop=True)
            )     
            # =========================
            # FORMAT JADI LIST
            # =========================
            df_aff_top5_grouped = (
                df_aff_top5.groupby("Category")
                .apply(lambda x: list(zip(x["Category_Pair"], x["Affinity"])))
                .reset_index(name="Top_5_Affinity")
            )

            # merge ke category insight
            df_final_category = df_final_category.merge(
                df_aff_top5_grouped,
                on="Category",
                how="left"
            )
            
            # =========================
            # FIX TYPE AFFINITY (ANTI ERROR STREAMLIT)
            # =========================
            def format_affinity(x):
                if isinstance(x, list):
                    return ", ".join([f"{a} ({b:.2f})" for a, b in x])
                return ""

            df_final_category["Top_5_Affinity"] = df_final_category["Top_5_Affinity"].apply(format_affinity)
        
        if "Top_5_Affinity" not in df_final_category.columns:
            df_final_category["Top_5_Affinity"] = ""
        
        df_final_category["Contribution_Category (%)"] = (
            df_final_category["Contribution_Category (%)"]
            .astype(float)
            .round(2)
            .astype(str) + "%"
        )
        # =========================
        # FORMAT GROWTH (SAFE + BUSINESS LOGIC)
        # =========================
        if "YTD %GROWTH NS Category" in df_final_category.columns:

            def format_growth(x):
                if pd.isna(x):
                    return "-"
                if str(x).strip() == "-":
                    return "-"
                try:
                    return f"{float(x)*100:.2f}%"
                except:
                    return "-"

            df_final_category["YTD %GROWTH NS Category"] = (
                df_final_category["YTD %GROWTH NS Category"]
                .apply(format_growth)
            )
            
        # =========================
        # 🆕 SUBDEPT PERFORMANCE
        # =========================
        df_subdept_perf = (
            df_final_category.groupby("Subdept")
            .agg(
                Total_NS_Subdept=("YTD ∑NS 2026 Category", "sum")
            )
            .reset_index()
        )

        # rank subdept
        df_subdept_perf = df_subdept_perf.sort_values(
            by="Total_NS_Subdept",
            ascending=False
        ).reset_index(drop=True)

        df_subdept_perf["Rank_Subdept"] = df_subdept_perf.index + 1
        total_ns_all = df_subdept_perf["Total_NS_Subdept"].sum()

        df_subdept_perf["Contribution_Subdept (%)"] = (
            df_subdept_perf["Total_NS_Subdept"] / total_ns_all
        ) * 100
        
        df_final_category = df_final_category.merge(
            df_subdept_perf,
            on="Subdept",
            how="left"
        )
        df_final_category["Contribution_Subdept (%)"] = (
            df_final_category["Contribution_Subdept (%)"]
            .fillna(0)
            .astype(float)
            .round(2)
            .astype(str) + "%"
        )
        
        # =========================
        # 🆕 RANK PER SUBDEPT
        # =========================
        df_final_category = df_final_category.sort_values(
            by=["Subdept", "YTD ∑NS 2026 Category"],
            ascending=[True, False]
        ).reset_index(drop=True)

        df_final_category["Rank Category_in_Subdept"] = (
            df_final_category.groupby("Subdept")
            .cumcount() + 1
        )

        # =========================
        # 🆕 GLOBAL RANK
        # =========================
        df_final_category["Rank Category_in_Section"] = (
            df_final_category["YTD ∑NS 2026 Category"]
            .rank(method="dense", ascending=False)
            .fillna(0)
            .astype(int)
        )
        
        df_final_category = df_final_category.sort_values(
            by=["Rank_Subdept", "Rank Category_in_Subdept", "Rank Category_in_Section"]
        ).reset_index(drop=True)
            
        df_final_category = df_final_category[[
            "Rank_Subdept",
            "Subdept",
            "Contribution_Subdept (%)", 
            "Rank Category_in_Subdept",
            "Rank Category_in_Section",
            "Category",
            "YTD ∑NS 2025 Category",
            "YTD ∑NS 2026 Category",
            "Contribution_Category (%)",
            "YTD %GROWTH NS Category",
            "Top_5_Affinity",
            "Jumlah_PLU",
            "Total_Lebar"
        ]]
        with st.expander("📊 Category Insight", expanded=True):
            # =========================
            # FORMAT NUMBER DISPLAY
            # =========================
            df_display = df_final_category.copy()
            for col in ["YTD ∑NS 2025 Category", "YTD ∑NS 2026 Category"]:
                df_display[col] = pd.to_numeric(df_display[col], errors="coerce") \
                    .fillna(0) \
                    .apply(lambda x: f"{int(x):,}")

            # =========================
            # DISPLAY
            # =========================
            st.dataframe(df_display, hide_index=True)
            st.session_state["df_category_rank"] = df_final_category
        
            # =========================
            # 📊 KPI CARDS (UPGRADE)
            # =========================
            total_category = df_final_category["Category"].nunique()
            total_subdept = df_final_category["Subdept"].nunique()

            # =========================
            # TOP 5 CATEGORY
            # =========================
            df_top_cat = df_final_category.copy()

            df_top_cat["Contribution_num"] = pd.to_numeric(
                df_top_cat["Contribution_Category (%)"].str.replace("%", ""),
                errors="coerce"
            )

            top_5_cat = (
                df_top_cat.sort_values(by="Contribution_num", ascending=False)
                .head(5)["Category"]
                .tolist()
            )

            top_5_cat_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(top_5_cat)])

            # =========================
            # TOP 3 SUBDEPT
            # =========================
            df_top_sub = (
                df_final_category[["Subdept", "Contribution_Subdept (%)"]]
                .drop_duplicates()
            )

            df_top_sub["Contribution_num"] = pd.to_numeric(
                df_top_sub["Contribution_Subdept (%)"].str.replace("%", ""),
                errors="coerce"
            )

            top_3_sub = (
                df_top_sub.sort_values(by="Contribution_num", ascending=False)
                .head(3)["Subdept"]
                .tolist()
            )

            top_3_sub_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(top_3_sub)])

            # =========================
            # DISPLAY
            # =========================
            col2, col1, col3, col4 = st.columns(4)

            col1.metric("Total Category", total_category)
            col2.metric("Total Subdept", total_subdept)
            col3.markdown(f"**Top 5 Category**\n\n{top_5_cat_text}")
            col4.markdown(f"**Top 3 Subdept**\n\n{top_3_sub_text}")
            
            # =========================
            # 📊 VISUAL INSIGHT (FINAL FIXED)
            # =========================
            col2, col1, col3 = st.columns(3)

            # =========================
            # 🍩 DONUT CATEGORY (FINAL CLEAN)
            # =========================
            with col1:
                import matplotlib.pyplot as plt

                st.markdown("**Contribution Category (Section)**")

                df_pie_cat = df_final_category.copy()

                # =========================
                # CONVERT % → NUMERIC
                # =========================
                df_pie_cat["Contribution_num"] = pd.to_numeric(
                    df_pie_cat["Contribution_Category (%)"].str.replace("%", ""),
                    errors="coerce"
                )

                df_pie_cat = df_pie_cat.dropna(subset=["Contribution_num"])

                # =========================
                # OPTIONAL: FILTER KECIL (BIAR GA RAMAI)
                # =========================
                df_pie_cat = df_pie_cat[df_pie_cat["Contribution_num"] > 1]

                # =========================
                # OPTIONAL: SORT
                # =========================
                df_pie_cat = df_pie_cat.sort_values(
                    by="Contribution_num",
                    ascending=False
                )

                # =========================
                # FORMAT LABEL (BISA DIPENDEKKAN)
                # =========================
                labels = df_pie_cat["Category"]

                # =========================
                # FUNCTION FILTER AUTOPCT
                # =========================
                def autopct_filter(pct):
                    return f'{pct:.1f}%' if pct > 3 else ''  # 🔥 hide kecil

                # =========================
                # PLOT DONUT
                # =========================
                fig, ax = plt.subplots()

                wedges, texts, autotexts = ax.pie(
                    df_pie_cat["Contribution_num"],
                    labels=labels,
                    autopct=autopct_filter,
                    pctdistance=0.75,   # 🔥 posisi persen di tengah ring
                    labeldistance=1.1,  # 🔥 label agak keluar
                    wedgeprops={'width': 0.4},
                    textprops={'fontsize': 9}
                )

                # =========================
                # STYLE TEXT %
                # =========================
                for autotext in autotexts:
                    autotext.set_color("black")
                    autotext.set_fontsize(10)
                    autotext.set_weight("bold")

                # =========================
                # OPTIONAL: TEXT TENGAH
                # =========================
                ax.text(0, 0, "", ha='center', va='center', fontsize=10)

                ax.set_ylabel("")

                st.pyplot(fig)
                
            # =========================
            # 🥧 PIE SUBDEPT (HARD FIX)
            # =========================
            with col2:
                import matplotlib.pyplot as plt

                st.markdown("**Contribution Subdept (Section)**")

                # 🔥 PAKSA 1 BARIS PER SUBDEPT
                df_pie_sub = (
                    df_final_category[["Subdept", "Contribution_Subdept (%)"]]
                    .copy()
                )

                # ambil unik subdept saja (aman)
                df_pie_sub = df_pie_sub.drop_duplicates(subset=["Subdept"])

                # convert ke numeric
                df_pie_sub["Contribution_num"] = pd.to_numeric(
                    df_pie_sub["Contribution_Subdept (%)"].str.replace("%", ""),
                    errors="coerce"
                )

                df_pie_sub = df_pie_sub.dropna(subset=["Contribution_num"])

                # 🔥 SORT
                df_pie_sub = df_pie_sub.sort_values(
                    by="Contribution_num",
                    ascending=False
                )

                # =========================
                # 🔥 PLOT MANUAL (NO REUSE)
                # =========================
                fig, ax = plt.subplots()

                ax.pie(
                    df_pie_sub["Contribution_num"],
                    labels=df_pie_sub["Subdept"],
                    autopct='%1.1f%%'
                )

                ax.set_ylabel("")

                st.pyplot(fig)

            # =========================
            # 📋 TABLE GROWTH + RANK
            # =========================
            with col3:
                st.markdown("**Top Growth Category**")
                def color_growth(val):
                    if "NEW" in val:
                        return "color: blue"
                    try:
                        v = float(val.replace("%", ""))
                        if v > 10:
                            return "color: green"
                        elif v < 0:
                            return "color: red"
                    except:
                        return ""
                    return ""

                df_growth = df_final_category.copy()

                df_growth["Growth_num"] = pd.to_numeric(
                    df_growth["YTD %GROWTH NS Category"].str.replace("%", ""),
                    errors="coerce"
                )

                df_growth = df_growth.dropna(subset=["Growth_num"])

                df_growth = df_growth.sort_values(
                    by="Growth_num",
                    ascending=False
                ).reset_index(drop=True)

                # =========================
                # 🔥 TAMBAH RANKING
                # =========================
                df_growth["Rank"] = df_growth.index + 1

                df_growth = df_growth.head(10)

                st.dataframe(
                    df_growth[["Rank","Category","YTD %GROWTH NS Category"]], hide_index=True
                )
        
    # =========================
    # 🔗 CATEGORY AFFINITY SELECTOR (FINAL + EXCLUDE SELF)
    # =========================
    with st.expander("🔗 Category Affinity", expanded=True):
        df_aff = st.session_state.get("df_affinity", pd.DataFrame())
        df_cat = st.session_state.get("df_category_rank", pd.DataFrame())

        if df_aff.empty or df_cat.empty:
            st.warning("Data affinity atau category belum tersedia")
        else:

            # =========================
            # DEFAULT CATEGORY
            # =========================
            df_cat_temp = df_cat.copy()

            df_cat_temp["Contribution_num"] = pd.to_numeric(
                df_cat_temp["Contribution_Category (%)"].str.replace("%", ""),
                errors="coerce"
            )

            default_category = (
                df_cat_temp.sort_values(by="Contribution_num", ascending=False)
                .iloc[0]["Category"]
            )

            # =========================
            # SELECT CATEGORY
            # =========================
            category_list = df_aff["Category"].tolist()

            selected_category = st.selectbox(
                "Pilih Category",
                options=category_list,
                index=category_list.index(default_category)
                if default_category in category_list else 0
            )

            # =========================
            # AMBIL DATA
            # =========================
            df_aff_row = df_aff[df_aff["Category"] == selected_category]

            if not df_aff_row.empty:

                aff_series = df_aff_row.drop(columns=["Category"]).iloc[0]

                df_top_aff = aff_series.reset_index()
                df_top_aff.columns = ["Category_Pair", "Affinity"]

                # convert numeric
                df_top_aff["Affinity"] = pd.to_numeric(
                    df_top_aff["Affinity"],
                    errors="coerce"
                )

                df_top_aff = df_top_aff.dropna(subset=["Affinity"])

                # =========================
                # 🔥 EXCLUDE DIRI SENDIRI
                # =========================
                df_top_aff = df_top_aff[
                    df_top_aff["Category_Pair"] != selected_category
                ]

                # =========================
                # SORT & TOP 5
                # =========================
                df_top_aff = df_top_aff.sort_values(
                    by="Affinity",
                    ascending=False
                ).head(5).reset_index(drop=True)

                # =========================
                # DISPLAY
                # =========================
                st.markdown(f"##### Top 5 Affinity untuk: {selected_category}")

                cols = st.columns(5)

                for i in range(len(df_top_aff)):
                    row = df_top_aff.iloc[i]

                    rank = i + 1
                    value_pct = row["Affinity"] * 100

                    label = f"{rank}. {row['Category_Pair']}"

                    cols[i].metric(
                        label=label,
                        value=f"{value_pct:.2f}%"
                    )

            else:
                st.info("Affinity tidak ditemukan untuk category ini")
        
        
    with st.expander("🏷️ Brand Insight", expanded=True):
        df = st.session_state["df_result"].copy()

        if "YTD ∑NS 2026 Brand" in df.columns:

            df_brand_insight = (
                df.groupby(["Category", "Brand"])
                .agg(
                    Jumlah_PLU=("PLU", "nunique"),
                    Total_NS=("YTD ∑NS 2026 Brand", "first"),
                    Growth=("YTD %GROWTH NS Brand", "first")
                )
                .reset_index()
            )

            df_brand_insight = df_brand_insight.sort_values(
                by=["Category", "Total_NS"],
                ascending=[True, False]
            )

            df_brand_insight["Rank_in_Category"] = (
                df_brand_insight.groupby("Category")
                .cumcount() + 1
            )

            # =========================
            # CONTRIBUTION
            # =========================
            df_brand_insight["Contribution_Brand (%)"] = (
                df_brand_insight["Total_NS"] /
                df_brand_insight.groupby("Category")["Total_NS"].transform("sum")
            )

            df_brand_insight["Contribution_Brand (%)"] = (
                df_brand_insight["Contribution_Brand (%)"]
                .fillna(0)
                .mul(100)
                .round(2)
                .astype(str) + "%"
            )

            # =========================
            # FORMAT TOTAL_NS
            # =========================
            df_brand_insight["Total_NS"] = (
                pd.to_numeric(df_brand_insight["Total_NS"], errors="coerce")
                .fillna(0)
                .apply(lambda x: f"{int(x):,}")
            )

            # =========================
            # FORMAT GROWTH (%)
            # =========================
            def format_growth(x):
                if pd.isna(x):
                    return "-"
                if str(x).strip() == "-":
                    return "NEW"
                try:
                    return f"{float(x)*100:.2f}%"
                except:
                    return "-"

            df_brand_insight["Growth (%)"] = df_brand_insight["Growth"].apply(format_growth)

            # =========================
            # DROP RAW GROWTH COLUMN
            # =========================
            df_brand_insight = df_brand_insight.drop(columns=["Growth"])

            # =========================
            # 🎨 STYLE COLOR GROWTH
            # =========================
            def color_growth(val):
                try:
                    if val == "NEW" or val == "-":
                        return "color: gray"
                    num = float(val.replace("%", ""))
                    if num > 0:
                        return "color: green; font-weight: bold"
                    elif num < 0:
                        return "color: red; font-weight: bold"
                    else:
                        return "color: black"
                except:
                    return ""

            styled_df = df_brand_insight.style.map(
                color_growth,
                subset=["Growth (%)"]
            )

            # =========================
            # DISPLAY
            # =========================
            st.dataframe(styled_df, hide_index=True)

            st.session_state["df_brand_insight"] = df_brand_insight

        else:
            st.warning("Kolom 'YTD ∑NS 2026 Brand' tidak ditemukan")
    
    # =========================
    # 🧠 SORTING PRIORITY ENGINE (FINAL + PREVIEW + PARTIAL FIX)
    # =========================
    with st.expander("⚙️ Setting Planogram", expanded=True):

        df = st.session_state.get("df_result", pd.DataFrame()).copy()
        df_cat_rank = st.session_state.get("df_category_rank", pd.DataFrame()).copy()
        df_brand_insight = st.session_state.get("df_brand_insight", pd.DataFrame()).copy()

        if df.empty:
            st.warning("Data belum tersedia")
            st.stop()

        st.subheader("Urutan PDT")

        dimension_options = [
            "Packtype","Brand","User",
            "Subcategory","Packsize",
            "Price Segment","Variant/Flavor"
        ]

        # =========================
        # 🔥 HELPER
        # =========================
        def merge_order(user_list, default_list):
            user_list = user_list or []
            remaining = [x for x in default_list if x not in user_list]
            return user_list + remaining

        # =========================
        # 🔥 SUBDEPT ORDER
        # =========================
        subdept_list = sorted(df["Subdept"].dropna().unique())

        subdept_input = st.multiselect("Urutan Subdept (opsional)", subdept_list)

        if not df_cat_rank.empty:
            df_temp = df_cat_rank.copy()
            df_temp["contri"] = pd.to_numeric(
                df_temp["Contribution_Subdept (%)"].astype(str).str.replace("%",""),
                errors="coerce"
            )
            subdept_default = (
                df_temp.drop_duplicates("Subdept")
                .sort_values("contri", ascending=False)["Subdept"]
                .tolist()
            )
        else:
            subdept_default = subdept_list

        subdept_order = merge_order(subdept_input, subdept_default)

        # =========================
        # 🔥 CATEGORY ORDER
        # =========================
        category_list = sorted(df["Category"].dropna().unique())

        category_input = st.multiselect("Urutan Category (opsional)", category_list)

        if not df_cat_rank.empty:
            df_temp = df_cat_rank.copy()
            df_temp["contri"] = pd.to_numeric(
                df_temp["Contribution_Category (%)"].astype(str).str.replace("%",""),
                errors="coerce"
            )
            category_default = (
                df_temp.sort_values("contri", ascending=False)["Category"]
                .tolist()
            )
        else:
            category_default = category_list

        category_order = merge_order(category_input, category_default)

        # =========================
        # 🔥 BRAND ORDER
        # =========================
        if not df_brand_insight.empty:
            df_temp = df_brand_insight.copy()
            df_temp["contri"] = pd.to_numeric(
                df_temp["Contribution_Brand (%)"].astype(str).str.replace("%",""),
                errors="coerce"
            )
            brand_default = (
                df_temp.sort_values("contri", ascending=False)["Brand"]
                .tolist()
            )
        else:
            brand_default = sorted(df["Brand"].dropna().unique())

        # =========================
        # 🔥 DIMENSION ORDER
        # =========================
        selected_dims = st.multiselect(
            "Pilih urutan sorting setelah Category:",
            options=dimension_options
        )

        value_order_map = {}

        for dim in selected_dims:

            if dim not in df.columns:
                continue

            unique_vals = sorted(df[dim].dropna().astype(str).unique())

            user_input = st.multiselect(f"Urutan untuk {dim}:", unique_vals)

            default_vals = brand_default if dim == "Brand" else unique_vals

            value_order_map[dim] = merge_order(user_input, default_vals)

        # =========================
        # 🔥 INFO
        # =========================
        st.info("➡️ Urutan Sorting: " + " → ".join(["Subdept","Category"] + selected_dims))

        # =========================
        # 🔥 BUILD SORTING
        # =========================
        df_sorted = df.copy()

        df_sorted["Subdept_Order"] = df_sorted["Subdept"].map(
            {v:i for i,v in enumerate(subdept_order)}
        )

        df_sorted["Category_Order"] = df_sorted["Category"].map(
            {v:i for i,v in enumerate(category_order)}
        )

        sort_cols = ["Subdept_Order","Category_Order"]
        parent_cols = ["Subdept","Category"]

        for dim in selected_dims:

            if dim not in df.columns:
                continue

            rank_map = {v:i for i,v in enumerate(value_order_map[dim])}
            col_rank = f"{dim}_Order"

            df_sorted[col_rank] = (
                df_sorted.groupby(parent_cols)[dim]
                .transform(lambda x: x.map(rank_map))
            )

            sort_cols.append(col_rank)
            parent_cols.append(dim)

        df_sorted = df_sorted.sort_values(sort_cols).reset_index(drop=True)

        st.session_state["df_sorted"] = df_sorted

        # =========================
        # 🌳 HIERARCHY TREE
        # =========================
        st.subheader("PDT Preview")

        preview_cols = ["Subdept","Category"] + selected_dims
        preview_cols = [c for c in preview_cols if c in df_sorted.columns]

        df_preview = df_sorted[preview_cols].drop_duplicates()

        def render_tree(df, dims, level=0):
            if not dims or df.empty:
                return

            dim = dims[0]

            for val in df[dim].dropna().unique():

                df_next = df[df[dim] == val]

                indent = "&nbsp;" * (level * 6)
                st.markdown(f"{indent}↳ <b>{val}</b>", unsafe_allow_html=True)

                render_tree(df_next, dims[1:], level+1)

        for sub in subdept_order:

            with st.expander(f"🟥 {sub}", expanded=True):

                df_sub = df_sorted[df_sorted["Subdept"] == sub]

                for cat in category_order:

                    df_cat = df_sub[df_sub["Category"] == cat]

                    if df_cat.empty:
                        continue

                    st.markdown(f"<b>{cat}</b>", unsafe_allow_html=True)

                    render_tree(df_cat, selected_dims)

        # =========================
        # 📋 HIERARCHY TABLE
        # =========================
        st.subheader("Table PDT")

        order_cols = ["Subdept_Order","Category_Order"] + [
            f"{d}_Order" for d in selected_dims if f"{d}_Order" in df_sorted.columns
        ]

        df_table = df_sorted[preview_cols + order_cols].copy()
        df_table = df_table.drop_duplicates(subset=preview_cols)

        df_table["Hierarchy_Path"] = df_table[preview_cols].astype(str).agg(" > ".join, axis=1)

        df_table = df_table.sort_values(order_cols)

        st.dataframe(
            df_table[preview_cols + ["Hierarchy_Path"]],
            hide_index=True,
            use_container_width=True
        )
        
    # =========================
    # 🚀 BUTTON TRIGGER
    # =========================
    if st.button("🚀 Generate Planogram"):
        st.session_state["generate_clicked"] = True

    # =========================
    # 🚀 RUN ENGINE
    # =========================
    if st.session_state.get("generate_clicked", False):

        df_items = st.session_state.get("df_sorted", pd.DataFrame()).copy()
        df_tier = st.session_state.get("df_tier", pd.DataFrame())
        rack_config = st.session_state.get("rack_config", pd.DataFrame())
        section_selected = st.session_state.get("section", "")

        if df_items.empty or rack_config.empty:
            st.warning("Data belum lengkap")
            st.stop()

        # =========================
        # 🧠 SAFE DESC
        # =========================
        def get_desc(row):
            for col in ["DESCP", "Deskripsi", "PLU_DESC", "Description"]:
                if col in row and pd.notna(row[col]):
                    return str(row[col])
            return ""

        # =========================
        # 🧱 BUILD SHELF
        # =========================
        shelves = []

        for _, rack in rack_config.iterrows():
            rak_no = int(rack["Rak"].split("_")[1])
            jumlah_shelf = int(rack["Jumlah Shelving"])

            for s in range(1, jumlah_shelf + 1):

                max_height = 9999
                if section_selected == "Chiller" or s != 1:
                    max_height = rack["Max Tinggi per Shelving"][s-1]

                shelves.append({
                    "Rak": rak_no,
                    "Shelving": s,
                    "Max_Width": rack["Lebar Shelving (cm)"],
                    "Max_Height": max_height,
                    "Used_Width": 0
                })

        df_shelf = pd.DataFrame(shelves)

        # =========================
        # 🎯 TIER MAP
        # =========================
        tier_map = {}
        if not df_tier.empty:
            for _, r in df_tier.iterrows():
                tier_map[r["PLU"]] = r.get("Tier", 1)

        # =========================
        # 📦 PLACEMENT ENGINE FINAL (RAK-FIRST + TRUE SNAKE)
        # =========================

        placements = []
        unplaced_rows = []

        df_items = st.session_state["df_sorted"].copy()

        group_cols = ["Subdept", "Category"] + selected_dims
        group_cols = [c for c in group_cols if c in df_items.columns]

        df_items["GROUP_KEY"] = df_items[group_cols].astype(str).agg("||".join, axis=1)

        # 🔥 preserve order PDT
        group_order = df_items["GROUP_KEY"].drop_duplicates().tolist()

        # convert ke dict biar mutable
        group_dict = {
            g: df_items[df_items["GROUP_KEY"] == g].copy().reset_index(drop=True)
            for g in group_order
        }

        current_rak = 1
        direction_down = True
        group_idx = 0

        # =========================
        # 🔄 LOOP PER RAK (INI FIX UTAMA)
        # =========================
        while group_idx < len(group_order):

            df_rak = df_shelf[df_shelf["Rak"] == current_rak]

            # ❗ kalau rak sudah habis → semua jadi UNPLACED
            if df_rak.empty:
                for g in group_order[group_idx:]:
                    for _, row in group_dict[g].iterrows():

                        unplaced = {
                            "Rak": None,
                            "Shelving": None,
                            "No_Urut": None,
                            "Posisi": "A",
                            "Tier_Kiri_Kanan": tier_map.get(row["PLU"], 1),
                            "PLU": row["PLU"],
                            "Desc": get_desc(row),
                            "Category": row["Category"],
                            "LEBAR PCS": row.get("LEBAR PCS", 0),
                            "TINGGI PCS": row.get("TINGGI PCS", 0),
                            "Sisa_Lebar": None,
                            "Status": "UNPLACED"
                        }

                        for col in dimension_options:
                            unplaced[col] = row[col] if col in row else None

                        unplaced_rows.append(unplaced)
                break

            # =========================
            # 🐍 SNAKE
            # =========================
            if direction_down:
                shelf_list = df_rak.sort_values("Shelving").index.tolist()
            else:
                shelf_list = df_rak.sort_values("Shelving", ascending=False).index.tolist()

            # =========================
            # 🔄 ISI RAK SAMPAI PENUH
            # =========================
            while group_idx < len(group_order):

                g = group_order[group_idx]
                df_group = group_dict[g]

                if df_group.empty:
                    group_idx += 1
                    continue

                remaining_items = []
                placed_any = False

                for _, row in df_group.iterrows():

                    width = float(row.get("LEBAR PCS", 0))
                    height = float(row.get("TINGGI PCS", 0))
                    plu = row.get("PLU")

                    tier = tier_map.get(plu, 1)
                    width_needed = width * tier

                    placed = False

                    for idx in shelf_list:

                        shelf = df_shelf.loc[idx]

                        if height > shelf["Max_Height"]:
                            continue

                        remaining = shelf["Max_Width"] - shelf["Used_Width"]

                        if width_needed <= remaining:

                            df_shelf.loc[idx, "Used_Width"] += width_needed

                            placement_row = {
                                "Rak": shelf["Rak"],
                                "Shelving": shelf["Shelving"],
                                "No_Urut": None,
                                "Posisi": "A",
                                "Tier_Kiri_Kanan": tier,
                                "PLU": plu,
                                "Desc": get_desc(row),
                                "Category": row["Category"],
                                "LEBAR PCS": width,
                                "TINGGI PCS": height,
                                "Sisa_Lebar": round(
                                    shelf["Max_Width"] - df_shelf.loc[idx, "Used_Width"], 2
                                ),
                                "Status": "PLACED"
                            }

                            for col in dimension_options:
                                placement_row[col] = row[col] if col in row else None

                            placements.append(placement_row)

                            placed = True
                            placed_any = True
                            break

                    if not placed:
                        remaining_items.append(row)

                group_dict[g] = pd.DataFrame(remaining_items)

                # kalau group habis → lanjut group berikutnya
                if group_dict[g].empty:
                    group_idx += 1
                else:
                    # group belum habis → rak penuh → pindah rak
                    break

                # kalau tidak ada yang bisa di-place lagi → break rak
                if not placed_any:
                    break

            # =========================
            # 🔄 PINDAH RAK
            # =========================
            current_rak += 1
            direction_down = not direction_down


        # =========================
        # 📊 FINAL DATAFRAME
        # =========================
        df_placed = pd.DataFrame(placements)
        df_unplaced = pd.DataFrame(unplaced_rows)

        # =========================
        # 🔢 FIX NO URUT
        # =========================
        if not df_placed.empty:

            df_placed = df_placed.sort_values(
                by=["Rak", "Shelving"]
            ).reset_index(drop=True)

            df_placed["No_Urut"] = (
                df_placed.groupby(["Rak", "Shelving"])
                .cumcount() + 1
            )

        # =========================
        # 🧹 CLEAN UNPLACED
        # =========================
        if not df_unplaced.empty:
            df_unplaced = df_unplaced.dropna(axis=1, how="all")

        # =========================
        # 💾 SAVE
        # =========================
        st.session_state["df_placed"] = df_placed
        st.session_state["df_unplaced"] = df_unplaced


    # =========================
    # 📊 DISPLAY & DOWNLOAD
    # =========================
    df_placed = st.session_state.get("df_placed", pd.DataFrame())
    df_unplaced = st.session_state.get("df_unplaced", pd.DataFrame())

    if not df_placed.empty:

        st.subheader("📦 Planogram Result")
        st.dataframe(df_placed, hide_index=True, use_container_width=True)

        # download placed
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_placed.to_excel(writer, index=False)

        st.download_button(
            label="📥 Download Planogram Excel",
            data=output.getvalue(),
            file_name="planogram_result.xlsx"
        )

    if not df_unplaced.empty:

        st.warning(f"⚠️ {len(df_unplaced)} item tidak masuk planogram")
        st.dataframe(df_unplaced, hide_index=True, use_container_width=True)

        output_unplaced = BytesIO()
        with pd.ExcelWriter(output_unplaced, engine="xlsxwriter") as writer:
            df_unplaced.to_excel(writer, index=False)

        st.download_button(
            label="📥 Download Unplaced Items",
            data=output_unplaced.getvalue(),
            file_name="unplaced_items.xlsx"
        )
        
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