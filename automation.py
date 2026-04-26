import streamlit as st
import pandas as pd
from io import BytesIO
# from utils.ui_styles import render_fixed_header

st.set_page_config(page_title="Planogram Automation", layout="wide")
st.title("📊 Planogram Automation")

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

                # =========================
                # DEFAULT MAX HEIGHT
                # =========================
                selected_section = st.session_state.get("section", "")
                max_height_shelf = [30] * jumlah_shelf
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
            st.dataframe(df_rack_config)
            
            # =========================
            # SETTING TIER PLU
            # =========================
            st.subheader("Setting Tier (Kiri-Kanan / Facing PLU)")

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
            st.dataframe(df_tier_valid)

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
        st.subheader("Category Insight")
        st.dataframe(df_final_category)
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
        col1, col2, col3, col4 = st.columns(4)

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
                df_growth[["Rank","Category","YTD %GROWTH NS Category"]]
                .style.applymap(color_growth, subset=["YTD %GROWTH NS Category"])
            )
        
        # =========================
        # 🏷️ BRAND INSIGHT PER CATEGORY
        # =========================
        st.subheader("Brand Insight")

        df = st.session_state["df_result"].copy()

        # ambil data brand performance
        if "YTD ∑NS 2026 Brand" in df.columns:

            df_brand_insight = (
                df.groupby(["Category", "Brand"])
                .agg(
                    Jumlah_PLU=("PLU", "nunique"),
                    Total_NS=("YTD ∑NS 2026 Brand", "first")  # tidak di-sum
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
            st.dataframe(df_brand_insight)
            st.session_state["df_brand_insight"] = df_brand_insight
        else:
            st.warning("Kolom 'YTD ∑NS 2026 Brand' tidak ditemukan")
            
        
        st.header(" Setting Planogram")
        # =========================
        # 🧠 SORTING PRIORITY ENGINE (USER INPUT)
        # =========================
        st.subheader("Urutan PDT")
        
        import pandas as pd

        df = st.session_state.get("df_result", pd.DataFrame())

        if df.empty:
            st.warning("Silakan upload data terlebih dahulu")
        else:

            # =========================
            # 🎛️ URUTAN DIMENSI
            # =========================
            dimension_options = [
                "Packtype",
                "Brand",
                "User",
                "Subcategory",
                "Packsize",
                "Price Segment",
                "Variant/Flavor"
            ]

            selected_dims = st.multiselect(
                "Pilih urutan penyusunan (setelah Category)",
                options=dimension_options
            )

            if not selected_dims:
                selected_dims = ["Packtype", "Brand"]

            st.info("Urutan: Subdept → Category → " + " → ".join(selected_dims))

            # =========================
            # 🔬 URUTAN NILAI PER DIMENSI
            # =========================
            st.markdown("### 🔬 Urutan Nilai per Dimensi")

            value_order_map = {}

            for dim in selected_dims:
                if dim in df.columns:

                    unique_vals = df[dim].dropna().unique().tolist()

                    selected_order = st.multiselect(
                        f"Urutan {dim}",
                        options=unique_vals,
                        key=f"order_{dim}"
                    )

                    remaining = [v for v in unique_vals if v not in selected_order]
                    final_order = selected_order + remaining

                    value_order_map[dim] = final_order

            # simpan
            st.session_state["dimension_order"] = selected_dims
            st.session_state["value_order_map"] = value_order_map

            # =========================
            # 🏆 SUBDEPT CONTRIBUTION SORT
            # =========================
            if "YTD ∑NS 2026 Category" in df.columns:

                df_subdept_rank = (
                    df.groupby("Subdept")["YTD ∑NS 2026 Category"]
                    .sum()
                    .reset_index()
                    .sort_values(by="YTD ∑NS 2026 Category", ascending=False)
                )

                subdept_order = df_subdept_rank["Subdept"].tolist()
            else:
                subdept_order = df["Subdept"].dropna().unique().tolist()

            # =========================
            # 🔥 BUILD SORTING
            # =========================
            df_sorted = df.copy()

            # mapping subdept rank
            subdept_map = {v: i for i, v in enumerate(subdept_order)}
            df_sorted["Subdept_Order"] = df_sorted["Subdept"].map(subdept_map)

            # wajib
            sort_cols = ["Subdept_Order", "Category"]
            sort_ascending = [True, True]

            # dynamic
            for dim in selected_dims:
                if dim in df_sorted.columns:

                    order_list = value_order_map.get(dim, [])
                    rank_map = {v: i for i, v in enumerate(order_list)}

                    col_rank = f"{dim}_Order"
                    df_sorted[col_rank] = df_sorted[dim].map(rank_map).fillna(999)

                    sort_cols.append(col_rank)
                    sort_ascending.append(True)

            df_sorted = df_sorted.sort_values(
                by=sort_cols,
                ascending=sort_ascending
            ).reset_index(drop=True)

            # =========================
            # 📊 PREVIEW HIERARKI (INI YANG BARU 🔥)
            # =========================
            st.markdown("### 📊 Struktur Hirarki Sorting")

            hierarchy_data = []

            for sub in subdept_order:
                hierarchy_data.append({"Level": "Subdept", "Value": sub})

                df_sub = df_sorted[df_sorted["Subdept"] == sub]

                for cat in df_sub["Category"].unique():
                    hierarchy_data.append({"Level": "  ↳ Category", "Value": cat})

                    df_cat = df_sub[df_sub["Category"] == cat]

                    for dim in selected_dims:
                        if dim in df_cat.columns:
                            vals = df_cat[dim].dropna().unique().tolist()
                            vals_ordered = value_order_map.get(dim, vals)

                            for v in vals_ordered:
                                if v in vals:
                                    hierarchy_data.append({
                                        "Level": f"    ↳ {dim}",
                                        "Value": v
                                    })

            df_hierarchy = pd.DataFrame(hierarchy_data)

            st.dataframe(df_hierarchy)

            # simpan
            st.session_state["df_sorted"] = df_sorted
        
        # =========================
        # 🧩 ORDER PACKTYPE (MULTISELECT)
        # =========================
        st.subheader("🧩 Urutan Packtype")
        packtype_options = (
            st.session_state["df_result"]["Packtype"]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        st.caption(
            "Pilih urutan packtype (opsional). "
            "Urutan mengikuti pilihan Anda. "
            "Yang tidak dipilih akan otomatis diurutkan di belakang."
        )

        # multiselect (order mengikuti klik)
        selected_packtype = st.multiselect(
            "Pilih Packtype Prioritas",
            options=packtype_options
        )

        # =========================
        # BUILD FINAL ORDER
        # =========================
        if selected_packtype:
            # urutan: pilihan user dulu, lalu sisanya
            remaining = [p for p in packtype_options if p not in selected_packtype]
            final_order = selected_packtype + remaining
        else:
            # default: urutan asli
            final_order = packtype_options

        # =========================
        # MAPPING PRIORITY
        # =========================
        packtype_priority_map = {
            p: i + 1 for i, p in enumerate(final_order)
        }

        # save ke session
        st.session_state["packtype_priority_map"] = packtype_priority_map

        # preview (opsional)
        df_packtype_order = pd.DataFrame({
            "Packtype": final_order,
            "Priority": range(1, len(final_order) + 1)
        })

        st.subheader("Preview Urutan Packtype")
        st.dataframe(df_packtype_order)

        
        if st.button("🚀 Generate Planogram"):    
            # =========================
            # 🚀 PLANOGRAM ENGINE (CATEGORY CHAIN + PACKTYPE ORDER)
            # =========================

            df = st.session_state["df_result"].copy()
            df_category = st.session_state["df_category_rank"].copy()
            df_rack = st.session_state["rack_config"].copy()
            df_tier = st.session_state.get("df_tier", pd.DataFrame())
            df_aff = st.session_state.get("df_affinity", pd.DataFrame())
            packtype_priority_map = st.session_state.get("packtype_priority_map", {})

            # =========================
            # SAFETY CHECK
            # =========================
            if df.empty:
                st.error("Data hasil upload masih kosong.")
                st.stop()

            if df_category.empty:
                st.error("Category Insight belum tersedia.")
                st.stop()

            if df_rack.empty:
                st.error("Konfigurasi rak belum tersedia.")
                st.stop()

            # =========================
            # TIER LOOKUP
            # =========================
            if not df_tier.empty and {"PLU", "Tier"}.issubset(df_tier.columns):
                tier_lookup = dict(zip(df_tier["PLU"].astype(str), df_tier["Tier"]))
            else:
                tier_lookup = {}

            # =========================
            # BRAND RANKING (kalau belum ada)
            # =========================
            if "Brand_Rank" not in df.columns:
                if "YTD ∑NS 2026 Brand" in df.columns:
                    df_brand_rank = (
                        df[["Brand", "YTD ∑NS 2026 Brand"]]
                        .drop_duplicates()
                        .sort_values(by="YTD ∑NS 2026 Brand", ascending=False)
                        .reset_index(drop=True)
                    )
                    df_brand_rank["Brand_Rank"] = df_brand_rank.index + 1
                    df = df.merge(
                        df_brand_rank[["Brand", "Brand_Rank"]],
                        on="Brand",
                        how="left"
                    )
                else:
                    df["Brand_Rank"] = 999999

            # =========================
            # PACKTYPE PRIORITY COLUMN
            # =========================
            if "Packtype" in df.columns:
                df["Packtype"] = df["Packtype"].fillna("UNKNOWN").astype(str).str.strip()
                df["Packtype_Priority"] = df["Packtype"].map(packtype_priority_map)
                df["Packtype_Priority"] = df["Packtype_Priority"].fillna(
                    len(packtype_priority_map) + 999
                ).astype(int)
            else:
                df["Packtype_Priority"] = 999999

            # =========================
            # PREPARE AFFINITY MAP
            # =========================
            aff_map = {}
            if not df_aff.empty and "Category" in df_aff.columns:
                df_aff_long = df_aff.melt(
                    id_vars="Category",
                    var_name="Category_Pair",
                    value_name="Affinity"
                )

                df_aff_long = df_aff_long[
                    df_aff_long["Category"].astype(str) != df_aff_long["Category_Pair"].astype(str)
                ]

                df_aff_top = (
                    df_aff_long.sort_values(["Category", "Affinity"], ascending=[True, False])
                    .drop_duplicates(subset=["Category"], keep="first")
                    .rename(columns={
                        "Category_Pair": "Top_Affinity_Category",
                        "Affinity": "Top_Affinity_Score"
                    })
                )

                df_category = df_category.merge(df_aff_top, on="Category", how="left")
                aff_map = dict(
                    zip(
                        df_category["Category"].astype(str),
                        df_category.get("Top_Affinity_Category", pd.Series([None] * len(df_category))).astype(str)
                    )
                )

            # =========================
            # EXPAND SHELVING
            # =========================
            shelf_list = []

            rack_order = df_rack["Rak"].tolist()

            for rack_idx, (_, row) in enumerate(df_rack.iterrows(), start=1):
                rak = row["Rak"]
                lebar = row["Lebar Shelving (cm)"]
                max_heights = row["Max Tinggi per Shelving"]

                for shelf_no, h in enumerate(max_heights, start=1):
                    shelf_list.append({
                        "Rack_Order": rack_idx,
                        "Rak": rack_idx,  # angka
                        "Shelving_No": shelf_no,
                        "Shelving": shelf_no,  # angka
                        "Max_Height": h,
                        "Max_Width": lebar,
                        "Used_Width": 0,
                        "Item_Count": 0
                    })

            df_shelf = pd.DataFrame(shelf_list)
            
            # =========================
            # FIX TYPE NUMERIC
            # =========================
            df_shelf["Used_Width"] = df_shelf["Used_Width"].astype(float)
            df_shelf["Max_Width"] = df_shelf["Max_Width"].astype(float)
            df_shelf["Max_Height"] = df_shelf["Max_Height"].astype(float)
            df_shelf["Item_Count"] = df_shelf["Item_Count"].astype(int)

            if df_shelf.empty:
                st.error("Shelving belum terbentuk dengan benar.")
                st.stop()

            df_shelf = df_shelf.sort_values(
                by=["Rack_Order", "Shelving_No"],
                ascending=[True, True]
            ).reset_index(drop=True)

            # =========================
            # CATEGORY SEED ORDER
            # =========================
            df_category = df_category.drop_duplicates(subset=["Category"]).copy()
            df_category = df_category.sort_values(by="Rank", ascending=True).reset_index(drop=True)

            ranked_categories = df_category["Category"].astype(str).tolist()
            placed_categories = set()

            # bikin chain kategori berdasarkan rank + affinity
            category_chain = []
            for seed in ranked_categories:
                if seed in category_chain:
                    continue

                current = seed
                local_seen = set()

                while current and current not in local_seen and current not in category_chain:
                    category_chain.append(current)
                    local_seen.add(current)

                    next_cat = aff_map.get(current, None)
                    if pd.isna(next_cat):
                        break

                    next_cat = str(next_cat).strip()
                    if not next_cat or next_cat.lower() == "nan":
                        break
                    if next_cat in local_seen:
                        break

                    current = next_cat

            # tambah category yang belum kebagian chain
            for cat in ranked_categories:
                if cat not in category_chain:
                    category_chain.append(cat)

            # =========================
            # SORT ITEM DALAM CATEGORY
            # =========================
            sort_cols = ["Category", "Packtype_Priority", "Brand_Rank", "NS/JTD"]
            sort_ascending = [True, True, True, False]

            available_sort_cols = [c for c in sort_cols if c in df.columns]
            df = df.sort_values(
                by=available_sort_cols,
                ascending=[sort_ascending[sort_cols.index(c)] for c in available_sort_cols]
            ).reset_index(drop=True)

            # =========================
            # PLACEMENT ENGINE
            # =========================
            planogram = []
            not_display = []
            placed_plu = set()

            for category in category_chain:
                df_cat_items = df[df["Category"].astype(str) == str(category)].copy()

                if df_cat_items.empty:
                    continue

                df_cat_items = df_cat_items.sort_values(
                    by=[c for c in ["Packtype_Priority", "Brand_Rank", "NS/JTD"] if c in df_cat_items.columns],
                    ascending=[True, True, False][:len([c for c in ["Packtype_Priority", "Brand_Rank", "NS/JTD"] if c in df_cat_items.columns])]
                )

                for _, item in df_cat_items.iterrows():
                    plu = str(item["PLU"])
                    if plu in placed_plu:
                        continue

                    tier = tier_lookup.get(plu, 1)
                    try:
                        tier = int(tier) if pd.notna(tier) else 1
                    except Exception:
                        tier = 1
                    if tier < 1:
                        tier = 1

                    width_needed = float(item["LEBAR PCS"]) * float(tier)
                    height_needed = float(item["TINGGI PCS"])

                    placed = False

                    for idx in df_shelf.index:
                        shelf = df_shelf.loc[idx]

                        if height_needed > float(shelf["Max_Height"]):
                            continue

                        if float(shelf["Used_Width"]) + width_needed <= float(shelf["Max_Width"]):
                            no_urut = int(df_shelf.loc[idx, "Item_Count"]) + 1

                            df_shelf.loc[idx, "Used_Width"] = float(shelf["Used_Width"]) + width_needed
                            df_shelf.loc[idx, "Item_Count"] = no_urut

                            planogram.append({
                                "Rak": shelf["Rak"],
                                "Shelving": shelf["Shelving"],
                                "No_Urut": no_urut,
                                "Posisi": "A",
                                "Tier_Kiri_Kanan": tier,
                                "PLU": plu,
                                "Desc": item.get("DESCP", ""),
                                "Category": item["Category"]
                            })

                            placed_plu.add(plu)
                            placed = True
                            break

                    if not placed:
                        not_display.append({
                            "PLU": plu,
                            "Category": item["Category"],
                            "Tier": tier,
                            "Reason": "Tidak cukup space / tinggi tidak sesuai"
                        })

            # =========================
            # FINAL OUTPUT
            # =========================
            df_planogram = pd.DataFrame(planogram)
            df_not_display = pd.DataFrame(not_display)

            if not df_planogram.empty:
                df_planogram = df_planogram[
                    ["Rak", "Shelving", "No_Urut", "Posisi", "Tier_Kiri_Kanan", "PLU", "Desc", "Category"]
                ].sort_values(
                    by=["Rak", "Shelving", "No_Urut"]
                ).reset_index(drop=True)

            st.subheader("📦 Planogram")
            st.dataframe(df_planogram)

            st.subheader("⚠️ Summary Item Tidak Masuk")
            if not df_not_display.empty:
                summary = (
                    df_not_display.groupby(["Category", "Reason"])
                    .size()
                    .reset_index(name="Jumlah")
                    .sort_values(by="Jumlah", ascending=False)
                )
                st.dataframe(summary)

            st.subheader("📋 Detail Item Tidak Masuk")
            st.dataframe(df_not_display)

            st.session_state["planogram"] = df_planogram
            st.session_state["not_display"] = df_not_display

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