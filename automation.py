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
    st.header(" Setting Planogram")
    def load_affinity(section):
        try:
            file_path = f"affinity/{section}.xlsx"
            df_aff = pd.read_excel(file_path)

            # Pastikan kolom rapih
            df_aff.columns = df_aff.columns.str.strip()

            return df_aff

        except Exception as e:
            st.warning(f"File affinity untuk section '{section}' tidak ditemukan")
            return pd.DataFrame()

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
    # LOAD AFFINITY
    # =========================
    df_affinity = load_affinity(selected_section)

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

            # =========================
            # INPUT MAX HEIGHT PER SHELVING
            # =========================
            st.markdown("##### Setting Tinggi Maksimal per Shelving")

            # Default semua shelving = 30 cm
            max_height_shelf = [30] * jumlah_shelf

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
        # 🧩 ORDER PACKTYPE (MULTISELECT)
        # =========================
        st.subheader("🧩 Urutan Packtype")

        # ambil packtype unik
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

            df_category_perf["Rank"] = df_category_perf.index + 1
            
            # =========================
            # DETAIL PER CATEGORY
            # =========================
            df_category_detail = (
                df.groupby("Category")
                .agg(
                    Jumlah_PLU=("PLU", "nunique"),
                    Total_Lebar=("LEBAR PCS", "sum"),
                    Avg_Lebar=("LEBAR PCS", "mean"),
                    Max_Lebar=("LEBAR PCS", "max"),
                    Avg_Tinggi=("TINGGI PCS", "mean"),
                    Max_Tinggi=("TINGGI PCS", "max")
                )
                .reset_index()
            )
            
            df_final_category = df_category_perf.merge(
                df_category_detail,
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
                df_aff_top = (
                    df_aff_long.sort_values("Affinity", ascending=False)
                    .groupby("Category")
                    .first()
                    .reset_index()
                )

                df_aff_top = df_aff_top.rename(columns={
                    "Category_Pair": "Top_Affinity_Category",
                    "Affinity": "Top_Affinity_Score"
                })
            else:
                df_aff_top = pd.DataFrame()
                
            if not df_aff_top.empty:
                df_final_category = df_final_category.merge(
                    df_aff_top,
                    on="Category",
                    how="left"
                )
            
            st.subheader("Category Insight")
            st.dataframe(df_final_category)
            st.session_state["df_category_rank"] = df_final_category
            
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

                # ranking per category
                df_brand_insight = df_brand_insight.sort_values(
                    by=["Category", "Total_NS"],
                    ascending=[True, False]
                )

                df_brand_insight["Rank_in_Category"] = (
                    df_brand_insight.groupby("Category")
                    .cumcount() + 1
                )

                # optional: contribution %
                df_brand_insight["Contribution (%)"] = (
                    df_brand_insight["Total_NS"] /
                    df_brand_insight.groupby("Category")["Total_NS"].transform("sum")
                ) * 100

                st.dataframe(df_brand_insight)

                # save kalau mau dipakai nanti
                st.session_state["df_brand_insight"] = df_brand_insight

            else:
                st.warning("Kolom 'YTD ∑NS 2026 Brand' tidak ditemukan")
            
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