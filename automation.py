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
        # SAVE KE SESSION
        # =========================
        df_tier = pd.DataFrame(tier_data)

        # hanya ambil yang valid
        df_tier_valid = df_tier[df_tier["Valid"] == True]

        st.session_state["df_tier"] = df_tier_valid

        # =========================
        # PREVIEW
        # =========================
        st.subheader("Preview Tier Setting")
        st.dataframe(df_tier_valid)

        
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
            # 🚀 PLANOGRAM ENGINE FINAL (STABLE VERSION)
            # =========================

            df = st.session_state["df_result"].copy()
            df_category = st.session_state["df_category_rank"].copy()
            df_rack = st.session_state["rack_config"].copy()
            df_tier = st.session_state.get("df_tier", pd.DataFrame())
            df_aff = st.session_state.get("df_affinity", pd.DataFrame())

            # =========================
            # 🎯 TIER LOOKUP
            # =========================
            if not df_tier.empty:
                tier_lookup = dict(zip(df_tier["PLU"], df_tier["Tier"]))
            else:
                tier_lookup = {}

            # =========================
            # 🧱 EXPAND SHELVING
            # =========================
            shelf_list = []

            for _, row in df_rack.iterrows():
                rak = row["Rak"]
                lebar = row["Lebar Shelving (cm)"]
                max_heights = row["Max Tinggi per Shelving"]

                for i, h in enumerate(max_heights):
                    shelf_list.append({
                        "Rak": rak,
                        "Shelving": f"{rak}_S{i+1}",
                        "Max_Height": h,
                        "Max_Width": lebar,
                        "Used_Width": 0
                    })

            df_shelf = pd.DataFrame(shelf_list)

            # =========================
            # 🥇 ASSIGN CATEGORY → RAK
            # =========================
            unique_rak = df_shelf["Rak"].unique().tolist()

            df_category = df_category.reset_index(drop=True)
            df_category["Assigned_Rak"] = None

            for i in range(len(df_category)):
                df_category.loc[i, "Assigned_Rak"] = unique_rak[i % len(unique_rak)]

            # =========================
            # 📦 SORT ITEM
            # =========================
            df = df.sort_values(
                by=["Category", "Packtype", "Brand", "NS/JTD"],
                ascending=[True, True, True, False]
            ).reset_index(drop=True)

            # =========================
            # 🚀 PLACEMENT ENGINE
            # =========================
            planogram = []
            not_display = []
            placed_plu = set()

            for _, cat_row in df_category.iterrows():
                category = cat_row["Category"]
                assigned_rak = cat_row["Assigned_Rak"]

                df_cat_items = df[df["Category"] == category]

                shelves_idx = df_shelf[df_shelf["Rak"] == assigned_rak].index.tolist()

                for _, item in df_cat_items.iterrows():

                    if item["PLU"] in placed_plu:
                        continue

                    tier = tier_lookup.get(item["PLU"], 1)
                    width_needed = item["LEBAR PCS"] * tier

                    placed = False

                    for idx in shelves_idx:
                        shelf = df_shelf.loc[idx]

                        # constraint tinggi
                        if item["TINGGI PCS"] > shelf["Max_Height"]:
                            continue

                        # constraint lebar
                        if shelf["Used_Width"] + width_needed <= shelf["Max_Width"]:

                            df_shelf.loc[idx, "Used_Width"] += width_needed

                            planogram.append({
                                "PLU": item["PLU"],
                                "Category": category,
                                "Rak": shelf["Rak"],
                                "Shelving": shelf["Shelving"],
                                "Tier": tier
                            })

                            placed_plu.add(item["PLU"])
                            placed = True
                            break

                    if not placed:
                        if item["TINGGI PCS"] > df_shelf.loc[shelves_idx, "Max_Height"].max():
                            reason = "Tinggi melebihi semua shelf"
                        elif all(
                            df_shelf.loc[shelves_idx, "Used_Width"] + width_needed > df_shelf.loc[shelves_idx, "Max_Width"]
                        ):
                            reason = "Lebar tidak cukup"
                        else:
                            reason = "Tidak teralokasi"

                        not_display.append({
                            "PLU": item["PLU"],
                            "Category": category,
                            "Tier": tier,
                            "Reason": reason
                        })

            # =========================
            # 🧩 FILL SISA SPACE
            # =========================
            remaining_items = df[~df["PLU"].isin(placed_plu)]

            for _, item in remaining_items.iterrows():

                tier = tier_lookup.get(item["PLU"], 1)
                width_needed = item["LEBAR PCS"] * tier

                placed = False

                for idx in df_shelf.index:
                    shelf = df_shelf.loc[idx]

                    if item["TINGGI PCS"] <= shelf["Max_Height"] and \
                    shelf["Used_Width"] + width_needed <= shelf["Max_Width"]:

                        df_shelf.loc[idx, "Used_Width"] += width_needed

                        planogram.append({
                            "PLU": item["PLU"],
                            "Category": item["Category"],
                            "Rak": shelf["Rak"],
                            "Shelving": shelf["Shelving"],
                            "Tier": tier
                        })

                        placed_plu.add(item["PLU"])
                        placed = True
                        break

                if not placed:
                    not_display.append({
                        "PLU": item["PLU"],
                        "Category": item["Category"],
                        "Tier": tier,
                        "Reason": "Tidak cukup space"
                    })

            # =========================
            # 📊 BUILD FINAL OUTPUT
            # =========================
            df_planogram = pd.DataFrame(planogram)
            df_not_display = pd.DataFrame(not_display)

            if not df_planogram.empty:

                df_planogram = df_planogram.merge(
                    df[["PLU", "DESCP", "Category"]],
                    on=["PLU", "Category"],
                    how="left"
                )

                df_planogram = df_planogram.rename(columns={
                    "DESCP": "Desc",
                    "Tier": "Tier_Kiri_Kanan"
                })

                df_planogram["Posisi"] = "A"

                df_planogram = df_planogram.sort_values(
                    by=["Rak", "Shelving"]
                ).reset_index(drop=True)

                df_planogram["No_Urut"] = (
                    df_planogram.groupby(["Rak", "Shelving"])
                    .cumcount() + 1
                )

                df_planogram = df_planogram[
                    ["Rak","Shelving","No_Urut","Posisi","Tier_Kiri_Kanan","PLU","Desc","Category"]
                ]

            # =========================
            # 📊 OUTPUT
            # =========================
            st.subheader("📦 Planogram")
            st.dataframe(df_planogram)

            st.subheader("⚠️ Summary Item Tidak Masuk")
            if not df_not_display.empty:
                summary = (
                    df_not_display.groupby(["Category","Reason"])
                    .size()
                    .reset_index(name="Jumlah")
                    .sort_values(by="Jumlah", ascending=False)
                )
                st.dataframe(summary)

            st.subheader("📋 Detail Item Tidak Masuk")
            st.dataframe(df_not_display)

            # =========================
            # 💾 SAVE
            # =========================
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