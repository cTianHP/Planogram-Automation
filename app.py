# app.py
import streamlit as st
import pandas as pd
from io import BytesIO

# ============================
# CONFIG
# ============================
SHELVING_COUNT = 7
SHELF_WIDTH = 51.5
MAX_FACING_PER_SKU = 2

YOGHURT_SUBCATEGORY = "6212A - YOGHURT PACK"

FLAVOUR_CATEGORIES = [
    "6227 - CHILLED SAUSAGES",
    "6226 - CHILLED MEAT BALLS"
]

FLAVOUR_ORDER = {"ORIGINAL": 1, "CHEESE": 2, "SPICY": 3}


# ============================
# TEMPLATE (downloadable)
# ============================
def generate_template():
    cols = [
        "PLU", "NAME", "CATEGORY", "SUBCATEGORY", "Brand",
        "Variant/Flavor", "Width",
        "Avg NS", "SALES CATEGORY", "SALES SUBCATEGORY", "YTD ∑NS 2025"
    ]
    df = pd.DataFrame(columns=cols)
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="MASTER_ITEM")
    return buf.getvalue()


# ============================
# SORTING BY HIERARCHY (no scoring for placement)
# ============================
def sort_by_hierarchy(df):
    df = df.copy()

    # Ensure necessary columns exist
    for c in ["CATEGORY", "SUBCATEGORY", "Brand", "SALES CATEGORY", "SALES SUBCATEGORY", "YTD ∑NS 2025", "Avg NS", "Width", "PLU", "Variant/Flavor", "NAME"]:
        if c not in df.columns:
            df[c] = 0 if c in ["SALES CATEGORY", "SALES SUBCATEGORY", "YTD ∑NS 2025", "Avg NS", "Width"] else ""

    # split yoghurt (hard rule)
    df_yoghurt = df[df["SUBCATEGORY"] == YOGHURT_SUBCATEGORY].copy()
    df_normal = df[df["SUBCATEGORY"] != YOGHURT_SUBCATEGORY].copy()

    # build ranks for normal group
    if not df_normal.empty:
        cat_order = df_normal.groupby("CATEGORY")["SALES CATEGORY"].sum().sort_values(ascending=False).index.tolist()
        subcat_order = df_normal.groupby("SUBCATEGORY")["SALES SUBCATEGORY"].sum().sort_values(ascending=False).index.tolist()
        brand_order = df_normal.groupby("Brand")["YTD ∑NS 2025"].sum().sort_values(ascending=False).index.tolist()

        df_normal["CAT_RANK"] = df_normal["CATEGORY"].apply(lambda x: cat_order.index(x) if x in cat_order else len(cat_order))
        df_normal["SUBCAT_RANK"] = df_normal["SUBCATEGORY"].apply(lambda x: subcat_order.index(x) if x in subcat_order else len(subcat_order))
        df_normal["BRAND_RANK"] = df_normal["Brand"].apply(lambda x: brand_order.index(x) if x in brand_order else len(brand_order))
    else:
        df_normal["CAT_RANK"] = 0
        df_normal["SUBCAT_RANK"] = 0
        df_normal["BRAND_RANK"] = 0

    # flavour rank only for flavour categories
    df_normal["FLAVOUR_RANK"] = df_normal.apply(
        lambda r: FLAVOUR_ORDER.get(str(r.get("Variant/Flavor", "")).upper(), 99)
        if r.get("CATEGORY") in FLAVOUR_CATEGORIES else 99,
        axis=1
    )

    # for categories without flavour rule, use Avg NS ascending (small -> big) so hero SKU goes last
    df_normal["AVG_NS_RANK"] = df_normal.apply(
        lambda r: r.get("Avg NS", 0) if r.get("CATEGORY") not in FLAVOUR_CATEGORIES else 0,
        axis=1
    )

    # final sort for normal
    df_normal = df_normal.sort_values(
        by=["CAT_RANK", "SUBCAT_RANK", "BRAND_RANK", "FLAVOUR_RANK", "AVG_NS_RANK", "PLU"],
        ascending=[True, True, True, True, True, True]
    ).reset_index(drop=True)

    # yoghurt deterministic order (small to big Avg NS), will be placed only on last shelf
    df_yoghurt["Avg NS"] = pd.to_numeric(df_yoghurt.get("Avg NS", 0), errors="coerce").fillna(0)
    df_yoghurt = df_yoghurt.sort_values("Avg NS").reset_index(drop=True)

    return df_normal, df_yoghurt


# ============================
# PLACE & GLOBAL FACING OPTIMIZATION (updated)
# ============================
def place_and_optimize(df_normal, df_yoghurt, shelving_count=SHELVING_COUNT, shelf_width=SHELF_WIDTH, max_facing=MAX_FACING_PER_SKU):
    # ensure numeric fields
    df_normal["Width"] = pd.to_numeric(df_normal["Width"], errors="coerce").fillna(0)
    df_normal["Avg NS"] = pd.to_numeric(df_normal["Avg NS"], errors="coerce").fillna(0)
    df_yoghurt["Width"] = pd.to_numeric(df_yoghurt["Width"], errors="coerce").fillna(0)
    df_yoghurt["Avg NS"] = pd.to_numeric(df_yoghurt["Avg NS"], errors="coerce").fillna(0)

    # placements per shelf: shelf_id -> list of {"row": row, "facing": int}
    shelves = {}
    current_shelf = 1

    # 1) place normal items across shelves (assortment-first)
    for _, r in df_normal.iterrows():
        w = float(r["Width"])
        if w <= 0:
            continue
        placed = False
        while not placed:
            if current_shelf > shelving_count:
                # no more shelves -> leave unplaced
                placed = True
                break
            shelves.setdefault(current_shelf, [])
            used = sum(float(it["row"]["Width"]) * it["facing"] for it in shelves[current_shelf])
            if used + w <= shelf_width + 1e-9:
                shelves[current_shelf].append({"row": r, "facing": 1})
                placed = True
            else:
                current_shelf += 1

    # 2) place yoghurt on last shelf only
    last_shelf = shelving_count
    shelves.setdefault(last_shelf, [])
    for _, r in df_yoghurt.iterrows():
        w = float(r["Width"])
        if w <= 0:
            continue
        used = sum(float(it["row"]["Width"]) * it["facing"] for it in shelves[last_shelf])
        if used + w <= shelf_width + 1e-9:
            shelves[last_shelf].append({"row": r, "facing": 1})
        else:
            # can't place more yoghurt -> leave unassigned
            pass

    # 3) collect unassigned items
    placed_plus = {it["row"]["PLU"] for s in shelves.values() for it in s}
    unassigned = []
    for _, r in pd.concat([df_normal, df_yoghurt]).iterrows():
        if r["PLU"] not in placed_plus:
            unassigned.append(r)

    # 4) prepare per-PLU facing counts
    facing_by_plu = {}
    for s_id, items in shelves.items():
        for it in items:
            plu = it["row"]["PLU"]
            facing_by_plu[plu] = facing_by_plu.get(plu, 0) + it["facing"]

    # 5) compute remaining_space per shelf
    remaining_space = {}
    for s_id, items in shelves.items():
        used = sum(float(it["row"]["Width"]) * it["facing"] for it in items)
        remaining_space[s_id] = round(shelf_width - used, 9)

    # 6) GLOBAL GREEDY: try to allocate additional facings anywhere (including empty shelves)
    # Build SKU list (non-yoghurt) that are eligible (exist in master and placed at least once)
    all_rows = pd.concat([df_normal, df_yoghurt], ignore_index=True)
    plu_to_row = {r["PLU"]: r for _, r in all_rows.iterrows()}

    while True:
        # collect candidates (plu -> best shelf for placing one more facing)
        candidates = []  # list of tuples (plu, shelf_id, width, avg_ns, existing_idx_or_None)
        for plu, row in plu_to_row.items():
            if row.get("SUBCATEGORY") == YOGHURT_SUBCATEGORY:
                continue  # yoghurt excluded
            current_total_facing = facing_by_plu.get(plu, 0)
            if current_total_facing >= max_facing:
                continue  # can't add more
            w = float(row.get("Width", 0) or 0)
            if w <= 0:
                continue

            # Option A: prefer shelves where SKU already exists and has room to increment that placement
            found = False
            for s_id, items in shelves.items():
                for idx, it in enumerate(items):
                    if it["row"]["PLU"] == plu:
                        # can we add to this placement?
                        if remaining_space.get(s_id, 0) + 1e-9 >= w:
                            candidates.append((plu, s_id, w, float(row.get("Avg NS", 0) or 0), idx))
                            found = True
                        break
                if found:
                    break
            if found:
                continue

            # Option B: try to place a new placement (duplicate) of this SKU into a shelf that has enough room
            # prefer shelf with largest remaining_space that can fit the item
            possible_shelves = [(s_id, rem) for s_id, rem in remaining_space.items() if rem + 1e-9 >= w]
            if possible_shelves:
                # choose shelf with largest remaining space
                s_choice = max(possible_shelves, key=lambda x: x[1])[0]
                candidates.append((plu, s_choice, w, float(row.get("Avg NS", 0) or 0), None))

        if not candidates:
            break

        # choose candidate with highest Avg NS
        best = max(candidates, key=lambda x: x[3])
        chosen_plu, chosen_shelf, chosen_w, _, existing_idx = best

        # apply placement: either increment existing placement facing or create new placement
        if existing_idx is not None:
            # increment facing in that placement
            shelves[chosen_shelf][existing_idx]["facing"] += 1
        else:
            # create new placement in chosen_shelf with facing = 1
            row = plu_to_row[chosen_plu]
            shelves[chosen_shelf].append({"row": row, "facing": 1})

        # update totals
        facing_by_plu[chosen_plu] = facing_by_plu.get(chosen_plu, 0) + 1
        remaining_space[chosen_shelf] = round(remaining_space[chosen_shelf] - chosen_w, 9)

    # 7) recompute final Start/End positions and build output rows
    output_rows = []
    max_shelf_used = max(shelves.keys()) if shelves else 0
    for shelf_id in range(1, max_shelf_used + 1):
        items = shelves.get(shelf_id, [])
        if not items:
            continue

        # final positioning: odd shelf fill left->right; even shelf fill right->left then convert to left->right
        if shelf_id % 2 == 1:
            cursor = 0.0
            arranged = []
            for it in items:
                w = float(it["row"]["Width"])
                total_w = w * it["facing"]
                start = cursor
                end = cursor + total_w
                arranged.append((it, start, end))
                cursor = end
        else:
            cursor = shelf_width
            arranged_rev = []
            for it in items:
                w = float(it["row"]["Width"])
                total_w = w * it["facing"]
                end = cursor
                start = cursor - total_w
                arranged_rev.append((it, start, end))
                cursor = start
            arranged = sorted(arranged_rev, key=lambda x: x[1])

        used_final = sum((p[2] - p[1]) for p in arranged)
        remaining_space_final = round(shelf_width - used_final, 6)

        for pos, (it, start, end) in enumerate(arranged, start=1):
            r = it["row"]
            output_rows.append({
                "Shelving": shelf_id,
                "No Urut": pos,
                "PLU": r.get("PLU"),
                "NAME": r.get("NAME"),
                "CATEGORY": r.get("CATEGORY"),
                "SUBCATEGORY": r.get("SUBCATEGORY"),
                "Brand": r.get("Brand"),
                "Variant/Flavor": r.get("Variant/Flavor"),
                "Facing": it["facing"],
                "Width per Item": float(r.get("Width", 0) or 0),
                "Total Width": round((float(r.get("Width", 0) or 0) * it["facing"]), 6),
                "Start_cm": round(start, 6),
                "End_cm": round(end, 6),
                "Sisa Space Shelving": remaining_space_final
            })

    planogram_df = pd.DataFrame(output_rows)
    unassigned_df = pd.DataFrame([{"PLU": r.get("PLU"), "NAME": r.get("NAME"), "REASON": "no shelf space / invalid width"} for r in unassigned])

    return planogram_df, unassigned_df


# ============================
# STREAMLIT UI
# ============================
st.set_page_config(page_title="Auto Planogram Engine (global-filling)", layout="wide")
st.title("🛒 Auto Planogram Engine (Global Facing fill)")
st.caption("Assortment-first • Yoghurt hard-rule • Snake layout • Global greedy facing fill (Max facing per SKU)")

st.subheader("1️⃣ Download Template")
st.download_button("⬇️ Download Template Master Item", generate_template(), "template_master_item.xlsx")

st.subheader("2️⃣ Upload Master Item (.xlsx)")
uploaded = st.file_uploader("Upload file", type=["xlsx"])
if uploaded is not None:
    df = pd.read_excel(uploaded)
    # ensure numeric columns
    df["Width"] = pd.to_numeric(df.get("Width", 0), errors="coerce").fillna(0)
    df["Avg NS"] = pd.to_numeric(df.get("Avg NS", 0), errors="coerce").fillna(0)
    st.success("File uploaded")
    st.dataframe(df.head())

    if st.button("🚀 Generate Planogram (with global fill)"):
        with st.spinner("Processing..."):
            df_normal, df_yoghurt = sort_by_hierarchy(df)
            planogram_df, unassigned_df = place_and_optimize(df_normal, df_yoghurt,
                                                             shelving_count=SHELVING_COUNT,
                                                             shelf_width=SHELF_WIDTH,
                                                             max_facing=MAX_FACING_PER_SKU)
        st.subheader("3️⃣ Planogram Output")
        st.dataframe(planogram_df, use_container_width=True)

        # Download both sheets
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            planogram_df.to_excel(writer, sheet_name="Planogram", index=False)
            unassigned_df.to_excel(writer, sheet_name="Unassigned", index=False)
        st.download_button("⬇️ Download Planogram + Unassigned", buf.getvalue(), "output_planogram.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.subheader("Unassigned Items")
        st.dataframe(unassigned_df)
