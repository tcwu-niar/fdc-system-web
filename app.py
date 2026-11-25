import io
import re
from pathlib import Path
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(
    page_title="Oxford Cobra FDC 製程監控儀表板",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 隱藏側邊欄、設定 16pt 主標題與手機響應式 (RWD) 樣式
st.markdown(
    """
    <style>
        [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {
            display: none !important;
        }
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 2rem !important;
            max-width: 100% !important;
        }
        .main-title {
            font-size: 16pt !important;
            font-weight: 600 !important;
            line-height: 1.4 !important;
            margin: 0 0 0.5rem 0 !important;
            padding: 0 !important;
        }
        .section-title {
            font-size: 13pt !important;
            font-weight: 600 !important;
            line-height: 1.4 !important;
            margin: 1rem 0 0.4rem 0 !important;
        }
        /* 手機與平板窄螢幕最佳化 */
        @media (max-width: 768px) {
            .block-container {
                padding-left: 0.75rem !important;
                padding-right: 0.75rem !important;
                padding-top: 1rem !important;
            }
            .main-title {
                margin-bottom: 0.25rem !important;
            }
            div[data-testid="stHorizontalBlock"] {
                gap: 0.5rem !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

GDRIVE_FDC_ROOT_ID = "1JkFRZpoB_HJaQlevo5tPGgnjPBWPu69g"
LOCAL_CANDIDATES = [
    Path("FDC"),
    Path("G:/我的雲端硬碟/FDC"),
    Path("G:/My Drive/FDC"),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

PRIORITY_FDC_COLS = [
    "Table AMU DCBias",
    "ICP RF Forward Power",
    "ICP RF Reflected Power",
    "Table RF Forward Power",
    "Table RF Reflected Power",
    "APC Pressure",
    "APC Position",
    "ICP AMU C1 Position",
    "ICP AMU C2 Position",
    "Table AMU C1 Position",
    "Table AMU C2 Position",
    "He Flow Controller Flow",
    "He Flow Controller Pressure",
    "Table Chiller Temperature",
    "Table Heater Temperature",
    "Chamber Heater Temperature",
]

# ================= 檔名解析：自動分離 Wafer ID 與下貨日期時間 =================
def parse_wafer_and_datetime(file_name: str):
    stem = Path(file_name).stem
    dt_pattern = r"(20\d{2}[-_]?\d{2}[-_]?\d{2})\s*[-_ ]*\s*(\d{2}[-_:]?\d{2}[-_:]?\d{2})"
    m = re.search(dt_pattern, stem)
    if m:
        raw_d, raw_t = m.groups()
        d_clean = re.sub(r"[-_]", "", raw_d)
        t_clean = re.sub(r"[-_:]", "", raw_t)
        run_dt = pd.to_datetime(f"{d_clean} {t_clean}", format="%Y%m%d %H%M%S", errors="coerce")

        wafer_part = f"{stem[:m.start()]} {stem[m.end():]}".strip()
        wafer_id = re.sub(r"^[\s\-_]+|[\s\-_]+$", "", wafer_part) or stem
        if pd.notna(run_dt):
            return wafer_id, run_dt
    return stem, pd.NaT

# ================= 資料來源讀取 =================
def get_local_root() -> Path | None:
    for p in LOCAL_CANDIDATES:
        if p.exists() and p.is_dir():
            return p
    return None

@st.cache_data(ttl=120, show_spinner=False)
def list_gdrive_children(folder_id: str) -> list[dict]:
    url = f"https://drive.google.com/embeddedfolderview?id={folder_id}#list"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    if resp.status_code != 200:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    items = []
    for entry in soup.select("div.flip-entry"):
        entry_id = entry.get("id", "")
        if not entry_id.startswith("entry-"):
            continue
        item_id = entry_id.replace("entry-", "", 1)
        title_el = entry.select_one("div.flip-entry-title")
        link_el = entry.select_one("a[href]")
        if not title_el or not link_el:
            continue
        name = title_el.get_text(strip=True)
        href = link_el.get("href", "")
        items.append({
            "id": item_id,
            "name": name,
            "type": "folder" if "/folders/" in href else "file",
        })
    return items

@st.cache_data(ttl=600, show_spinner=False)
def fetch_gdrive_csv_bytes(file_id: str) -> bytes:
    session = requests.Session()
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    resp = session.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    if b"confirm=" in resp.content and b"<html" in resp.content[:200].lower():
        soup = BeautifulSoup(resp.text, "html.parser")
        form = soup.find("form", {"id": "download-form"})
        if form and form.get("action"):
            params = {i.get("name"): i.get("value") for i in form.find_all("input") if i.get("name")}
            resp = session.get(form["action"], params=params, headers=HEADERS, timeout=60)
            resp.raise_for_status()
    return resp.content

def list_subfolders(parent_ref: str, is_local: bool) -> dict[str, str]:
    if is_local:
        folders = [p for p in sorted(Path(parent_ref).iterdir()) if p.is_dir()]
        return {p.name: str(p) for p in folders}
    items = list_gdrive_children(parent_ref)
    folders = sorted([x for x in items if x["type"] == "folder"], key=lambda x: x["name"])
    return {x["name"]: x["id"] for x in folders}

def list_recipe_csvs(recipe_ref: str, is_local: bool) -> list[dict]:
    raw_files = []
    if is_local:
        for p in Path(recipe_ref).glob("*.csv"):
            wafer_id, run_dt = parse_wafer_and_datetime(p.name)
            raw_files.append({"ref": str(p), "name": p.name, "wafer_id": wafer_id, "run_dt": run_dt})
    else:
        for x in list_gdrive_children(recipe_ref):
            if x["type"] == "file" and x["name"].lower().endswith(".csv"):
                wafer_id, run_dt = parse_wafer_and_datetime(x["name"])
                raw_files.append({"ref": x["id"], "name": x["name"], "wafer_id": wafer_id, "run_dt": run_dt})

    raw_files.sort(key=lambda item: (pd.isna(item["run_dt"]), item["run_dt"], item["name"]))
    return raw_files

# ================= 製程步階偵測與統計計算 =================
def detect_plasma_steps(df: pd.DataFrame, icp_thresh: float, bias_thresh: float, min_step_sec: float = 2.0):
    icp_on = df["ICP RF Forward Power"] > icp_thresh if "ICP RF Forward Power" in df.columns else False
    rf_on = df["Table RF Forward Power"] > bias_thresh if "Table RF Forward Power" in df.columns else False
    is_plasma = icp_on | rf_on

    block_id = is_plasma.ne(is_plasma.shift()).cumsum()
    step_labels = pd.Series("Step 0 (Idle/Gas)", index=df.index)

    plasma_idx = 1
    for _, grp in df[is_plasma].groupby(block_id[is_plasma]):
        if (grp["Time"].iloc[-1] - grp["Time"].iloc[0]) >= min_step_sec:
            step_labels.loc[grp.index] = f"Plasma Step {plasma_idx}"
            plasma_idx += 1
    return step_labels

@st.cache_data(show_spinner=False)
def load_and_analyze_csv(
    file_ref: str, file_name: str, is_local: bool,
    icp_thresh: float, bias_thresh: float, trim_start: float, trim_end: float
):
    if is_local:
        df = pd.read_csv(file_ref)
    else:
        raw_bytes = fetch_gdrive_csv_bytes(file_ref)
        df = pd.read_csv(io.BytesIO(raw_bytes))

    if "Time" not in df.columns or df.empty:
        return [], pd.DataFrame()

    wafer_id, run_dt = parse_wafer_and_datetime(file_name)
    dt_str = run_dt.strftime("%Y/%m/%d %H:%M:%S") if pd.notna(run_dt) else "未標示時間"
    short_dt = run_dt.strftime("%m/%d %H:%M") if pd.notna(run_dt) else file_name
    x_label = f"{short_dt} | {wafer_id}"

    df["Step_Auto"] = detect_plasma_steps(df, icp_thresh, bias_thresh)
    sensor_cols = [c for c in df.select_dtypes(include="number").columns if c != "Time"]

    groups = [("All Process (全區段)", df)] + list(df.groupby("Step_Auto"))
    summary_rows = []

    for step_name, grp in groups:
        t_rel = grp["Time"] - grp["Time"].iloc[0]
        dur = float(t_rel.iloc[-1])
        if step_name.startswith("Plasma Step") and dur > (trim_start + trim_end):
            steady = grp[(t_rel >= trim_start) & (t_rel <= dur - trim_end)]
        else:
            steady = grp

        for col in sensor_cols:
            s = steady[col].dropna()
            if not s.empty:
                summary_rows.append({
                    "X_Label": x_label,
                    "Wafer_ID": wafer_id,
                    "Run_DateTime": dt_str,
                    "Sort_Time": run_dt,
                    "Step": step_name,
                    "Duration_s": round(dur, 2),
                    "Parameter": col,
                    "Mean": float(s.mean()),
                    "Max": float(s.max()),
                    "Min": float(s.min()),
                    "Std": float(s.std()),
                })

    trace_df = df.iloc[::10].copy()
    return summary_rows, trace_df

# ================= 頂部標題與同步按鈕 =================
title_col, btn_col = st.columns([5, 1])
with title_col:
    st.markdown('<div class="main-title">Oxford Cobra FDC 製程監控儀表板</div>', unsafe_allow_html=True)
with btn_col:
    if st.button("重新同步目錄", use_container_width=True):
        st.cache_data.clear()

local_root = get_local_root()
is_local_mode = local_root is not None
root_ref = str(local_root) if is_local_mode else GDRIVE_FDC_ROOT_ID

tool_map = list_subfolders(root_ref, is_local_mode)
if not tool_map:
    st.warning(
        "尚未讀取到雲端 FDC 下的機台資料夾。請確認已將 Google 雲端 FDC 資料夾共用設為「知道連結的任何人 (檢視者)」，設定後點擊「重新同步目錄」。"
    )
    st.stop()

# ================= 三段式主選單（桌面橫排、手機自動垂直堆疊） =================
col_tool, col_recipe, col_item = st.columns(3)

sel_tool = col_tool.selectbox("1. 選擇機台編號 (Tool ID)", list(tool_map.keys()))

recipe_map = list_subfolders(tool_map[sel_tool], is_local_mode)
if not recipe_map:
    st.info(f"機台 {sel_tool} 資料夾內目前沒有任何 Process Recipe 子資料夾。")
    st.stop()

sel_recipe = col_recipe.selectbox("2. 選擇 Process Recipe", list(recipe_map.keys()))

# 進階運算設定：改為兩列三欄，確保平板與手機畫面不擠壓
with st.expander("步階區段、管制界線 (UCL/LCL) 與暫態過濾設定", expanded=False):
    r1_c1, r1_c2, r1_c3 = st.columns(3)
    sel_stat = r1_c1.selectbox("統計特徵值", ["Mean", "Max", "Min", "Std"])
    ucl = r1_c2.number_input("上限 UCL (0=隱藏)", value=0.0)
    lcl = r1_c3.number_input("下限 LCL (0=隱藏)", value=0.0)

    r2_c1, r2_c2, r2_c3 = st.columns(3)
    icp_th = r2_c1.number_input("ICP 點火門檻 (W)", value=20.0, step=5.0)
    trim_s = r2_c2.number_input("切除開頭暫態 (秒)", min_value=0.0, value=3.0, step=0.5)
    trim_e = r2_c3.number_input("切除結尾暫態 (秒)", min_value=0.0, value=1.0, step=0.5)

csv_list = list_recipe_csvs(recipe_map[sel_recipe], is_local_mode)
if not csv_list:
    st.info(f"{sel_tool} / {sel_recipe} 資料夾內目前沒有 .csv 檔案。")
    st.stop()

all_summary = []
traces_dict = {}
prog = st.progress(0, text=f"正在載入 {sel_recipe} 共 {len(csv_list)} 片 Wafer 紀錄...")
for i, f_info in enumerate(csv_list):
    rows, t_df = load_and_analyze_csv(
        f_info["ref"], f_info["name"], is_local_mode, icp_th, 5.0, trim_s, trim_e
    )
    if rows:
        all_summary.extend(rows)
        traces_dict[rows[0]["X_Label"]] = t_df
    prog.progress((i + 1) / len(csv_list))
prog.empty()

fdc_df = pd.DataFrame(all_summary)
if fdc_df.empty:
    st.error("CSV 檔案內容格式不符（缺少 Time 或數值欄位）。")
    st.stop()

avail_params = fdc_df["Parameter"].unique().tolist()
ordered_params = [p for p in PRIORITY_FDC_COLS if p in avail_params] + [
    p for p in avail_params if p not in PRIORITY_FDC_COLS
]
sel_item = col_item.selectbox("3. 選擇 FDC Item (監控參數)", ordered_params)

avail_steps = sorted(fdc_df["Step"].unique())
default_step = next((s for s in avail_steps if s == "Plasma Step 1"), avail_steps[0])
sel_step = st.radio(
    "製程區段篩選：",
    options=avail_steps,
    index=avail_steps.index(default_step),
    horizontal=True,
)

plot_df = fdc_df[(fdc_df["Step"] == sel_step) & (fdc_df["Parameter"] == sel_item)].sort_values(
    by=["Sort_Time", "Wafer_ID"], na_position="last"
)

PLOTLY_CONFIG = {"displaylogo": False, "responsive": True}

# ================= 圖表一：FDC 趨勢管制圖 =================
st.markdown(
    f'<div class="section-title">FDC Chart：{sel_tool} / {sel_recipe} / {sel_item} ({sel_stat})</div>',
    unsafe_allow_html=True,
)
fig_fdc = px.line(
    plot_df,
    x="X_Label",
    y=sel_stat,
    markers=True,
    hover_data={
        "Wafer_ID": True,
        "Run_DateTime": True,
        "Duration_s": True,
        "Mean": ":.3f",
        "Max": ":.3f",
        "Min": ":.3f",
        "Std": ":.3f",
        "X_Label": False,
    },
)
if ucl != 0.0:
    fig_fdc.add_hline(y=ucl, line_dash="dash", line_color="red", annotation_text=f"UCL ({ucl})")
if lcl != 0.0:
    fig_fdc.add_hline(y=lcl, line_dash="dash", line_color="red", annotation_text=f"LCL ({lcl})")

fig_fdc.update_layout(
    xaxis_title="下貨日期時間 | Wafer ID",
    xaxis_tickangle=-35,
    xaxis=dict(automargin=True, tickfont=dict(size=11)),
    yaxis=dict(automargin=True, tickfont=dict(size=11)),
    margin=dict(l=10, r=10, t=20, b=40),
    height=380,
)
st.plotly_chart(fig_fdc, use_container_width=True, config=PLOTLY_CONFIG)

# ================= 圖表二 & Wafer 明細分頁 =================
tab_trace, tab_table = st.tabs(["單片原始曲線疊圖 (Trace Chart)", "已讀取 Wafer 日期時間與統計明細"])

with tab_trace:
    wafer_options = plot_df["X_Label"].tolist()
    sel_wafers = st.multiselect(
        "選擇要疊圖比對的 Wafer（預設顯示最新 2 片）：",
        options=wafer_options,
        default=wafer_options[-2:],
    )
    if sel_wafers:
        t_list = []
        for w_lbl in sel_wafers:
            sub_t = traces_dict[w_lbl][["Time", "Step_Auto", sel_item]].copy()
            sub_t["Wafer"] = w_lbl
            t_list.append(sub_t)
        fig_trace = px.line(
            pd.concat(t_list, ignore_index=True),
            x="Time",
            y=sel_item,
            color="Wafer",
            hover_data=["Step_Auto"],
        )
        # 圖例置於圖表上方橫排，避免在手機窄螢幕擠壓曲線寬度
        fig_trace.update_layout(
            xaxis_title="Time (sec)",
            xaxis=dict(automargin=True, tickfont=dict(size=11)),
            yaxis=dict(automargin=True, tickfont=dict(size=11)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
            margin=dict(l=10, r=10, t=35, b=35),
            height=380,
        )
        st.plotly_chart(fig_trace, use_container_width=True, config=PLOTLY_CONFIG)

with tab_table:
    display_cols = ["Run_DateTime", "Wafer_ID", "Step", "Duration_s", "Mean", "Max", "Min", "Std"]
    st.dataframe(plot_df[display_cols], use_container_width=True, hide_index=True)
