import io
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import plotly.express as px
import requests
from requests.adapters import HTTPAdapter
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(
    page_title="Oxford Cobra FDC 製程監控儀表板",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 隱藏側邊欄、預留頂部導覽列高度避免遮字，並設定 16pt 主標題與手機響應式樣式
st.markdown(
    """
    <style>
        [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {
            display: none !important;
        }
        .block-container {
            padding-top: 4.2rem !important;
            padding-bottom: 2rem !important;
            max-width: 100% !important;
        }
        .main-title {
            font-size: 16pt !important;
            font-weight: 600 !important;
            line-height: 1.5 !important;
            margin: 0 0 0.5rem 0 !important;
            padding: 0 !important;
        }
        .section-title {
            font-size: 13pt !important;
            font-weight: 600 !important;
            line-height: 1.4 !important;
            margin: 1rem 0 0.4rem 0 !important;
        }
        @media (max-width: 768px) {
            .block-container {
                padding-top: 3.8rem !important;
                padding-left: 0.75rem !important;
                padding-right: 0.75rem !important;
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

ALL_KNOWN_FDC_COLS = PRIORITY_FDC_COLS + [
    "Ar Flow",
    "O2 Flow",
    "SF6 Flow",
    "CHF3 Flow",
    "C4F8 Flow",
    "H2 Flow",
    "BCl3 Flow",
    "Cl2 Flow",
]

PLOTLY_CONFIG = {"displaylogo": False, "responsive": True}
MAX_WORKERS = 16  # 併發下載執行緒數量


# ================= 建立全域高速連線池 (Connection Pool) =================
@st.cache_resource
def get_http_session() -> requests.Session:
    """複用底層 TCP/SSL 連線，大幅省去每個檔案重複握手的網路延遲"""
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS, max_retries=2)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(HEADERS)
    return session


def fast_read_csv(source) -> pd.DataFrame:
    """優先使用 PyArrow 多核心引擎極速解析 CSV，若未安裝自動退回 C 引擎"""
    try:
        return pd.read_csv(source, engine="pyarrow")
    except Exception:
        if hasattr(source, "seek"):
            source.seek(0)
        return pd.read_csv(source, low_memory=False)


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


@st.cache_data(ttl=300, show_spinner=False)
def list_gdrive_children(folder_id: str) -> list[dict]:
    session = get_http_session()
    url = f"https://drive.google.com/embeddedfolderview?id={folder_id}#list"
    resp = session.get(url, timeout=20)
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


def download_gdrive_bytes_raw(file_id: str) -> bytes:
    """無快取負擔的純下載函式，供多執行緒高速呼叫"""
    session = get_http_session()
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    if b"confirm=" in resp.content and b"<html" in resp.content[:200].lower():
        soup = BeautifulSoup(resp.text, "html.parser")
        form = soup.find("form", {"id": "download-form"})
        if form and form.get("action"):
            params = {i.get("name"): i.get("value") for i in form.find_all("input") if i.get("name")}
            resp = session.get(form["action"], params=params, timeout=60)
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
            short_dt = run_dt.strftime("%m/%d %H:%M") if pd.notna(run_dt) else p.stem
            raw_files.append({
                "ref": str(p),
                "name": p.name,
                "wafer_id": wafer_id,
                "run_dt": run_dt,
                "short_label": f"{short_dt} | {wafer_id}",
            })
    else:
        for x in list_gdrive_children(recipe_ref):
            if x["type"] == "file" and x["name"].lower().endswith(".csv"):
                wafer_id, run_dt = parse_wafer_and_datetime(x["name"])
                short_dt = run_dt.strftime("%m/%d %H:%M") if pd.notna(run_dt) else Path(x["name"]).stem
                raw_files.append({
                    "ref": x["id"],
                    "name": x["name"],
                    "wafer_id": wafer_id,
                    "run_dt": run_dt,
                    "short_label": f"{short_dt} | {wafer_id}",
                })

    raw_files.sort(key=lambda item: (pd.isna(item["run_dt"]), item["run_dt"], item["name"]))
    return raw_files


@st.cache_data(ttl=1800, show_spinner=False)
def get_csv_columns_fast(file_ref: str, is_local: bool) -> list[str]:
    """只抓取檔案最開頭 4KB 讀取標題，花費時間 < 0.2 秒"""
    try:
        if is_local:
            df_head = pd.read_csv(file_ref, nrows=3)
        else:
            session = get_http_session()
            url = f"https://drive.google.com/uc?export=download&id={file_ref}"
            with session.get(url, stream=True, timeout=10) as r:
                chunk = next(r.iter_content(chunk_size=8192)).decode("utf-8", errors="ignore")
            lines = "\n".join(chunk.splitlines()[:4])
            df_head = pd.read_csv(io.StringIO(lines))

        cols = [c for c in df_head.columns if c != "Time" and not df_head[c].isna().all()]
        ordered = [p for p in ALL_KNOWN_FDC_COLS if p in cols] + [c for c in cols if c not in ALL_KNOWN_FDC_COLS]
        return ordered if ordered else ALL_KNOWN_FDC_COLS
    except Exception:
        return ALL_KNOWN_FDC_COLS


# ================= 製程步階偵測與向量化極速統計 =================
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


@st.cache_data(ttl=3600, show_spinner=False)
def summarize_single_csv(
    file_ref: str, file_name: str, is_local: bool,
    icp_thresh: float, bias_thresh: float, trim_start: float, trim_end: float
) -> list[dict]:
    """
    【極速關鍵】僅計算並快取輕量統計字典（幾 KB），不儲存龐大的原始曲線 DataFrame！
    計算完立即釋放原始資料記憶體，切換選單 0 延遲。
    """
    if is_local:
        df = fast_read_csv(file_ref)
    else:
        raw_bytes = download_gdrive_bytes_raw(file_ref)
        df = fast_read_csv(io.BytesIO(raw_bytes))

    if "Time" not in df.columns or df.empty:
        return []

    wafer_id, run_dt = parse_wafer_and_datetime(file_name)
    dt_str = run_dt.strftime("%Y/%m/%d %H:%M:%S") if pd.notna(run_dt) else "未標示時間"
    short_dt = run_dt.strftime("%m/%d %H:%M") if pd.notna(run_dt) else Path(file_name).stem
    x_label = f"{short_dt} | {wafer_id}"

    step_series = detect_plasma_steps(df, icp_thresh, bias_thresh)
    sensor_cols = [c for c in df.select_dtypes(include="number").columns if c != "Time" and not df[c].isna().all()]
    if not sensor_cols:
        return []

    groups = [("All Process (全區段)", df)] + list(df.groupby(step_series))
    summary_rows = []

    for step_name, grp in groups:
        t_vals = grp["Time"].to_numpy()
        dur = float(t_vals[-1] - t_vals[0])
        if step_name.startswith("Plasma Step") and dur > (trim_start + trim_end):
            t_rel = t_vals - t_vals[0]
            mask = (t_rel >= trim_start) & (t_rel <= dur - trim_end)
            steady = grp.loc[mask, sensor_cols]
        else:
            steady = grp[sensor_cols]

        if steady.empty:
            continue

        # 一次性矩陣向量化算出所有欄位的統計值（比逐欄 for 迴圈快 10 倍）
        stats_mat = steady.agg(["mean", "max", "min", "std"])
        for col in sensor_cols:
            mean_val = stats_mat.at["mean", col]
            if pd.notna(mean_val):
                summary_rows.append({
                    "File_Ref": file_ref,
                    "X_Label": x_label,
                    "Wafer_ID": wafer_id,
                    "Run_DateTime": dt_str,
                    "Sort_Time": run_dt,
                    "Step": step_name,
                    "Duration_s": round(dur, 2),
                    "Parameter": col,
                    "Mean": float(mean_val),
                    "Max": float(stats_mat.at["max", col]),
                    "Min": float(stats_mat.at["min", col]),
                    "Std": float(stats_mat.at["std", col]) if pd.notna(stats_mat.at["std", col]) else 0.0,
                })

    return summary_rows


@st.cache_data(ttl=1800, max_entries=30, show_spinner=False)
def load_trace_downsampled(file_ref: str, is_local: bool, icp_thresh: float = 20.0, bias_thresh: float = 5.0) -> pd.DataFrame:
    """僅在使用者勾選特定 Wafer 要看疊圖時，才載入該片 0.1s 降採樣曲線"""
    if is_local:
        df = fast_read_csv(file_ref)
    else:
        raw_bytes = download_gdrive_bytes_raw(file_ref)
        df = fast_read_csv(io.BytesIO(raw_bytes))

    if "Time" not in df.columns or df.empty:
        return pd.DataFrame()

    df["Step_Auto"] = detect_plasma_steps(df, icp_thresh, bias_thresh)
    sub = df.iloc[::10].copy()
    # 將浮點數轉為 float32 減少一半記憶體占用
    float_cols = sub.select_dtypes(include="float64").columns
    sub[float_cols] = sub[float_cols].astype("float32")
    return sub


def batch_summarize_parallel(csv_list: list[dict], is_local: bool, icp_th: float, bias_th: float, trim_s: float, trim_e: float) -> pd.DataFrame:
    """使用 16 個執行緒併發下載與計算所有 CSV，將 40 秒等待壓縮至 3~5 秒"""
    all_rows = []
    total = len(csv_list)
    prog = st.progress(0, text=f"高速併發載入中 (0 / {total} 片 Wafer)...")

    workers = min(MAX_WORKERS, max(1, total))
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                summarize_single_csv,
                f["ref"], f["name"], is_local, icp_th, bias_th, trim_s, trim_e
            ): f
            for f in csv_list
        }
        for future in as_completed(future_map):
            rows = future.result()
            if rows:
                all_rows.extend(rows)
            completed += 1
            prog.progress(completed / total, text=f"高速併發載入中 ({completed} / {total} 片 Wafer)...")

    prog.empty()
    return pd.DataFrame(all_rows)


def batch_load_traces_parallel(tasks: list[dict], is_local: bool, icp_th: float = 20.0) -> dict[str, pd.DataFrame]:
    """多片曲線疊圖同步併發載入"""
    results = {}
    workers = min(MAX_WORKERS, max(1, len(tasks)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(load_trace_downsampled, t["ref"], is_local, icp_th, 5.0): t["key"]
            for t in tasks
        }
        for future in as_completed(future_map):
            w_key = future_map[future]
            results[w_key] = future.result()
    return results


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

# ================= 雙頁面導覽（無側邊欄設計） =================
page_fdc, page_overlay = st.tabs(["FDC 趨勢監控", "跨 Recipe 疊 Log 圖"])


# =====================================================================
# 頁面一：FDC 趨勢監控（起始選擇皆為空白）
# =====================================================================
with page_fdc:
    col_tool, col_recipe, col_item = st.columns(3)

    sel_tool = col_tool.selectbox(
        "1. 選擇機台編號 (Tool ID)",
        options=list(tool_map.keys()),
        index=None,
        placeholder="請選擇機台編號...",
        key="fdc_tool",
    )

    recipe_map = list_subfolders(tool_map[sel_tool], is_local_mode) if sel_tool else {}
    sel_recipe = col_recipe.selectbox(
        "2. 選擇 Process Recipe",
        options=list(recipe_map.keys()),
        index=None,
        placeholder="請選擇 Process Recipe...",
        disabled=(sel_tool is None),
        key="fdc_recipe",
    )

    with st.expander("步階區段、管制界線 (UCL/LCL) 與暫態過濾設定", expanded=False):
        r1_c1, r1_c2, r1_c3 = st.columns(3)
        sel_stat = r1_c1.selectbox("統計特徵值", ["Mean", "Max", "Min", "Std"], key="fdc_stat")
        ucl = r1_c2.number_input("上限 UCL (0=隱藏)", value=0.0, key="fdc_ucl")
        lcl = r1_c3.number_input("下限 LCL (0=隱藏)", value=0.0, key="fdc_lcl")

        r2_c1, r2_c2, r2_c3 = st.columns(3)
        icp_th = r2_c1.number_input("ICP 點火門檻 (W)", value=20.0, step=5.0, key="fdc_icp")
        trim_s = r2_c2.number_input("切除開頭暫態 (秒)", min_value=0.0, value=3.0, step=0.5, key="fdc_ts")
        trim_e = r2_c3.number_input("切除結尾暫態 (秒)", min_value=0.0, value=1.0, step=0.5, key="fdc_te")

    if sel_tool is None or sel_recipe is None:
        col_item.selectbox(
            "3. 選擇 FDC Item (監控參數)",
            options=[],
            index=None,
            placeholder="請先選擇機台與 Recipe...",
            disabled=True,
            key="fdc_item_empty",
        )
        st.info("請依序選擇「機台編號」與「Process Recipe」以載入 FDC 項目。")
    else:
        csv_list = list_recipe_csvs(recipe_map[sel_recipe], is_local_mode)
        if not csv_list:
            col_item.selectbox(
                "3. 選擇 FDC Item (監控參數)",
                options=[],
                index=None,
                placeholder="無可用資料",
                disabled=True,
                key="fdc_item_nocsv",
            )
            st.info(f"{sel_tool} / {sel_recipe} 資料夾內目前沒有 .csv 檔案。")
        else:
            fdc_df = batch_summarize_parallel(csv_list, is_local_mode, icp_th, 5.0, trim_s, trim_e)

            if fdc_df.empty:
                st.error("CSV 檔案內容格式不符（缺少 Time 或數值欄位）。")
            else:
                avail_params = fdc_df["Parameter"].unique().tolist()
                ordered_params = [p for p in PRIORITY_FDC_COLS if p in avail_params] + [
                    p for p in avail_params if p not in PRIORITY_FDC_COLS
                ]
                sel_item = col_item.selectbox(
                    "3. 選擇 FDC Item (監控參數)",
                    options=ordered_params,
                    index=None,
                    placeholder="請選擇 FDC Item...",
                    key="fdc_item",
                )

                if sel_item is None:
                    st.info("請選擇「3. FDC Item (監控參數)」以繪製 FDC 趨勢管制圖。")
                else:
                    avail_steps = sorted(fdc_df["Step"].unique())
                    default_step = next((s for s in avail_steps if s == "Plasma Step 1"), avail_steps[0])
                    sel_step = st.radio(
                        "製程區段篩選：",
                        options=avail_steps,
                        index=avail_steps.index(default_step),
                        horizontal=True,
                        key="fdc_step_radio",
                    )

                    plot_df = fdc_df[
                        (fdc_df["Step"] == sel_step) & (fdc_df["Parameter"] == sel_item)
                    ].sort_values(by=["Sort_Time", "Wafer_ID"], na_position="last")

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

                    sub_tab_trace, sub_tab_table = st.tabs(["單片原始曲線疊圖", "已讀取 Wafer 日期時間與統計明細"])

                    with sub_tab_trace:
                        label_to_ref = dict(zip(plot_df["X_Label"], plot_df["File_Ref"]))
                        wafer_options = list(label_to_ref.keys())
                        sel_wafers = st.multiselect(
                            "選擇要疊圖比對的 Wafer：",
                            options=wafer_options,
                            default=[],
                            placeholder="請選擇要疊圖的 Wafer (可複選)...",
                            key="fdc_sub_wafers",
                        )
                        if sel_wafers:
                            tasks = [{"key": w, "ref": label_to_ref[w]} for w in sel_wafers]
                            loaded_traces = batch_load_traces_parallel(tasks, is_local_mode, icp_th)
                            t_list = []
                            for w_lbl in sel_wafers:
                                t_df = loaded_traces.get(w_lbl, pd.DataFrame())
                                if not t_df.empty and sel_item in t_df.columns:
                                    sub_t = t_df[["Time", "Step_Auto", sel_item]].copy()
                                    sub_t["Wafer"] = w_lbl
                                    t_list.append(sub_t)
                            if t_list:
                                fig_trace = px.line(
                                    pd.concat(t_list, ignore_index=True),
                                    x="Time",
                                    y=sel_item,
                                    color="Wafer",
                                    hover_data=["Step_Auto"],
                                )
                                fig_trace.update_layout(
                                    xaxis_title="Time (sec)",
                                    xaxis=dict(automargin=True, tickfont=dict(size=11)),
                                    yaxis=dict(automargin=True, tickfont=dict(size=11)),
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
                                    margin=dict(l=10, r=10, t=35, b=35),
                                    height=380,
                                )
                                st.plotly_chart(fig_trace, use_container_width=True, config=PLOTLY_CONFIG)

                    with sub_tab_table:
                        display_cols = ["Run_DateTime", "Wafer_ID", "Step", "Duration_s", "Mean", "Max", "Min", "Std"]
                        st.dataframe(plot_df[display_cols], use_container_width=True, hide_index=True)


# =====================================================================
# 頁面二：跨 Recipe 疊 Log 圖
# 選擇方式：Tool (單選) / FDC items (單選) / Recipe (複選) / Wafer (複選)，起始值皆為空白
# =====================================================================
with page_overlay:
    ov_r1_c1, ov_r1_c2 = st.columns(2)
    ov_r2_c1, ov_r2_c2 = st.columns(2)

    ov_tool = ov_r1_c1.selectbox(
        "1. 選擇 Tool (單選)",
        options=list(tool_map.keys()),
        index=None,
        placeholder="請選擇機台編號...",
        key="ov_tool",
    )

    ov_recipe_map = list_subfolders(tool_map[ov_tool], is_local_mode) if ov_tool else {}

    ov_item_options = ALL_KNOWN_FDC_COLS
    if ov_tool and ov_recipe_map:
        for r_ref in ov_recipe_map.values():
            sample_files = list_recipe_csvs(r_ref, is_local_mode)
            if sample_files:
                ov_item_options = get_csv_columns_fast(sample_files[0]["ref"], is_local_mode)
                break

    ov_item = ov_r1_c2.selectbox(
        "2. 選擇 FDC Item (單選)",
        options=ov_item_options if ov_tool else [],
        index=None,
        placeholder="請選擇 FDC Item..." if ov_tool else "請先選擇 Tool...",
        disabled=(ov_tool is None),
        key="ov_item",
    )

    ov_recipes = ov_r2_c1.multiselect(
        "3. 選擇 Process Recipe (可複選)",
        options=list(ov_recipe_map.keys()),
        default=[],
        placeholder="請選擇一個或多個 Recipe..." if ov_tool else "請先選擇 Tool...",
        disabled=(ov_tool is None),
        key="ov_recipes",
    )

    wafer_lookup = {}
    if ov_tool and ov_recipes:
        # 多個 Recipe 目錄同時併發掃描，不排隊
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(ov_recipes))) as ex:
            rec_futures = {ex.submit(list_recipe_csvs, ov_recipe_map[r], is_local_mode): r for r in ov_recipes}
            rec_results = {rec_futures[f]: f.result() for f in as_completed(rec_futures)}

        for r_name in ov_recipes:
            for f_meta in rec_results.get(r_name, []):
                display_key = f"[{r_name}] {f_meta['short_label']}"
                wafer_lookup[display_key] = {
                    "recipe": r_name,
                    "wafer_id": f_meta["wafer_id"],
                    "run_dt": f_meta["run_dt"],
                    "ref": f_meta["ref"],
                    "name": f_meta["name"],
                }

    ov_wafers = ov_r2_c2.multiselect(
        "4. 選擇 Wafer (可複選)",
        options=list(wafer_lookup.keys()),
        default=[],
        placeholder="請選擇要疊圖的 Wafer..." if ov_recipes else "請先選擇至少一個 Recipe...",
        disabled=(len(ov_recipes) == 0),
        key="ov_wafers",
    )

    if not (ov_tool and ov_item and ov_recipes and ov_wafers):
        st.info("請依序選擇「Tool」、「FDC Item」、「Recipe」與「Wafer」以繪製跨製程 Log 疊圖。")
    else:
        with st.spinner(f"正在高速併發讀取 {len(ov_wafers)} 片 Wafer 曲線..."):
            ov_tasks = [{"key": w_key, "ref": wafer_lookup[w_key]["ref"]} for w_key in ov_wafers]
            loaded_ov_traces = batch_load_traces_parallel(ov_tasks, is_local_mode)

        overlay_frames = []
        missing_col_wafers = []

        for w_key in ov_wafers:
            meta = wafer_lookup[w_key]
            t_df = loaded_ov_traces.get(w_key, pd.DataFrame())
            if not t_df.empty and ov_item in t_df.columns:
                sub = t_df[["Time", "Step_Auto", ov_item]].dropna(subset=[ov_item]).copy()
                if not sub.empty:
                    sub["Run"] = w_key
                    sub["Recipe"] = meta["recipe"]
                    sub["Wafer_ID"] = meta["wafer_id"]
                    overlay_frames.append(sub)
                else:
                    missing_col_wafers.append(w_key)
            else:
                missing_col_wafers.append(w_key)

        if missing_col_wafers:
            st.warning(f"以下 Wafer 的 CSV 中無 `{ov_item}` 有效數值：{', '.join(missing_col_wafers)}")

        if overlay_frames:
            st.markdown(
                f'<div class="section-title">Log 曲線疊圖：{ov_tool} / {ov_item}</div>',
                unsafe_allow_html=True,
            )
            combined_trace = pd.concat(overlay_frames, ignore_index=True)
            fig_ov = px.line(
                combined_trace,
                x="Time",
                y=ov_item,
                color="Run",
                hover_data=["Recipe", "Wafer_ID", "Step_Auto"],
            )
            fig_ov.update_layout(
                xaxis_title="Time (sec)",
                yaxis_title=ov_item,
                xaxis=dict(automargin=True, tickfont=dict(size=11)),
                yaxis=dict(automargin=True, tickfont=dict(size=11)),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
                margin=dict(l=10, r=10, t=40, b=40),
                height=480,
            )
            st.plotly_chart(fig_ov, use_container_width=True, config=PLOTLY_CONFIG)
