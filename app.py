import re
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Cobra 100 FDC 分析系統", layout="wide")
st.title("Oxford PlasmaPro 100 Cobra - FDC 圖表分析系統 (免金鑰版)")

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

def parse_filename_info(file_name: str):
    """解析 'LOT4-11F0 - 20260907 - 11-20-11.csv' 檔名中的批號與時間"""
    stem = Path(file_name).stem
    m = re.match(r"^(.*?)\s*-\s*(\d{8})\s*-\s*(\d{2}-\d{2}-\d{2})$", stem)
    if m:
        lot_id, d_str, t_str = m.groups()
        run_dt = pd.to_datetime(f"{d_str} {t_str.replace('-', ':')}", format="%Y%m%d %H:%M:%S")
        return lot_id.strip(), run_dt
    return stem, pd.Timestamp.now()

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
def process_raw_csv(file_bytes: bytes, file_name: str, recipe_tag: str, icp_thresh: float, bias_thresh: float, trim_start: float, trim_end: float):
    df = pd.read_csv(pd.io.common.BytesIO(file_bytes))
    if "Time" not in df.columns or df.empty:
        return [], None, None

    lot_id, run_time = parse_filename_info(file_name)
    run_label = f"{run_time.strftime('%Y/%m/%d %H:%M')} | {lot_id}"
    df["Step_Auto"] = detect_plasma_steps(df, icp_thresh, bias_thresh)

    sensor_cols = [c for c in df.select_dtypes(include="number").columns if c != "Time"]
    rows = []
    for step_name, grp in df.groupby("Step_Auto"):
        t_rel = grp["Time"] - grp["Time"].iloc[0]
        dur = float(t_rel.iloc[-1])
        steady = grp[(t_rel >= trim_start) & (t_rel <= dur - trim_end)] if dur > (trim_start + trim_end) else grp

        for col in sensor_cols:
            s = steady[col].dropna()
            if not s.empty:
                rows.append({
                    "Recipe": recipe_tag,
                    "Run_Label": run_label,
                    "Lot_ID": lot_id,
                    "Run_Time": str(run_time),
                    "Step": step_name,
                    "Step_Duration_s": round(dur, 2),
                    "Parameter": col,
                    "Mean": float(s.mean()),
                    "Max": float(s.max()),
                    "Min": float(s.min()),
                    "Std": float(s.std()),
                })

    # 降採樣 0.1 秒供單片曲線快速疊圖
    trace_df = df.iloc[::10].copy()
    return rows, run_label, trace_df

# ================= 側邊欄：上傳與參數控制 =================
with st.sidebar:
    st.header("📂 1. 匯入製程 Log 檔")
    st.markdown("可從 [Google 雲端 FDC 資料夾](https://drive.google.com/drive/folders/1JkFRZpoB_HJaQlevo5tPGgnjPBWPu69g) 或隨身碟將 `.csv` 拖拉至此：")

    recipe_input = st.text_input("指定這批檔案的 Recipe 名稱", value="TSVET1")
    uploaded_logs = st.file_uploader(
        "選擇或拖曳多個機台 CSV 檔 (可多選)",
        type=["csv"],
        accept_multiple_files=True,
    )

    st.divider()
    st.header("📜 2. 載入歷史 FDC 總表 (選用)")
    hist_file = st.file_uploader("上傳先前匯出的 FDC_History.csv", type=["csv"])

    st.divider()
    st.header("⚙️ 3. 步階與暫態過濾")
    icp_th = st.number_input("電漿啟動 ICP Power 門檻 (W)", value=20.0, step=5.0)
    bias_th = st.number_input("電漿啟動 Table RF 門檻 (W)", value=5.0, step=1.0)
    trim_s = st.number_input("濾除步階開頭暫態 (秒)", min_value=0.0, value=3.0, step=0.5)
    trim_e = st.number_input("濾除步階結尾暫態 (秒)", min_value=0.0, value=1.0, step=0.5)

# 整合上傳的原始 CSV 與歷史總表
summary_list = []
traces_dict = {}

if hist_file is not None:
    hist_df = pd.read_csv(hist_file)
    summary_list.append(hist_df)

if uploaded_logs:
    new_rows = []
    for up_f in uploaded_logs:
        rows, r_lbl, t_df = process_raw_csv(
            up_f.getvalue(), up_f.name, recipe_input.strip() or "Default",
            icp_th, bias_th, trim_s, trim_e
        )
        if rows:
            new_rows.extend(rows)
            traces_dict[r_lbl] = t_df
    if new_rows:
        summary_list.append(pd.DataFrame(new_rows))

if not summary_list:
    st.info("👈 請從左側欄拖曳上傳機台匯出的 `.csv` 檔案（支援一次框選多個檔案），即可立即產生 FDC 圖表！")
    st.stop()

fdc_df = pd.concat(summary_list, ignore_index=True)
fdc_df["Run_Time"] = pd.to_datetime(fdc_df["Run_Time"])
# 自動去除重複匯入的相同紀錄
fdc_df = fdc_df.drop_duplicates(subset=["Recipe", "Run_Label", "Step", "Parameter"], keep="last")

# 提供一鍵下載合併後的歷史總表，方便存回 Google 雲端
with st.sidebar:
    st.divider()
    csv_export = fdc_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "💾 下載最新 FDC_History.csv 總表",
        data=csv_export,
        file_name="FDC_History.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ================= 頂部篩選選單 =================
c1, c2, c3, c4 = st.columns(4)
sel_recipe = c1.selectbox("1. 選擇 Recipe", sorted(fdc_df["Recipe"].unique()))
df_rec = fdc_df[fdc_df["Recipe"] == sel_recipe]

steps = sorted(df_rec["Step"].unique())
default_step_idx = next((i for i, s in enumerate(steps) if s.startswith("Plasma Step 1")), 0)
sel_step = c2.selectbox("2. 選擇製程區段", steps, index=default_step_idx)

avail_params = df_rec["Parameter"].unique().tolist()
ordered_params = [p for p in PRIORITY_FDC_COLS if p in avail_params] + [p for p in avail_params if p not in PRIORITY_FDC_COLS]
sel_param = c3.selectbox("3. 選擇 FDC 監控項目", ordered_params)
sel_stat = c4.selectbox("4. 統計特徵值", ["Mean", "Max", "Min", "Std"])

chart_df = df_rec[(df_rec["Step"] == sel_step) & (df_rec["Parameter"] == sel_param)].sort_values("Run_Time")

u_col, l_col = st.columns(2)
ucl = u_col.number_input("上限 UCL (填 0 不顯示)", value=0.0)
lcl = l_col.number_input("下限 LCL (填 0 不顯示)", value=0.0)

# ================= 圖表 1：跨批次 FDC 趨勢管制圖 =================
st.subheader(f"📊 FDC 趨勢管制圖：{sel_recipe} | {sel_step} | {sel_param} ({sel_stat})")
fig_fdc = px.line(
    chart_df, x="Run_Label", y=sel_stat, markers=True,
    hover_data=["Lot_ID", "Run_Time", "Step_Duration_s", "Mean", "Max", "Min", "Std"],
)
if ucl != 0.0:
    fig_fdc.add_hline(y=ucl, line_dash="dash", line_color="red", annotation_text=f"UCL ({ucl})")
if lcl != 0.0:
    fig_fdc.add_hline(y=lcl, line_dash="dash", line_color="red", annotation_text=f"LCL ({lcl})")
fig_fdc.update_layout(xaxis_tickangle=-35, height=420)
st.plotly_chart(fig_fdc, use_container_width=True)

# ================= 圖表 2：單片每秒曲線疊圖 =================
if traces_dict:
    st.subheader(f"🔍 單片原始曲線比對 (Trace Overlay)：{sel_param}")
    avail_traces = [r for r in chart_df["Run_Label"].tolist() if r in traces_dict]
    sel_runs = st.multiselect("勾選要疊圖比對的片號", options=avail_traces, default=avail_traces[-2:])

    if sel_runs:
        t_frames = []
        for r_lbl in sel_runs:
            t_df = traces_dict[r_lbl][["Time", "Step_Auto", sel_param]].copy()
            t_df["Run_Label"] = r_lbl
            t_frames.append(t_df)
        fig_trace = px.line(
            pd.concat(t_frames, ignore_index=True),
            x="Time", y=sel_param, color="Run_Label", hover_data=["Step_Auto"]
        )
        fig_trace.update_layout(height=420)
        st.plotly_chart(fig_trace, use_container_width=True)
