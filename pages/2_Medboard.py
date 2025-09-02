# dashboard.py
import streamlit as st
import pandas as pd
import json, os, time, glob
from PIL import Image

LOG_FILE = "output.jsonl"
OUT_DIR  = "yolo_outputs"

st.set_page_config(page_title="Smart EMS – Doctor Dashboard", layout="wide")
st.title("🚑 Smart EMS AI – Doctor Dashboard")

placeholder_summary = st.empty()
col1, col2 = st.columns([2, 1])
tbl = col1.empty()
img_box = col2.empty()

def load_rows():
    rows = []
    if not os.path.exists(LOG_FILE):
        return rows
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                ts = entry.get("timestamp")
                frame_id = entry.get("frame_id", None)
                for det in entry.get("detections", []):
                    rows.append({
                        "frame_id": frame_id,
                        "timestamp": ts,
                        "status": det["status"],
                        "bbox": det["bbox"]
                    })
            except:
                continue
    return rows

def latest_image():
    files = sorted(glob.glob(os.path.join(OUT_DIR, "*.jpg")))
    return files[-1] if files else None

# อัปเดตอัตโนมัติ
while True:
    rows = load_rows()
    df = pd.DataFrame(rows)
    if df.empty:
        placeholder_summary.info("กำลังรอข้อมูลจากตัวตรวจจับ…")
    else:
        bleeding = (df["status"]=="bleeding").sum()
        unconscious = (df["status"]=="unconscious").sum()
        total = len(df)

        with placeholder_summary.container():
            c1, c2, c3 = st.columns(3)
            c1.metric("Total detections", total)
            c2.metric("Bleeding", bleeding)
            c3.metric("Unconscious", unconscious)
            if bleeding > 0 or unconscious > 0:
                st.error("🚨 RED ALERT: พบผู้ป่วยวิกฤต")
            else:
                st.success("✅ GREEN: ยังไม่พบภาวะวิกฤต")

        tbl.dataframe(df.tail(100), use_container_width=True)

        img_path = latest_image()
        if img_path:
            img_box.image(Image.open(img_path), caption=f"Latest processed: {os.path.basename(img_path)}", use_container_width=True)

    time.sleep(1)
