# pages/1_Dashboard.py
import json
from pathlib import Path
from datetime import datetime
import streamlit as st

st.set_page_config(page_title="EMS Dashboard", page_icon="📊", layout="wide")

# ---------- GLOBAL STYLES ----------
st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #5e72e4 0%, #825ee4 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h2 { margin: 0 0 0.5rem 0; font-size: 2rem; }
    .main-header .subtitle { opacity: 0.9; font-size: 1rem; }

    /* Card look for st.container(border=True) */
    [data-testid="stContainer"] > div:has(> h3.card-title) {
        background: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #e6ebf1;
        padding: 1.25rem 1.25rem 1rem 1.25rem;
    }

    h3.card-title {
        color: #5e72e4;
        margin: 0 0 1rem 0;
        font-size: 1.25rem;
        display: flex; align-items: center; gap: .6rem;
    }

    .badge {
        display:inline-block; padding:.3rem .8rem; border-radius:16px;
        font-weight:700; color:#fff; font-size:.9rem;
    }
    .badge-red{background:#f5365c;} .badge-yellow{background:#fb6340;}
    .badge-green{background:#2dce89;} .badge-unknown{background:#8898aa;}

    .kv-pair{ display:flex; margin-bottom:.6rem; font-size:1rem; }
    .kv-pair .key{ color:#8898aa; width:120px; flex-shrink:0; }
    .kv-pair .value{ color:#32325d; font-weight:600; }
    
    .special-request {
        font-size: 1.2rem;      /* ขยายขนาด */
        font-weight: 700;       /* ตัวหนา */
        color: #d6336c;         /* สีเด่น เช่น แดงอมชมพู */
        margin-top: .5rem;
        display: block;
    }

    /* make metrics look boxed a bit */
    .stMetric { background:#f7fafc; border-radius:8px; padding: .75rem; border:1px solid #e6ebf1; }
    .sub-header { font-weight:700; color:#32325d; margin:.75rem 0 .25rem; }
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
def _triage_badge(level: str) -> str:
    if not level: return '<span class="badge badge-unknown">UNKNOWN</span>'
    lvl = str(level).upper()
    if lvl == "RED":    return f'<span class="badge badge-red">{lvl}</span>'
    if lvl == "YELLOW": return f'<span class="badge badge-yellow">{lvl}</span>'
    if lvl == "GREEN":  return f'<span class="badge badge-green">{lvl}</span>'
    return f'<span class="badge badge-unknown">{lvl}</span>'

def _load_latest_from_outputs():
    p = Path("outputs/latest_case.json")
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def _safe_get(d, *keys, default="-"):
    cur = d or {}
    for k in keys:
        if not isinstance(cur, dict): return default
        cur = cur.get(k)
        if cur is None: return default
    return cur if cur not in (None, "", []) else default

def display_kv(key: str, value):
    st.markdown(
        f'<div class="kv-pair"><span class="key">{key}</span> <span class="value">{value}</span></div>',
        unsafe_allow_html=True
    )

# ---------- Data source ----------
data = st.session_state.get("dashboard_payload") or _load_latest_from_outputs()

with st.sidebar:
    st.header("📂 โหลดข้อมูลเคส")
    with st.expander("อัปโหลดไฟล์ JSON", expanded=(data is None)):
        up = st.file_uploader("เลือกไฟล์ JSON ของเคส EMS", type=["json"], label_visibility="collapsed")
        if up is not None:
            try:
                data = json.load(up)
                st.session_state["dashboard_payload"] = data
                st.success("โหลดไฟล์สำเร็จ!")
                st.rerun()
            except Exception as e:
                st.error(f"ไฟล์ไม่ถูกต้อง: {e}")

if data is None:
    st.info("ยังไม่มีข้อมูลสำหรับ Dashboard\n\n- กรุณากลับไปที่หน้าหลักเพื่อ **“ส่งข้อมูลมาที่ Dashboard”**\n- หรือ **อัปโหลดไฟล์เคส JSON** จากแถบด้านข้าง")
    st.stop()

# ---------- Header ----------
unit_id = _safe_get(data, "unit_id")
ts = _safe_get(data, "timestamp")
try:
    ts_dt = datetime.fromisoformat(str(ts))
    ts_fmt = ts_dt.strftime("%d %b %Y, %H:%M:%S")
except Exception:
    ts_fmt = str(ts)

st.markdown(f"""
<div class="main-header">
  <h2>EMS Case Dashboard</h2>
  <div class="subtitle"><strong>Unit:</strong> {unit_id} &nbsp; | &nbsp; <strong>Timestamp:</strong> {ts_fmt}</div>
</div>
""", unsafe_allow_html=True)

# ---------- Top Row ----------
col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown('<h3 class="card-title">🚨 เหตุการณ์ (Incident)</h3>', unsafe_allow_html=True)
        inc = data.get("incident", {})
        display_kv("ประเภท", _safe_get(inc, 'type'))
        display_kv("สาเหตุ", _safe_get(inc, 'mechanism'))
        display_kv("สถานที่", _safe_get(inc, 'location'))
        notes = _safe_get(inc, 'safety_notes')
        if notes != "-":
            st.caption(f"⚠️ Safety Notes: {notes}")

with col2:
    with st.container(border=True):
        st.markdown('<h3 class="card-title">👤 ผู้ป่วย (Patient)</h3>', unsafe_allow_html=True)
        pt = data.get("patient", {})
        display_kv("อายุ/เพศ", f"{_safe_get(pt, 'age')} / {_safe_get(pt, 'gender')}")
        display_kv("ความรู้สึกตัว", _safe_get(pt, 'consciousness'))
        display_kv("อาการสำคัญ", _safe_get(pt, 'chief_complaint'))

with col3:
    with st.container(border=True):
        st.markdown('<h3 class="card-title">❤️ ชีพจรและระดับความรุนแรง</h3>', unsafe_allow_html=True)
        vs = data.get("vital_signs", {})
        assess = data.get("assessment", {})

        triage_html = _triage_badge(_safe_get(assess, "triage_level"))
        st.markdown(f"**Triage Level:** {triage_html}", unsafe_allow_html=True)
        st.write("")

        m1, m2 = st.columns(2)
        m1.metric("BP", _safe_get(vs, "bp"), "mmHg")
        m2.metric("HR", _safe_get(vs, "hr"), "bpm")
        m1.metric("RR", _safe_get(vs, "rr"), "bpm")
        m2.metric("SpO₂", _safe_get(vs, "spo2"), "%")
        m1.metric("Temp", _safe_get(vs, "temp"), "°C")

# ---------- Bottom Row ----------
col4, col5 = st.columns(2)

with col4:
    with st.container(border=True):
        st.markdown('<h3 class="card-title">🩺 การประเมิน (Assessment)</h3>', unsafe_allow_html=True)
        assess = data.get("assessment", {})
        st.markdown('<div class="sub-header">การบาดเจ็บ (Injuries):</div>', unsafe_allow_html=True)
        inj = _safe_get(assess, "injuries", default=[])
        st.write("• " + "\n• ".join(map(str, inj)) if inj else "-")

        st.markdown('<div class="sub-header">ภาวะที่สงสัย (Suspected Conditions):</div>', unsafe_allow_html=True)
        sus = _safe_get(assess, "suspected_conditions", default=[])
        st.write("• " + "\n• ".join(map(str, sus)) if sus else "-")

with col5:
    with st.container(border=True):
        st.markdown('<h3 class="card-title">🚑 การจัดการและนำส่ง</h3>', unsafe_allow_html=True)
        logi = data.get("logistics", {})
        inter = _safe_get(data, "interventions", default=[])
        display_kv("รพ. ปลายทาง", _safe_get(logi, 'hospital_destination'))
        display_kv("ETA", f"{_safe_get(logi, 'eta_minutes')} นาที")

        st.markdown('<div class="sub-header">หัตถการ (Interventions):</div>', unsafe_allow_html=True)
        st.write("• " + "\n• ".join(map(str, inter)) if inter else "-")

        reqs = _safe_get(logi, "special_requests", default=[])
        if reqs and reqs != "-":
            st.markdown(f"<span class='special-request'>คำขอพิเศษ: {', '.join(reqs)}</span>", unsafe_allow_html=True)

# ---------- Raw Data + Download ----------
st.divider()
with st.expander("🔍 ดูข้อมูลดิบ (Raw JSON)"):
    st.json(data)

s = json.dumps(data, ensure_ascii=False, indent=2)
st.download_button(
    "⬇️ ดาวน์โหลดข้อมูลเคส (JSON)",
    data=s.encode("utf-8"),
    file_name=f"case_dashboard_{unit_id}_{str(ts).split('T')[0]}.json",
    mime="application/json",
    use_container_width=True
)
