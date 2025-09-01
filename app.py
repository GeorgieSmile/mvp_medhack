# app.py — STT → EMS Extractor (Streamlit end-to-end)
"""
EMS Speech-to-Text ไทย + EMS Extractor
เวอร์ชัน: 2025 (pipeline integrated)
ผู้พัฒนา: EMS Team
"""

import os
import time
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path

import streamlit as st
import logging
from dotenv import load_dotenv

from speech_to_text import EMSSpeechToText
# reuse robust post-process + summary from friend's module
from ems_extract import normalize_after_llm, to_human_text

# OpenAI client (v1)
from openai import OpenAI

# -------------------------
# Streamlit page & styles
# -------------------------
st.set_page_config(
    page_title="EMS Speech-to-Text ไทย + Extractor",
    page_icon="🚑",
    layout="centered",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- NEW STYLES ---
st.markdown("""
<style>
    /* General body styles */
    .stApp {
        background-color: #f0f2f6;
    }

    /* Main header with gradient */
    .main-header {
        background: linear-gradient(135deg, #5e72e4 0%, #825ee4 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }

    /* Step headers for different sections */
    .step-header {
        background-color: #ffffff;
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #5e72e4;
        margin: 1.5rem 0 1rem 0;
        font-weight: bold;
        color: #32325d;
        font-size: 1.2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Result cards for displaying outputs */
    .result-card {
        background-color: #ffffff;
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #e6ebf1;
        margin-top: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .result-card h4 {
        color: #5e72e4;
        margin-bottom: 0.75rem;
    }
    .result-card p {
        margin-bottom: 0.5rem;
    }
    
    /* Subtle text for hints or secondary info */
    .subtle {
        color: #525f7f;
        font-size: 0.9rem;
        margin-top: 1rem;
    }

    /* Custom button styles */
    .stButton>button {
        border-radius: 8px;
    }

    /* Expander styles */
    .stExpander {
        border-radius: 10px !important;
        border: 1px solid #e6ebf1 !important;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------
# Session state initialization
# -------------------------
def init_session_state():
    """Initialize all session state variables"""
    if 'stt_client' not in st.session_state:
        st.session_state.stt_client = None
        st.session_state.is_initialized = False
    
    # STT results
    if 'stt_results' not in st.session_state:
        st.session_state.stt_results = None
        st.session_state.raw_transcript = ""
        st.session_state.processing_time = 0
    
    # EMS extraction results
    if 'ems_data' not in st.session_state:
        st.session_state.ems_data = None
        st.session_state.human_text = ""
    
    # Current uploaded file info
    if 'current_file_info' not in st.session_state:
        st.session_state.current_file_info = None

# -------------------------
# Helpers
# -------------------------
def initialize_stt_client():
    """เริ่มต้นการเชื่อมต่อ Speech-to-Text client"""
    try:
        if st.session_state.stt_client is None:
            with st.spinner("🔧 กำลังเชื่อมต่อ Google Cloud Speech-to-Text..."):
                st.session_state.stt_client = EMSSpeechToText()
                st.session_state.is_initialized = True
            st.success("✅ เชื่อมต่อสำเร็จ!")
        return True
    except Exception as e:
        st.error(f"❌ การเชื่อมต่อล้มเหลว: {str(e)}")
        st.error("กรุณาตรวจสอบว่า service account JSON (เช่น neon-*.json) อยู่ในโฟลเดอร์เดียวกัน")
        return False


def build_system_prompt_for_ems() -> str:
    """ยก system prompt เดียวกับใน ems_extract.py มาใช้เพื่อได้ผลลัพธ์สอดคล้องกัน"""
    return (
        "คุณคือระบบสกัดข้อมูลเหตุฉุกเฉิน EMS ไทย/อังกฤษ "
        "ให้ตอบกลับข้อมูลเป็นภาษาไทยเท่านั้น ถ้าไม่มีคำไทยที่เหมาะสมให้ทับศัพท์ได้"
        "ให้ตอบกลับเป็น JSON เท่านั้น ตามคีย์: "
        "{unit_id,timestamp,incident:{type,mechanism,location,safety_notes},"
        "patient:{id,age,gender,consciousness,chief_complaint},"
        "vital_signs:{bp,hr,rr,spo2,temp},"
        "assessment:{injuries,suspected_conditions,triage_level},"
        "interventions,logistics:{hospital_destination,eta_minutes,special_requests}} "
        "ถ้าไม่ทราบค่า ให้เป็น null หรือ [] ห้ามมีข้อความอื่นนอกเหนือ JSON. "
        "กฎเคร่งครัด: "
        "1) patient.id ต้องเป็น null ถ้าไม่มีข้อมูล "
        "2) triage_level ต้องเป็นหนึ่งใน {RED,YELLOW,GREEN} เท่านั้น "
        "3) ถ้า unit_id เป็นตัวเลขล้วน ให้แปลงเป็น 'EMS{เลข}' เช่น 112 -> EMS112 "
        "4) injuries และ suspected_conditions: ถ้ามีข้อความบ่งชี้อาการ/โรค ให้ใส่ ห้ามปล่อยว่าง "
        "5) interventions ต้องเลือกจากลิสต์คงที่: "
        "[\"intubation\",\"oxygen\",\"IV fluids\",\"defibrillation\",\"splint\","
        "\"cooling\",\"epinephrine IM\",\"diazepam IV\",\"nebulization\"] "
        "6) ถ้า triage_level = RED และไปโรงพยาบาล ให้ใส่ special_requests เช่น "
        "[\"prepare ER\",\"alert trauma team\"] "
        "7) ห้ามมีคำอธิบายยาวในฟิลด์ interventions หรือ triage "
        "ให้ใช้คำสั้น ๆ ตามลิสต์ที่กำหนดเท่านั้น. "
        "หมายเหตุ: incident.location คือ 'ที่เกิดเหตุ'; "
        "logistics.hospital_destination คือ 'โรงพยาบาลปลายทาง'."
    )


def run_ems_extractor(raw_report: str) -> tuple[dict, str]:
    """
    เรียก LLM เพื่อสกัดข้อมูล EMS จาก raw_report
    คืนค่า (data_dict_after_post_process, human_readable_text)
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY ไม่พบ กรุณาใส่ค่าในไฟล์ .env หรือ environment")

    # แก้ไขชื่อ Model เป็น gpt-4o-mini
    client = OpenAI(api_key=api_key)

    system_msg = build_system_prompt_for_ems()
    user_msg = (
        f"รายงานกู้ภัย (ข้อความดิบ):\n{raw_report}\n"
        f"เวลารายงาน: {datetime.now().astimezone().isoformat()}"
    )

    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        response_format={"type": "json_object"}
    )

    text = resp.choices[0].message.content
    try:
        data = json.loads(text)
    except Exception as e:
        # ส่ง raw ดูได้ใน UI เพื่อดีบั๊ก
        raise RuntimeError(f"LLM ส่งคืนไม่ใช่ JSON: {e}\nRAW: {text[:500]}...")

    # ใช้ post-process แบบเดียวกับสคริปต์เพื่อน
    data = normalize_after_llm(data, raw_report)
    human_text = to_human_text(data)
    return data, human_text


def join_transcripts(results: list[dict]) -> str:
    """
    รวม transcripts หลายผลลัพธ์จาก STT เป็นสตริงเดียว
    """
    parts = []
    for r in results:
        t = (r.get("transcript") or "").strip()
        if t:
            parts.append(t)
    return " ".join(parts).strip()

# ===== Helper: ส่งข้อมูลไปหน้า Dashboard =====
def send_to_dashboard():
    data = st.session_state.get("ems_data")
    if not data:
        st.warning("ยังไม่มีข้อมูล EMS ให้ส่งไป Dashboard")
        return

    # สำรองไฟล์ล่าสุดเพื่อกัน session หลุด
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_path = out_dir / "latest_case.json"
    latest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # แชร์ผ่าน session_state
    st.session_state["dashboard_payload"] = data
    st.session_state["dashboard_latest_path"] = str(latest_path.resolve())
    
    st.success("ส่งข้อมูลไป Dashboard แล้ว ✅ — เปิดหน้า 'Dashboard' จากแถบซ้ายได้เลย")


def clear_results():
    """Clear all results from session state"""
    st.session_state.stt_results = None
    st.session_state.raw_transcript = ""
    st.session_state.processing_time = 0
    st.session_state.ems_data = None
    st.session_state.human_text = ""
    st.session_state.current_file_info = None


# -------------------------
# Main App
# -------------------------
def main():
    # Initialize session state
    init_session_state()
    
    st.markdown("""
    <div class="main-header">
        <h1>EMS Speech-to-Text & Extractor</h1>
        <p>อัปโหลดไฟล์เสียง → ถอดข้อความ → สกัดข้อมูล EMS (JSON + สรุปอ่านเร็ว)</p>
    </div>
    """, unsafe_allow_html=True)

    # เชื่อมต่อ STT อัตโนมัติเมื่อเปิดแอป
    if not st.session_state.is_initialized:
        if not initialize_stt_client():
            return

    # ========================
    # STEP 1: อัปโหลดไฟล์เสียง
    # ========================
    st.markdown('<div class="step-header">📁 ขั้นตอนที่ 1: อัปโหลดไฟล์เสียง</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "เลือกไฟล์เสียงที่ต้องการประมวลผล",
        type=['wav', 'mp3', 'flac', 'webm', 'ogg', 'm4a'],
        help="ไฟล์ที่รองรับ: WAV, MP3, FLAC, WebM, OGG, M4A"
    )

    if st.session_state.current_file_info:
        if st.button("🗑️ เริ่มใหม่ (ล้างผลลัพธ์)", help="ล้างผลลัพธ์ทั้งหมดเพื่อเริ่มใหม่"):
            clear_results()
            st.rerun()

    # ========================
    # STEP 2: การถอดข้อความ (STT)
    # ========================
    if uploaded_file:
        current_file_info = {
            "name": uploaded_file.name,
            "size": len(uploaded_file.getvalue())
        }
        uploaded_file.seek(0)
        
        is_new_file = (st.session_state.current_file_info != current_file_info)
        
        if is_new_file:
            clear_results()
            st.session_state.current_file_info = current_file_info

        st.markdown('<div class="step-header">🎤 ขั้นตอนที่ 2: แปลงเสียงเป็นข้อความ (STT)</div>', unsafe_allow_html=True)
        
        st.info(f"📄 ไฟล์ที่เลือก: **{uploaded_file.name}** | ขนาด: **{current_file_info['size']:,}** ไบต์")

        if st.button("▶️ เริ่มแปลงเสียงเป็นข้อความ", type="primary", use_container_width=True, disabled=st.session_state.stt_results is not None):
            audio_bytes = uploaded_file.read()
            effective_audio_bytes = audio_bytes
            effective_encoding = "WEBM_OPUS"
            effective_sr = 16000

            try:
                if st.session_state.stt_client.is_m4a(audio_bytes, uploaded_file.name):
                    with st.spinner("🔄 กำลังแปลงไฟล์ M4A เป็น WAV..."):
                        effective_audio_bytes = st.session_state.stt_client.convert_m4a_to_wav(
                            audio_bytes, target_sample_rate=16000
                        )
                        effective_encoding = "LINEAR16"
                        effective_sr = 16000
            except Exception as e:
                st.error(f"❌ การแปลงไฟล์ M4A ล้มเหลว: {e}")
                return

            start_time = time.time()
            with st.spinner("🔄 กำลังถอดเสียงเป็นข้อความ กรุณารอสักครู่..."):
                result = st.session_state.stt_client.transcribe_audio(
                    audio_content=effective_audio_bytes,
                    language_code="th-TH",
                    enable_automatic_punctuation=True,
                    enable_word_time_offsets=False,
                    sample_rate=effective_sr,
                    audio_encoding=effective_encoding
                )
            processing_time = time.time() - start_time

            if not result.get('success'):
                st.error(f"❌ การแปลงเสียงล้มเหลว: {result.get('error', 'ไม่ทราบข้อผิดพลาด')}")
                return

            st.session_state.stt_results = result
            st.session_state.raw_transcript = join_transcripts(result['results'])
            st.session_state.processing_time = processing_time
            
            st.rerun()

        if st.session_state.stt_results:
            st.success(f"✅ แปลงเสียงสำเร็จ! ใช้เวลา {st.session_state.processing_time:.2f} วินาที")
            
            for i, res in enumerate(st.session_state.stt_results['results'], 1):
                st.markdown(f"""
                <div class="result-card">
                    <h4>ผลลัพธ์ส่วนที่ #{i}</h4>
                    <p><strong>ข้อความที่ถอดได้:</strong> {res.get('transcript','')}</p>
                    <p><strong style="color:#525f7f;">ความมั่นใจ (Confidence):</strong> {res.get('confidence',0):.1%}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div class='subtle'>ข้อความทั้งหมดที่จะส่งเข้า EMS Extractor:</div>", unsafe_allow_html=True)
            st.code(st.session_state.raw_transcript, language='text')

            # ========================
            # STEP 3: สกัดข้อมูล EMS
            # ========================
            st.markdown('<div class="step-header">🧠 ขั้นตอนที่ 3: สกัดข้อมูล EMS (LLM)</div>', unsafe_allow_html=True)
            
            if not st.session_state.raw_transcript:
                st.warning("⚠️ ไม่มีข้อความจาก STT เพียงพอที่จะสกัดข้อมูล EMS ได้")
                return

            if st.button("🤖 เริ่มสกัดข้อมูล EMS", type="primary", use_container_width=True, disabled=st.session_state.ems_data is not None):
                try:
                    with st.spinner("🤖 กำลังวิเคราะห์และสกัดข้อมูลจากรายงาน..."):
                        data, human_text = run_ems_extractor(st.session_state.raw_transcript)
                        
                    st.session_state.ems_data = data
                    st.session_state.human_text = human_text
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ เกิดข้อผิดพลาดขณะสกัดข้อมูล: {e}")
                    return

        # ========================
        # STEP 4: แสดงผลลัพธ์และดาวน์โหลด
        # ========================
        if st.session_state.ems_data:
            st.markdown('<div class="step-header">📋 ขั้นตอนที่ 4: ผลลัพธ์และการส่งต่อข้อมูล</div>', unsafe_allow_html=True)
            
            tab1, tab2 = st.tabs(["📄 สรุปอ่านเร็ว", "🔍 JSON (ข้อมูลดิบ)"])

            with tab1:
                st.markdown("**สรุปรายงาน (SBAR/MIST style):**")
                st.text_area("สรุป", st.session_state.human_text, height=250, disabled=True)

            with tab2:
                st.markdown("**JSON (หลัง post-process):**")
                st.json(st.session_state.ems_data)

            st.divider()
            
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_bytes = json.dumps(st.session_state.ems_data, ensure_ascii=False, indent=2).encode("utf-8")
            txt_bytes = st.session_state.human_text.encode("utf-8")

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ ดาวน์โหลด JSON",
                    data=json_bytes,
                    file_name=f"case_{ts}.json",
                    mime="application/json",
                    use_container_width=True
                )
            with col2:
                st.download_button(
                    "⬇️ ดาวน์โหลด TXT (สรุป)",
                    data=txt_bytes,
                    file_name=f"case_{ts}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            out_dir = Path("outputs")
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"case_{ts}.json").write_bytes(json_bytes)
                st.caption(f"✓ บันทึกไฟล์ JSON ซ้ำลงในโฟลเดอร์ `./outputs` เรียบร้อย (case_{ts}.json)")
            except Exception as io_err:
                st.caption(f"✗ ไม่สามารถบันทึกไฟล์ลงโฟลเดอร์ `./outputs`: {io_err}")
            
            st.markdown("---")
            if st.button("📊 ส่งข้อมูลไปแสดงผลที่ Dashboard", type="primary", use_container_width=True):
                send_to_dashboard()

    # ========================
    # สถานะการประมวลผล (ท้ายหน้า)
    # ========================
    st.divider()
    
    # --- FIXED SECTION ---
    with st.expander("📊 สถานะการประมวลผลปัจจุบัน", expanded=False):
        # สร้าง HTML string แยกต่างหากเพื่อความชัดเจน
        stt_status_html = "<span style='color:green;'>✅ เชื่อมต่อแล้ว</span>" if st.session_state.is_initialized else "<span style='color:red;'>❌ ยังไม่เชื่อมต่อ</span>"
        audio_status_html = "<span style='color:green;'>✅ อัปโหลดแล้ว</span>" if st.session_state.current_file_info else "❌ ยังไม่อัปโหลด"
        stt_result_html = "<span style='color:green;'>✅ มีผลลัพธ์</span>" if st.session_state.stt_results else "❌ ยังไม่มีผลลัพธ์"
        ems_data_html = "<span style='color:green;'>✅ สกัดข้อมูลแล้ว</span>" if st.session_state.ems_data else "❌ ยังไม่มีข้อมูล"
        
        # แสดงผล
        st.markdown(f"**STT Client:** {stt_status_html}", unsafe_allow_html=True)
        st.markdown(f"**ไฟล์เสียง:** {audio_status_html}", unsafe_allow_html=True)
        st.markdown(f"**ผลลัพธ์ STT:** {stt_result_html}", unsafe_allow_html=True)
        st.markdown(f"**ข้อมูล EMS:** {ems_data_html}", unsafe_allow_html=True)
    # --- END OF FIXED SECTION ---

    st.markdown("""
    <div style="text-align: center; color: #8898aa; margin-top: 2rem;">
        <p><strong>EMS Speech-to-Text & Extractor</strong> | Version 2025</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()