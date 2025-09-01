"""
EMS Speech-to-Text ไทย - ฉบับง่าย
เวอร์ชัน: 2025
ผู้พัฒนา: EMS Team
"""

import streamlit as st
import time
from datetime import datetime
from speech_to_text import EMSSpeechToText
import logging

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="EMS Speech-to-Text ไทย",
    page_icon="🎤",
    layout="centered",
)

# ตั้งค่าการบันทึก log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CSS สำหรับปรับแต่งหน้าตา
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin-top: 1rem;
    }
    .info-box {
        background-color: #cce5ff;
        border: 1px solid #99ccff;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# khởi tạo session state
if 'stt_client' not in st.session_state:
    st.session_state.stt_client = None
    st.session_state.is_initialized = False

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
        st.error("กรุณาตรวจสอบว่ามีไฟล์ 'neon-emitter-253311-fbdd8dc19be3.json' อยู่ในโฟลเดอร์เดียวกัน")
        return False

def main():
    """แอปพลิเคชันหลัก"""
    
    st.markdown("""
    <div class="main-header">
        <h1>EMS Speech-to-Text ไทย </h1>
        <p>อัปโหลดไฟล์เสียงเพื่อแปลงเป็นข้อความและดูค่าความเชื่อมั่น</p>
    </div>
    """, unsafe_allow_html=True)

    # เชื่อมต่ออัตโนมัติเมื่อเปิดแอป
    if not st.session_state.is_initialized:
        if not initialize_stt_client():
            return # หยุดการทำงานหากเชื่อมต่อไม่สำเร็จ

    st.header("📁 อัปโหลดไฟล์เสียง")
    uploaded_file = st.file_uploader(
        "เลือกไฟล์เสียง",
        type=['wav', 'mp3', 'flac', 'webm', 'ogg', 'm4a'],
        help="ไฟล์ที่รองรับ: WAV, MP3, FLAC, WebM, OGG, M4A"
    )
    
    if uploaded_file:
        if st.button("🎤 เริ่มแปลงเสียง", type="primary", use_container_width=True):
            
            # อ่านและเตรียมไฟล์เสียง
            audio_bytes = uploaded_file.read()
            effective_audio_bytes = audio_bytes
            effective_encoding = "WEBM_OPUS" # ค่าเริ่มต้น
            effective_sr = 16000 # ค่าเริ่มต้น

            # จัดการไฟล์ M4A อัตโนมัติ
            if st.session_state.stt_client.is_m4a(audio_bytes, uploaded_file.name):
                try:
                    with st.spinner("🔄 กำลังแปลงไฟล์ M4A เป็น WAV..."):
                        effective_audio_bytes = st.session_state.stt_client.convert_m4a_to_wav(audio_bytes, target_sample_rate=16000)
                        effective_encoding = "LINEAR16"
                        effective_sr = 16000
                except Exception as e:
                    st.error(f"❌ การแปลงไฟล์ M4A ล้มเหลว: {e}")
                    return

            st.info(f"📄 ไฟล์: {uploaded_file.name} | ขนาด: {len(effective_audio_bytes):,} ไบต์")
            
            # เริ่มการแปลงเสียง
            start_time = time.time()
            with st.spinner("🔄 กำลังแปลงเสียงเป็นข้อความ..."):
                result = st.session_state.stt_client.transcribe_audio(
                    audio_content=effective_audio_bytes,
                    language_code="th-TH",
                    enable_automatic_punctuation=True,
                    enable_word_time_offsets=False, # ปิดใช้งานเพื่อความเรียบง่าย
                    sample_rate=effective_sr,
                    audio_encoding=effective_encoding
                )
            
            processing_time = time.time() - start_time
            
            if result['success']:
                st.success(f"✅ แปลงเสียงสำเร็จในเวลา {processing_time:.2f} วินาที!")
                
                st.subheader("📝 ผลลัพธ์การแปลงเสียง")
                for i, res in enumerate(result['results'], 1):
                    st.markdown(f"""
                    <div class="result-card">
                        <h4>ผลลัพท์</h4>
                        <p><strong>ข้อความ:</strong> {res['transcript']}</p>
                        <p><strong>ความมั่นใจ:</strong> {res['confidence']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error(f"❌ การแปลงเสียงล้มเหลว: {result.get('error', 'ไม่ทราบข้อผิดพลาด')}")

    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p><strong>EMS Speech-to-Text ไทย</strong> | เวอร์ชัน 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()