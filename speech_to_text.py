"""
EMS Speech-to-Text ไทย - Google Cloud Speech-to-Text Module
Version: 2025
Author: EMS Team
"""

import os
import io
import json
from typing import Optional, List, Dict, Any
from google.cloud import speech
from google.oauth2 import service_account
import logging

# NEW: for M4A → WAV conversion
from io import BytesIO
try:
    from pydub import AudioSegment
    _HAS_PYDUB = True
except Exception:
    _HAS_PYDUB = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EMSSpeechToText:
    """
    EMS Speech-to-Text class สำหรับการแปลงเสียงเป็นข้อความภาษาไทย
    ใช้ Google Cloud Speech-to-Text API
    """
    
    def __init__(self, credentials_path: str = "neon-emitter-253311-fbdd8dc19be3.json"):
        """
        Initialize EMS Speech-to-Text client
        
        Args:
            credentials_path (str): Path to Google Cloud service account JSON file
        """
        self.credentials_path = credentials_path
        self.client = None
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize Google Cloud Speech client with service account credentials"""
        try:
            if not os.path.exists(self.credentials_path):
                raise FileNotFoundError(f"Credentials file not found: {self.credentials_path}")
                
            # Load service account credentials
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path
            )
            
            # Initialize Speech client
            self.client = speech.SpeechClient(credentials=credentials)
            logger.info("✅ EMS Speech-to-Text client initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Speech client: {str(e)}")
            raise

    # ---------- M4A helpers ----------
    def is_m4a(self, audio_content: bytes, filename: Optional[str] = None) -> bool:
        """
        ตรวจสอบว่าเป็นไฟล์ M4A/MP4 container หรือไม่
        - เช็คจากนามสกุลไฟล์ (ถ้ามี)
        - เช็ค 'ftyp' ภายใน header ของ MP4 family
        """
        try:
            if filename and filename.lower().endswith(".m4a"):
                return True
            header = audio_content[:64]
            # MP4 header: [size:4 bytes][ftyp:4 bytes]...
            # เราเช็คคำว่า 'ftyp' และแบรนด์ยอดนิยมในช่วง 64 ไบต์แรก
            return (b'ftyp' in header and any(tag in header for tag in [
                b'M4A ', b'isom', b'iso2', b'mp41', b'mp42'
            ]))
        except Exception:
            return False

    def convert_m4a_to_wav(
        self,
        audio_content: bytes,
        target_sample_rate: int = 16000
    ) -> bytes:
        """
        แปลง M4A (AAC/MP4) → WAV (LINEAR16) 16kHz mono ในหน่วยความจำ
        ต้องใช้ pydub + ffmpeg
        """
        if not _HAS_PYDUB:
            raise RuntimeError("pydub not installed. Please `pip install pydub` and install ffmpeg.")
        try:
            seg = AudioSegment.from_file(BytesIO(audio_content), format="m4a")
            seg = seg.set_frame_rate(target_sample_rate).set_channels(1).set_sample_width(2)
            buf = BytesIO()
            seg.export(buf, format="wav")  # LINEAR16 PCM
            return buf.getvalue()
        except Exception as e:
            logger.error(f"❌ M4A→WAV conversion failed: {e}")
            raise
    
    # ---------- Transcription ----------
    def transcribe_audio(
        self,
        audio_content: bytes,
        language_code: str = "th-TH",
        enable_automatic_punctuation: bool = True,
        enable_word_time_offsets: bool = True,
        sample_rate: int = 16000,
        audio_encoding: str = "WEBM_OPUS"
    ) -> Dict[str, Any]:
        """
        Transcribe audio content to text
        """
        try:
            audio = speech.RecognitionAudio(content=audio_content)
            config = speech.RecognitionConfig(
                encoding=getattr(speech.RecognitionConfig.AudioEncoding, audio_encoding),
                sample_rate_hertz=sample_rate,
                language_code=language_code,
                enable_automatic_punctuation=enable_automatic_punctuation,
                enable_word_time_offsets=enable_word_time_offsets,
                model="latest_long",
                use_enhanced=True,
            )
            logger.info("🎤 Starting speech recognition...")
            response = self.client.recognize(config=config, audio=audio)
            results = []
            for result in response.results:
                alternative = result.alternatives[0]
                words = []
                if enable_word_time_offsets:
                    for word_info in alternative.words:
                        words.append({
                            "word": word_info.word,
                            "start_time": word_info.start_time.total_seconds(),
                            "end_time": word_info.end_time.total_seconds(),
                        })
                results.append({
                    "transcript": alternative.transcript,
                    "confidence": alternative.confidence,
                    "words": words
                })
            logger.info("✅ Speech recognition completed successfully")
            return {"success": True, "results": results, "total_results": len(results)}
        except Exception as e:
            logger.error(f"❌ Speech recognition failed: {str(e)}")
            return {"success": False, "error": str(e), "results": []}
    
    def transcribe_long_audio(
        self,
        audio_content: bytes,
        language_code: str = "th-TH",
        enable_automatic_punctuation: bool = True,
        enable_word_time_offsets: bool = True,
        sample_rate: int = 16000,
        audio_encoding: str = "WEBM_OPUS"
    ) -> Dict[str, Any]:
        """
        Transcribe long audio content using long-running operation
        """
        try:
            audio = speech.RecognitionAudio(content=audio_content)
            config = speech.RecognitionConfig(
                encoding=getattr(speech.RecognitionConfig.AudioEncoding, audio_encoding),
                sample_rate_hertz=sample_rate,
                language_code=language_code,
                enable_automatic_punctuation=enable_automatic_punctuation,
                enable_word_time_offsets=enable_word_time_offsets,
                model="latest_long",
                use_enhanced=True,
            )
            logger.info("🎤 Starting long audio recognition...")
            operation = self.client.long_running_recognize(config=config, audio=audio)
            logger.info("⏳ Waiting for operation to complete...")
            response = operation.result(timeout=300)

            results = []
            for result in response.results:
                alternative = result.alternatives[0]
                words = []
                if enable_word_time_offsets:
                    for word_info in alternative.words:
                        words.append({
                            "word": word_info.word,
                            "start_time": word_info.start_time.total_seconds(),
                            "end_time": word_info.end_time.total_seconds(),
                        })
                results.append({
                    "transcript": alternative.transcript,
                    "confidence": alternative.confidence,
                    "words": words
                })

            logger.info("✅ Long audio recognition completed successfully")
            return {"success": True, "results": results, "total_results": len(results)}
        except Exception as e:
            logger.error(f"❌ Long audio recognition failed: {str(e)}")
            return {"success": False, "error": str(e), "results": []}
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes"""
        return ["th-TH", "en-US", "en-GB", "zh-CN", "ja-JP", "ko-KR"]
    
    def validate_audio_format(self, audio_content: bytes) -> bool:
        """
        Basic validation of audio content
        """
        if not audio_content or len(audio_content) < 1000:
            return False
        header = audio_content[:64]
        audio_headers = [
            b'RIFF',                 # WAV
            b'ID3',                  # MP3
            b'OggS',                 # OGG
            b'\x1a\x45\xdf\xa3',     # WebM (Matroska EBML)
        ]
        # allow MP4/M4A: look for 'ftyp' + brand
        if any(header.startswith(h) for h in audio_headers):
            return True
        if (b'ftyp' in header and any(tag in header for tag in [b'M4A ', b'isom', b'iso2', b'mp41', b'mp42'])):
            return True
        return False

# Example usage (unchanged)
if __name__ == "__main__":
    stt = EMSSpeechToText()
    try:
        with open("sample_audio.wav", "rb") as audio_file:
            audio_content = audio_file.read()
        if stt.validate_audio_format(audio_content):
            result = stt.transcribe_audio(audio_content)
            if result["success"]:
                print("🎉 Transcription Results:")
                for i, res in enumerate(result["results"], 1):
                    print(f"\n--- Result {i} ---")
                    print(f"Text: {res['transcript']}")
                    print(f"Confidence: {res['confidence']:.2%}")
                    if res["words"]:
                        print("Word-level timestamps:")
                        for word in res["words"][:5]:
                            print(f"  '{word['word']}': {word['start_time']:.2f}s - {word['end_time']:.2f}s")
            else:
                print(f"❌ Error: {result['error']}")
        else:
            print("❌ Invalid audio format")
    except FileNotFoundError:
        print("ℹ️  Sample audio file not found. Please provide an audio file to test.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
