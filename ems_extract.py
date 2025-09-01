# ems_extract.py — save JSON + human-readable TXT (with robust post-process)
import os, sys, json, datetime, re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# Helpers & Constants
# =========================
ALLOWED_INTERVENTIONS = {
    "intubation", "oxygen", "IV fluids", "defibrillation", "splint",
    "cooling", "epinephrine IM", "diazepam IV", "nebulization"
}
TRIAGE_ENUM = {"RED", "YELLOW", "GREEN"}
THAI_CHAR_RE = re.compile(r"[\u0E00-\u0E7F]")

def none_if_blank(x):
    """Return None for '', '-', 'N/A', 'na', 'null', 'None'; else stripped string or original."""
    if x is None:
        return None
    s = str(x).strip()
    return None if s.lower() in {"", "-", "n/a", "na", "null", "none"} else s

def normalize_unit_id(raw: str | None) -> str | None:
    """Normalize anything like 'รถ 221', 'EMS112', 'หน่วย 7' -> 'EMS<digits>'."""
    if not raw:
        return None
    # prefer numeric capture
    m = re.search(r"(?:EMS)?\s*(\d{1,5})\b", str(raw), flags=re.IGNORECASE)
    if m:
        return f"EMS{int(m.group(1))}"
    # fallback: already 'EMS NNN'
    m2 = re.search(r"\bEMS\s*(\d{1,5})\b", str(raw), flags=re.IGNORECASE)
    if m2:
        return f"EMS{int(m2.group(1))}"
    return str(raw).strip()

def force_triage_enum(level: str | None) -> str | None:
    if not level:
        return None
    s = str(level).strip().upper()
    return s if s in TRIAGE_ENUM else None

def map_interventions(items):
    """Map synonyms to allowed interventions; de-dup and keep order."""
    if not items:
        return []
    cleaned = []
    for it in items:
        if not it:
            continue
        t = str(it).lower()

        if re.search(r"\b(et|ett|intub)\b", t):
            key = "intubation"
        elif re.search(r"\b(o2|oxygen|o₂)\b", t):
            key = "oxygen"
        elif re.search(r"\biv\b.*\b(fluid|ns|nss|rl|ringer)\b", t) or re.search(r"\bfluid\b", t):
            key = "IV fluids"
        elif "defib" in t or "aed" in t:
            key = "defibrillation"
        elif "splint" in t or "immobil" in t:
            key = "splint"
        elif "cool" in t or "ice" in t or "evaporative" in t:
            key = "cooling"
        elif re.search(r"\b(epi|epinephrine)\b", t) and (" im" in t or t.endswith(" im")):
            key = "epinephrine IM"
        elif ("diazepam" in t or "valium" in t) and " iv" in t:
            key = "diazepam IV"
        elif "neb" in t or "nebuliz" in t or "salbutamol" in t:
            key = "nebulization"
        else:
            key = None

        if key and key in ALLOWED_INTERVENTIONS:
            cleaned.append(key)

    # de-duplicate preserving order
    uniq = []
    for x in cleaned:
        if x not in uniq:
            uniq.append(x)
    return uniq

def canonical_incident(inc: dict, raw_text: str) -> dict:
    """Force incident.type/mechanism to canonical labels when detectable; ensure location fallback."""
    t = (inc.get("type") or "").lower()
    mech = (inc.get("mechanism") or "").lower()
    raw = raw_text.lower()

    # Anaphylaxis canonicalization
    if "anaphylaxis" in raw or "แพ้" in raw or "epinephrine" in raw or re.search(r"\bepi\b", raw):
        t = "anaphylaxis"
        if "shrimp" in raw or "กุ้ง" in raw:
            mech = "food_allergy_shrimp"
        elif "อาหาร" in raw or "seafood" in raw:
            mech = "food_allergy"

    # (ต่อยอดได้ เช่น stroke, acs, seizure, heat stroke mapping)

    out = {
        "type": none_if_blank(t) or None,
        "mechanism": none_if_blank(mech) or None,
        "location": none_if_blank(inc.get("location")) or "on-scene",
        "safety_notes": none_if_blank(inc.get("safety_notes"))
    }
    return out

def preserve_thai_destination(dest: str | None, raw_text: str) -> str | None:
    """If raw_text has Thai and dest is English, try to keep Thai hospital name from raw_text."""
    dest = none_if_blank(dest)
    if not dest:
        # Try extract from raw
        m = re.search(r"(รพ\.?\s*[^\s,]+|โรงพยาบาล[^\s,]+)", raw_text)
        return m.group(1) if m else None
    # If raw contains Thai but dest has no Thai, prefer Thai from raw_text
    if THAI_CHAR_RE.search(raw_text) and not THAI_CHAR_RE.search(dest):
        m = re.search(r"(รพ\.?\s*[^\s,]+|โรงพยาบาล[^\s,]+)", raw_text)
        if m:
            return m.group(1)
    return dest

# =========================
# Human-readable TXT
# =========================
def to_human_text(d: dict) -> str:
    """สรุปผลแบบอ่านเร็ว (SBAR/MIST style)"""
    inc = d.get("incident", {}) or {}
    pt  = d.get("patient", {}) or {}
    vs  = d.get("vital_signs", {}) or {}
    ass = d.get("assessment", {}) or {}
    log = d.get("logistics", {}) or {}

    lines = []
    lines.append("=== EMS CASE SUMMARY ===")
    lines.append(f"Unit: {d.get('unit_id') or '-'}   Time: {d.get('timestamp') or '-'}")
    lines.append(f"Incident: {inc.get('type') or '-'} | Mechanism: {inc.get('mechanism') or '-'}")
    lines.append(f"Location: {inc.get('location') or '-'}")
    if inc.get("safety_notes"): lines.append(f"Safety: {inc.get('safety_notes')}")
    lines.append("")
    lines.append(f"Patient: age {pt.get('age') or '-'}, gender {pt.get('gender') or '-'}, "
                 f"consciousness: {pt.get('consciousness') or '-'}")
    lines.append(f"Chief complaint: {pt.get('chief_complaint') or '-'}")
    lines.append("")
    lines.append("Vital Signs:")
    lines.append(f"  BP: {vs.get('bp') or '-'}   HR: {vs.get('hr') or '-'}   RR: {vs.get('rr') or '-'}   "
                 f"SpO2: {vs.get('spo2') or '-'}   Temp: {vs.get('temp') or '-'}")
    lines.append("")
    tri = ass.get("triage_level") or "-"
    injuries = ", ".join(ass.get("injuries") or []) or "-"
    suspected = ", ".join(ass.get("suspected_conditions") or []) or "-"
    lines.append(f"Triage: {tri}")
    lines.append(f"Injuries: {injuries}")
    lines.append(f"Suspected conditions: {suspected}")
    lines.append("")
    inter = ", ".join(d.get("interventions") or []) or "-"
    lines.append(f"Interventions: {inter}")
    lines.append("")
    eta_str = "-"
    if (em := (d.get("logistics") or {}).get("eta_minutes")) is not None:
        eta_str = f"{em}"
    lines.append(f"Destination: {log.get('hospital_destination') or '-'}   ETA: {eta_str} min")
    sp_req = ", ".join(log.get("special_requests") or []) if log.get("special_requests") else "-"
    lines.append(f"Requests: {sp_req}")
    return "\n".join(lines)

# =========================
# Post-process (robust)
# =========================
def normalize_after_llm(data: dict, raw_text: str) -> dict:
    """Post-process: ทำให้ข้อมูลคงรูป/เชื่อถือได้มากขึ้น โดยไม่สมมติค่าเกินจำเป็น"""
    if not isinstance(data, dict):
        return {}

    # timestamp fallback -> ISO now
    if not none_if_blank(data.get("timestamp")):
        data["timestamp"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")

    # unit_id: use JSON or derive from raw_text; normalize -> EMS###
    unit = none_if_blank(data.get("unit_id"))
    if not unit:
        m = re.search(r"(?:หน่วย|unit|รถ|ems)\s*([0-9]{1,5})", raw_text, flags=re.IGNORECASE)
        unit = m.group(1) if m else None
    data["unit_id"] = normalize_unit_id(unit)

    # incident canonicalization + location fallback
    data["incident"] = canonical_incident(data.get("incident") or {}, raw_text)

    # patient clean + id reliability
    pt = data.get("patient") or {}
    for k in ("id", "age", "gender", "consciousness", "chief_complaint"):
        pt[k] = none_if_blank(pt.get(k))
    if pt.get("id"):
        pid = str(pt["id"]).strip().lower()
        if pid in {"unknown", "na", "none"} or len(pid) < 3 or not re.fullmatch(r"[A-Za-z0-9\-]{3,}", pid):
            pt["id"] = None
    data["patient"] = pt

    # vitals clean
    vs = data.get("vital_signs") or {}
    for k in ("bp", "hr", "rr", "spo2", "temp"):
        vs[k] = none_if_blank(vs.get(k))
    data["vital_signs"] = vs

    # assessment: enforce triage enum, de-dup arrays
    ass = data.get("assessment") or {}
    ass["triage_level"] = force_triage_enum(ass.get("triage_level"))
    def dedup(xs):
        out = []
        for x in (xs or []):
            if x and x not in out:
                out.append(x)
        return out
    ass["injuries"] = dedup(ass.get("injuries"))
    ass["suspected_conditions"] = dedup(ass.get("suspected_conditions"))
    data["assessment"] = ass

    # interventions mapping & de-dup
    data["interventions"] = map_interventions(data.get("interventions"))

    # logistics: preserve Thai hospital name; eta to int if possible; add default special_requests if RED
    log = data.get("logistics") or {}
    dest = preserve_thai_destination(log.get("hospital_destination"), raw_text)
    # try parse ETA from raw if missing
    eta = log.get("eta_minutes")
    if eta is None:
        # หา "ETA 7 นาที" / "ถึงใน 7 นาที"
        m = re.search(r"(?:ETA|ถึงใน)\s*:? ?(\d{1,3})\s*(?:นาที|min)", raw_text, flags=re.IGNORECASE)
        if m:
            eta = int(m.group(1))
    else:
        # sanitize eta if str
        if isinstance(eta, str):
            m = re.search(r"\d+", eta)
            eta = int(m.group(0)) if m else None

    sr = log.get("special_requests") or []
    if ass.get("triage_level") == "RED" and dest and not sr:
        sr = ["prepare ER", "alert trauma team"]

    data["logistics"] = {
        "hospital_destination": dest,
        "eta_minutes": eta,
        "special_requests": sr
    }

    return data

# =========================
# Main
# =========================
def main():
    print("RUNNING FILE :", __file__)
    print("CWD          :", os.getcwd())

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in .env")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # รับรายงานจาก args หรือใช้ค่าเริ่มต้น
    if len(sys.argv) > 1:
        raw_report = " ".join(sys.argv[1:])
    else:
        raw_report = (
            "รถ 221 รายงาน ผู้ป่วยชาย 28 ปี แพ้กุ้งรุนแรง BP 85/50 HR 120 SpO2 90% "
            "ให้ Epi IM แล้ว มุ่งหน้าไป รพ.จุฬาฯ ถึงใน 7 นาที"
        )

    system_msg = (
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

    user_msg = (
        f"รายงานกู้ภัย (ข้อความดิบ):\n{raw_report}\n"
        f"เวลารายงาน: {datetime.datetime.now().astimezone().isoformat()}"
    )

    # เรียก Chat Completions (เสถียร + บังคับ JSON)
    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        response_format={"type": "json_object"}
        # ถ้ารุ่นนี้ error เรื่อง temperature ให้เพิ่ม/ลบ parameter ตามความเหมาะสม
    )

    text = resp.choices[0].message.content
    try:
        data = json.loads(text)
    except Exception as e:
        print("RAW RESPONSE (not JSON):")
        print(text)
        print("JSON error:", e)
        sys.exit(1)

    # --- Post-process ปรับรูปแบบให้คงที่/อ่านง่าย ---
    data = normalize_after_llm(data, raw_report)

    # เตรียมโฟลเดอร์และชื่อไฟล์
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = out_dir / f"case_{ts}.json"
    txt_path  = out_dir / f"case_{ts}.txt"

    # เขียนไฟล์ JSON (pretty)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # เขียนไฟล์ TXT (อ่านง่าย)
    human_text = to_human_text(data)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(human_text + "\n")

    print(f"✅ Saved JSON: {json_path}")
    print(f"✅ Saved Text: {txt_path}")
    print("\n--- Preview (TXT) ---")
    print(human_text)

if __name__ == "__main__":
    main()
