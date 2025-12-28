import os
import re
import traceback
from enum import Enum
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from azure.ai.formrecognizer import DocumentAnalysisClient, AnalyzeResult
from azure.core.credentials import AzureKeyCredential
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

# --- SETUP ---
load_dotenv()
app = FastAPI(title="KDxAI - Intelligent Identity Engine")

ENDPOINT = os.getenv("AZURE_DOC_INTEL_ENDPOINT")
KEY = os.getenv("AZURE_DOC_INTEL_KEY")

# --- DATA MODELS ---
class DocType(str, Enum):
    AUTO = "auto"
    PAN = "pan"
    AADHAAR = "aadhaar"
    CHEQUE = "cheque"  # NEW

class IdentityResponse(BaseModel):
    document_type: str
    id_number: str | None          # PAN / Aadhaar / Account Number
    full_name: str | None          # Name / Account Holder Name
    gender: str | None = None
    date_of_birth: str | None      # DOB (IDs) or Date (Cheque)
    address: str | None = None
    ifsc_code: str | None = None   # NEW: Cheque Specific
    micr_code: str | None = None   # NEW: Cheque Specific
    bank_name: str | None = None   # NEW: Cheque Specific
    confidence_score: float
    validation_status: str
    warnings: List[str] = []

# --- SRE HELPERS ---
def extract_gender_fallback(full_text: str) -> Optional[str]:
    match = re.search(r"\b(MALE|FEMALE|TRANSGENDER)\b", full_text, re.IGNORECASE)
    if match: return match.group(1).upper()
    return None

def extract_address_fallback(full_text: str) -> Optional[str]:
    try:
        pattern = r"(?:Address|To|Address:|To:)\s*([\s\S]*?)(\d{6})"
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            raw_addr = match.group(1).strip()
            pincode = match.group(2)
            clean_addr = re.sub(r"Print Date\s*[:\-]?\s*\d{2}/\d{2}/\d{4},?", "", raw_addr, flags=re.IGNORECASE)
            clean_addr = re.sub(r"^[:\-\s]+", "", clean_addr)
            clean_addr = re.sub(r'\s+', ' ', clean_addr).strip()
            clean_addr = clean_addr.strip(",")
            return f"{clean_addr}, {pincode}"
    except Exception: pass
    return None

def is_english(text: str) -> bool:
    try:
        ascii_chars = len([c for c in text if ord(c) < 128])
        return ascii_chars / len(text) > 0.5
    except: return False

def refine_name_using_anchor(result: AnalyzeResult, partial_name: str) -> str:
    all_lines = []
    for page in result.pages:
        all_lines.extend([line.content for line in page.lines])

    dob_index = -1
    for i, line in enumerate(all_lines):
        if "DOB" in line or "Year of Birth" in line or "Born" in line:
            dob_index = i
            break
            
    if dob_index > 0:
        candidate_line = all_lines[dob_index - 1].strip()
        if not is_english(candidate_line):
             candidate_line = all_lines[dob_index - 2].strip()
        if partial_name and partial_name.lower() in candidate_line.lower():
            return candidate_line
    return partial_name

# --- CHEQUE SPECIFIC HELPERS ---
def extract_ifsc(text: str) -> Optional[str]:
    # Regex: 4 chars, '0', 6 alphanum (e.g. SBIN0123456)
    match = re.search(r"[A-Z]{4}0[A-Z0-9]{6}", text)
    return match.group(0) if match else None

def extract_micr(text: str) -> Optional[str]:
    # Regex: 9 digits, usually surrounded by special chars or at bottom
    # We look for 9 digits that are NOT part of a longer number sequence
    matches = re.findall(r"(?<!\d)\d{9}(?!\d)", text)
    # Filter: MICR usually starts with 1-9 (zip code based)
    for m in matches:
        if m[0] != '0': return m
    return matches[0] if matches else None

def extract_account_number(text: str) -> Optional[str]:
    # Strategy 1: Look for "A/c No" label
    match = re.search(r"(?:A/c|Account|Acc)\s*(?:No|Number)?\.?\s*[:\-]?\s*(\d{9,18})", text, re.IGNORECASE)
    if match: return match.group(1)
    
    # Strategy 2: Fallback - Find longest number sequence (10-18 digits) common in India
    # Excluding numbers starting with dates like 202...
    long_numbers = re.findall(r"(?<!\d)\d{10,18}(?!\d)", text)
    if long_numbers:
        return max(long_numbers, key=len) # Return longest
    return None

# --- PROCESSORS ---
def process_cheque(result: AnalyzeResult) -> IdentityResponse:
    print("DEBUG: Using CHEQUE Strategy")
    content = result.content
    
    # 1. Hunt for Numbers
    ifsc = extract_ifsc(content)
    micr = extract_micr(content)
    acc_no = extract_account_number(content)
    
    # 2. Hunt for Bank Name (Simple heuristic based on IFSC or keywords)
    bank_name = "Unknown Bank"
    if ifsc:
        # Map first 4 chars to Bank Name (Top 5 banks for demo)
        prefix = ifsc[:4]
        banks = {"SBIN": "State Bank of India", "HDFC": "HDFC Bank", "ICIC": "ICICI Bank", "UTIB": "Axis Bank", "CNRB": "Canara Bank"}
        bank_name = banks.get(prefix, f"Bank ({prefix})")
    
    # 3. Validation
    warnings = []
    if not ifsc: warnings.append("IFSC Missing")
    if not acc_no: warnings.append("Account Number Missing")
    
    return IdentityResponse(
        document_type="CHEQUE",
        id_number=acc_no,
        full_name=None, # Name on cheque is too hard for basic OCR without positional logic
        ifsc_code=ifsc,
        micr_code=micr,
        bank_name=bank_name,
        date_of_birth=None, # Or extract Cheque Date if needed
        confidence_score=0.85, # Heuristic confidence
        validation_status="VALID" if not warnings else "REVIEW_NEEDED",
        warnings=warnings
    )

def process_pan(result: AnalyzeResult) -> IdentityResponse:
    print("DEBUG: Using PAN Strategy")
    if not result.documents:
        return IdentityResponse(document_type="UNKNOWN", id_number=None, full_name=None, confidence_score=0.0, validation_status="INVALID", warnings=["No ID detected"])
    doc = result.documents[0]
    fields = doc.fields
    id_num = fields.get("DocumentNumber").value if fields.get("DocumentNumber") else None
    
    first = fields.get("FirstName").value if fields.get("FirstName") else ""
    middle = fields.get("MiddleName").value if fields.get("MiddleName") else ""
    last = fields.get("LastName").value if fields.get("LastName") else ""
    full_name = f"{first} {middle} {last}".strip()
    full_name = re.sub(r'\s+', ' ', full_name)
    
    dob = str(fields.get("DateOfBirth").value) if fields.get("DateOfBirth") else None
    gender = extract_gender_fallback(result.content)
    warnings = []
    if not id_num: warnings.append("PAN Missing")
    elif not re.match(r"[A-Z]{5}[0-9]{4}[A-Z]{1}", id_num): warnings.append("Invalid PAN Pattern")

    return IdentityResponse(document_type="PAN_CARD", id_number=id_num, full_name=full_name, gender=gender, date_of_birth=dob, confidence_score=doc.confidence, validation_status="VALID" if not warnings else "INVALID", warnings=warnings)

def process_aadhaar(result: AnalyzeResult) -> IdentityResponse:
    print("DEBUG: Using AADHAAR Strategy")
    merged = {"id": None, "name": None, "dob": None, "address": None, "conf": []}
    for doc in result.documents:
        fields = doc.fields
        if fields.get("DocumentNumber") and not merged["id"]: merged["id"] = fields.get("DocumentNumber").value
        if not merged["name"]:
            first = fields.get("FirstName").value if fields.get("FirstName") else ""
            last = fields.get("LastName").value if fields.get("LastName") else ""
            combined = f"{first} {last}".strip()
            if combined: merged["name"] = combined
        if fields.get("Address") and not merged["address"]: merged["address"] = fields.get("Address").value
        if fields.get("DateOfBirth") and not merged["dob"]: merged["dob"] = str(fields.get("DateOfBirth").value)
        merged["conf"].append(doc.confidence)

    if merged["name"]:
        refined = refine_name_using_anchor(result, merged["name"])
        clean_refined = re.sub(r"(Name|Father's Name|Husband's Name)[:\-\s]*", "", refined, flags=re.IGNORECASE).strip()
        merged["name"] = clean_refined

    if not merged["address"]:
        addr = extract_address_fallback(result.content)
        if addr: merged["address"] = addr

    gender = extract_gender_fallback(result.content)
    warnings = []
    if merged["id"]:
        uid = merged["id"].replace(" ", "")
        if len(uid) != 12: warnings.append("Invalid Aadhaar Length")
    
    avg_conf = sum(merged["conf"]) / len(merged["conf"]) if merged["conf"] else 0.0

    return IdentityResponse(document_type="AADHAAR", id_number=merged["id"], full_name=merged["name"], gender=gender, date_of_birth=merged["dob"], address=merged["address"], confidence_score=avg_conf, validation_status="VALID" if not warnings else "REVIEW_NEEDED", warnings=warnings)

def classify_document(result: AnalyzeResult) -> DocType:
    all_content = (result.content or "").upper()
    print(f"DEBUG: Content Sample: {all_content[:100]}...")
    
    # Cheque Keywords
    if "IFSC" in all_content or "PAY " in all_content or "RUPEES" in all_content or "A/C NO" in all_content:
        # Strong signal: IFSC regex match
        if re.search(r"[A-Z]{4}0[A-Z0-9]{6}", all_content):
            return DocType.CHEQUE

    if "INCOME TAX" in all_content or "PERMANENT ACCOUNT" in all_content: return DocType.PAN
    if "UNIQUE IDENTIFICATION" in all_content or "AADHAAR" in all_content: return DocType.AADHAAR
    
    return DocType.PAN

@app.post("/extract/identity", response_model=IdentityResponse)
async def extract_identity(file: UploadFile = File(...), doc_type: DocType = Form(DocType.AUTO)):
    if not KEY or not ENDPOINT: raise HTTPException(500, "Missing Keys")
    await file.seek(0)
    content = await file.read()
    try:
        client = DocumentAnalysisClient(ENDPOINT, AzureKeyCredential(KEY))
        # Use 'prebuilt-read' if we suspect a cheque (better for raw text), but 'prebuilt-idDocument' handles IDs better.
        # For this hybrid engine, 'prebuilt-idDocument' is okay, but if cheque fails, we might need 'prebuilt-read'.
        # Let's stick to prebuilt-idDocument for now as it also does OCR.
        poller = client.begin_analyze_document("prebuilt-idDocument", document=content)
        result = poller.result()
        
        strategy = doc_type
        if strategy == DocType.AUTO: strategy = classify_document(result)
        
        print(f"DEBUG: Selected Strategy: {strategy}")

        if strategy == DocType.AADHAAR: return process_aadhaar(result)
        elif strategy == DocType.CHEQUE: return process_cheque(result)
        else: return process_pan(result)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)