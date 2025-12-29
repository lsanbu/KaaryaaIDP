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
app = FastAPI(title="Kaaryaa IDP - Intelligent Identity Engine")

ENDPOINT = os.getenv("AZURE_DOC_INTEL_ENDPOINT")
KEY = os.getenv("AZURE_DOC_INTEL_KEY")

# --- DATA MODELS ---
class DocType(str, Enum):
    AUTO = "auto"
    PAN = "pan"
    AADHAAR = "aadhaar"
    CHEQUE = "cheque"
    FORM16 = "form16"
    ITRV = "itrv"

class IdentityResponse(BaseModel):
    document_type: str
    id_number: str | None = None
    full_name: str | None = None
    gender: str | None = None
    date_of_birth: str | None = None
    address: str | None = None
    
    # Banking Fields
    ifsc_code: str | None = None   
    micr_code: str | None = None   
    bank_name: str | None = None   
    
    # Income Fields
    employer_name: str | None = None
    assessment_year: str | None = None
    gross_income: str | None = None
    tax_paid: str | None = None
    
    confidence_score: float
    validation_status: str
    warnings: List[str] = []

# --- GENERAL HELPERS ---
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
            return f"{clean_addr}, {pincode}"
    except Exception: pass
    return None

def refine_name_using_anchor(result: AnalyzeResult, partial_name: str) -> str:
    all_lines = []
    for page in result.pages:
        all_lines.extend([line.content for line in page.lines])
    
    for i, line in enumerate(all_lines):
        if "DOB" in line or "Year of Birth" in line:
            candidate = all_lines[i-1].strip()
            if partial_name.lower() in candidate.lower():
                return candidate
    return partial_name

# --- CHEQUE HELPERS ---
def extract_ifsc(text: str) -> Optional[str]:
    match = re.search(r"[A-Z]{4}0[A-Z0-9]{6}", text)
    return match.group(0) if match else None

def extract_account_number(text: str) -> Optional[str]:
    match = re.search(r"(?:A/c|Account|Acc)\s*(?:No|Number)?\.?\s*[:\-]?\s*(\d{9,18})", text, re.IGNORECASE)
    if match: return match.group(1)
    long_numbers = re.findall(r"(?<!\d)\d{10,18}(?!\d)", text)
    return max(long_numbers, key=len) if long_numbers else None

# --- INCOME PROCESSORS (V18 - UNIVERSAL WHITESPACE REGEX) ---
def process_form16(result: AnalyzeResult) -> IdentityResponse:
    print("DEBUG: Using Form 16 V18 (Universal Regex)")
    content = result.content or ""
    
    # 1. Employer Name
    employer = None
    lines = [line.content for page in result.pages for line in page.lines]
    for i, line in enumerate(lines):
        if "Name and address of the Employer" in line or "Name and address of the Employer/Specified Bank" in line:
            for offset in range(1, 4):
                if i + offset >= len(lines): break
                candidate = lines[i+offset].strip()
                if "Name and address" in candidate or "Employee" in candidate or "Specified senior" in candidate:
                    continue
                employer = candidate
                if i + offset + 1 < len(lines):
                    next_l = lines[i+offset+1].strip()
                    if next_l.isupper() and ("PRIVATE" in next_l or "LIMITED" in next_l):
                        employer += " " + next_l
                break
            break

    # 2. Assessment Year
    ay_match = re.search(r"Assessment\s*Year\s*[:\-]?\s*(\d{4}-\d{2})", content, re.IGNORECASE)
    ay = ay_match.group(1) if ay_match else None

    # 3. Income Extraction
    gross = None
    
    # --- STRATEGY A: THE UNIVERSAL REGEX (Works on Layout AND ID Models) ---
    # Matches "12." -> Any space/newline -> "Total taxable income" -> Any space/newline -> "(9-11)" -> Number
    # We allow optional spaces inside (9-11) just in case: \(9\s*-\s*11\)
    universal_pattern = r"12\.\s+Total\s+taxable\s+income\s*\(9\s*-\s*11\)\s+(\d[\d,]*\.\d{2})"
    
    match = re.search(universal_pattern, content, re.IGNORECASE)
    if match:
        gross = match.group(1)
        print(f"DEBUG: Extracted via Universal Regex: {gross}")

    # --- STRATEGY B: TABLE LOOKUP (Best for Layout Model) ---
    if not gross and result.tables:
        target_row_keywords = ["Total taxable income", "Gross Salary", "Net Salary"]
        for table in result.tables:
            for cell in table.cells:
                cell_text = cell.content.replace("\n", " ").strip()
                if any(k.lower() in cell_text.lower() for k in target_row_keywords):
                    # Get all cells in this row
                    row_cells = [c for c in table.cells if c.row_index == cell.row_index]
                    row_cells.sort(key=lambda x: x.column_index)
                    # Iterate backwards to find the number
                    for c in reversed(row_cells):
                        val = re.sub(r"[^\d\.]", "", c.content)
                        # Look for a number > 1000 to avoid row numbers
                        if re.match(r"^\d+(\.\d{1,2})?$", val) and float(val) > 1000:
                            gross = c.content
                            print(f"DEBUG: Extracted via Table: {gross}")
                            break
                    if gross: break
            if gross: break

    # --- STRATEGY C: GENERIC FALLBACK (Last Resort) ---
    if not gross:
        # Fallback for standard forms: Look for "Total taxable income" followed closely by a big number
        regex_hunt = re.search(r"(?:Total\s*taxable\s*income|Gross\s*Salary)[\s\S]{1,150}?(\d[\d,]*\.\d{2})", content, re.IGNORECASE)
        if regex_hunt:
            gross = regex_hunt.group(1)
            print(f"DEBUG: Extracted via Generic Regex: {gross}")

    # 4. Tax Payable
    tax_match = re.search(r"(?:Net\s*tax\s*payable|Total\s*Tax\s*Payable)[\s\S]{0,100}?(\d[\d,]*\.\d{2})", content, re.IGNORECASE)
    tax = tax_match.group(1) if tax_match else None
    
    warnings = []
    if not gross: warnings.append("Income info not found")
    if not employer: warnings.append("Employer Name not found")
    
    return IdentityResponse(
        document_type="FORM_16",
        employer_name=employer,
        assessment_year=ay,
        gross_income=gross,
        tax_paid=tax,
        confidence_score=0.9,
        validation_status="VALID" if not warnings else "REVIEW_NEEDED",
        warnings=warnings
    )

def process_itrv(result: AnalyzeResult) -> IdentityResponse:
    content = result.content or ""
    ay = re.search(r"Assessment\s*Year\s*[:\-]?\s*(\d{4}-\d{2})", content, re.IGNORECASE)
    income = re.search(r"Gross\s*Total\s*Income[\s\S]*?(\d{1,3}(?:,\d{2,3})*)", content, re.IGNORECASE)
    tax = re.search(r"(?:Total\s*Tax\s*Payable|Refund)\s*[\s\S]*?(\d{1,3}(?:,\d{2,3})*)", content, re.IGNORECASE)
    pan = re.search(r"[A-Z]{5}[0-9]{4}[A-Z]{1}", content)
    return IdentityResponse(document_type="ITR-V", id_number=pan.group(0) if pan else None, assessment_year=ay.group(1) if ay else None, gross_income=income.group(1) if income else None, tax_paid=tax.group(1) if tax else None, confidence_score=0.9, validation_status="VALID", warnings=[])

# --- ID PROCESSORS ---
def process_aadhaar(result: AnalyzeResult) -> IdentityResponse:
    merged = {"id": None, "name": None, "dob": None, "address": None, "conf": []}
    for doc in result.documents:
        fields = doc.fields
        if fields.get("DocumentNumber") and not merged["id"]: merged["id"] = fields.get("DocumentNumber").value
        if not merged["name"]:
            f = fields.get("FirstName").value if fields.get("FirstName") else ""
            l = fields.get("LastName").value if fields.get("LastName") else ""
            merged["name"] = f"{f} {l}".strip()
        if fields.get("Address") and not merged["address"]: merged["address"] = fields.get("Address").value
        if fields.get("DateOfBirth") and not merged["dob"]: merged["dob"] = str(fields.get("DateOfBirth").value)
        merged["conf"].append(doc.confidence)
    if merged["name"]:
        refined = refine_name_using_anchor(result, merged["name"])
        merged["name"] = re.sub(r"(Name|Father's Name)[:\-\s]*", "", refined, flags=re.IGNORECASE).strip()
    if not merged["address"]:
        addr = extract_address_fallback(result.content)
        if addr: merged["address"] = addr
    warnings = []
    if merged["id"] and len(merged["id"].replace(" ","")) != 12: warnings.append("Invalid Aadhaar Length")
    return IdentityResponse(document_type="AADHAAR", id_number=merged["id"], full_name=merged["name"], gender=extract_gender_fallback(result.content), date_of_birth=merged["dob"], address=merged["address"], confidence_score=0.8, validation_status="VALID" if not warnings else "REVIEW_NEEDED", warnings=warnings)

def process_pan(result: AnalyzeResult) -> IdentityResponse:
    if not result.documents: return IdentityResponse(document_type="UNKNOWN", confidence_score=0, validation_status="INVALID")
    doc = result.documents[0]
    fields = doc.fields
    id_num = fields.get("DocumentNumber").value if fields.get("DocumentNumber") else None
    f = fields.get("FirstName").value if fields.get("FirstName") else ""
    l = fields.get("LastName").value if fields.get("LastName") else ""
    return IdentityResponse(document_type="PAN_CARD", id_number=id_num, full_name=f"{f} {l}".strip(), gender=extract_gender_fallback(result.content), date_of_birth=str(fields.get("DateOfBirth").value) if fields.get("DateOfBirth") else None, confidence_score=doc.confidence, validation_status="VALID", warnings=[])

def process_cheque(result: AnalyzeResult) -> IdentityResponse:
    content = result.content
    ifsc = extract_ifsc(content)
    acc = extract_account_number(content)
    return IdentityResponse(document_type="CHEQUE", id_number=acc, ifsc_code=ifsc, confidence_score=0.85, validation_status="VALID", warnings=[])

# --- ROUTER ---
@app.post("/extract/identity", response_model=IdentityResponse)
async def extract_identity(file: UploadFile = File(...), doc_type: DocType = Form(DocType.AUTO)):
    if not KEY or not ENDPOINT: raise HTTPException(500, "Missing Keys")
    await file.seek(0)
    content = await file.read()
    
    client = DocumentAnalysisClient(ENDPOINT, AzureKeyCredential(KEY))
    
    # FORMS = prebuilt-layout (For Tables)
    model_id = "prebuilt-idDocument"
    if doc_type in [DocType.FORM16, DocType.ITRV, DocType.CHEQUE]:
        model_id = "prebuilt-layout"
        
    print(f"DEBUG: Selected Model: {model_id} for DocType: {doc_type}")
    
    poller = client.begin_analyze_document(model_id, document=content)
    result = poller.result()
    
    strategy = doc_type
    if strategy == DocType.AUTO:
        content_upper = (result.content or "").upper()
        if "FORM 16" in content_upper: strategy = DocType.FORM16
        elif "ITR-V" in content_upper: strategy = DocType.ITRV
        elif "PAY " in content_upper and "RUPEES" in content_upper: strategy = DocType.CHEQUE
        elif "INCOME TAX DEPARTMENT" in content_upper: strategy = DocType.PAN
        else: strategy = DocType.AADHAAR
    
    print(f"DEBUG: Classified as {strategy}")

    if strategy == DocType.AADHAAR: return process_aadhaar(result)
    elif strategy == DocType.PAN: return process_pan(result)
    elif strategy == DocType.CHEQUE: return process_cheque(result)
    elif strategy == DocType.FORM16: return process_form16(result)
    elif strategy == DocType.ITRV: return process_itrv(result)
    else: return process_pan(result)

@app.get("/")
async def root():
    return {"message": "Kaaryaa IDP Engine is Online", "info": "Visit /docs to test the API interactively."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)