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

# --- DATA MODELS (FIXED) ---
class DocType(str, Enum):
    AUTO = "auto"
    PAN = "pan"
    AADHAAR = "aadhaar"
    CHEQUE = "cheque"
    FORM16 = "form16"
    ITRV = "itrv"

class IdentityResponse(BaseModel):
    document_type: str
    # We add '= None' to all optional fields so Pydantic doesn't crash if we skip them
    id_number: str | None = None
    full_name: str | None = None
    gender: str | None = None
    date_of_birth: str | None = None  # <--- FIXED: Now defaults to None
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

# --- INCOME HELPERS ---
def process_itrv(result: AnalyzeResult) -> IdentityResponse:
    print("DEBUG: Using ITR-V Strategy")
    content = result.content or ""
    
    ay_match = re.search(r"Assessment\s*Year\s*[:\-]?\s*(\d{4}-\d{2})", content, re.IGNORECASE)
    ay = ay_match.group(1) if ay_match else None

    income_match = re.search(r"Gross\s*Total\s*Income[\s\S]*?(\d{1,3}(?:,\d{2,3})*)", content, re.IGNORECASE)
    income = income_match.group(1) if income_match else None
    
    tax_match = re.search(r"(?:Total\s*Tax\s*Payable|Refund)\s*[\s\S]*?(\d{1,3}(?:,\d{2,3})*)", content, re.IGNORECASE)
    tax = tax_match.group(1) if tax_match else None

    pan_match = re.search(r"[A-Z]{5}[0-9]{4}[A-Z]{1}", content)
    pan = pan_match.group(0) if pan_match else None
    
    name_match = re.search(r"Name[:\s]*([A-Z\s\.]+)", content)
    name = name_match.group(1).strip() if name_match else None

    warnings = []
    if not income: warnings.append("Gross Income not found")
    if not ay: warnings.append("Assessment Year not found")

    return IdentityResponse(
        document_type="ITR-V",
        id_number=pan,
        full_name=name,
        assessment_year=ay,
        gross_income=income,
        tax_paid=tax,
        confidence_score=0.9,
        validation_status="VALID" if not warnings else "REVIEW_NEEDED",
        warnings=warnings
    )

def process_form16(result: AnalyzeResult) -> IdentityResponse:
    print("DEBUG: Using Form 16 Strategy V11")
    content = result.content or ""
    
    # 1. Employer Name (The "Sandwich" Strategy)
    # We look for text BETWEEN "Name and address of the Employer..." AND "Name and address of the Employee..."
    employer = None
    emp_start = re.search(r"Name\s*and\s*address\s*of\s*the\s*Employer/?(?:Specified\s*Bank)?", content, re.IGNORECASE)
    emp_end = re.search(r"Name\s*and\s*address\s*of\s*the\s*Employee", content, re.IGNORECASE)
    
    if emp_start and emp_end:
        # Grab everything between the two headers
        raw_emp_text = content[emp_start.end():emp_end.start()]
        # Clean up newlines and extra spaces
        lines = [line.strip() for line in raw_emp_text.split('\n') if line.strip()]
        # The first non-empty line usually contains the company name
        # We join the first two lines just in case the name wraps (e.g., "STANDARD CHARTERED \n GBS")
        if lines:
            employer = " ".join(lines[:2]) 

    # Fallback if the "Sandwich" fails
    if not employer:
        emp_match = re.search(r"Name\s*and\s*address\s*of\s*Employer[\s\S]*?\n([A-Za-z\s\.,&]+)", content, re.IGNORECASE)
        employer = emp_match.group(1).strip() if emp_match else None

    # 2. Assessment Year (e.g., 2024-25)
    ay_match = re.search(r"Assessment\s*Year\s*[:\-]?\s*(\d{4}-\d{2})", content, re.IGNORECASE)
    ay = ay_match.group(1) if ay_match else None

    # 3. Gross Salary (Specific Phrase from Part B)
    # Look for "Total amount of salary received from current employer" -> any characters -> Number
    salary_match = re.search(r"Total\s*amount\s*of\s*salary\s*received\s*from\s*current\s*employer[\s\S]{0,50}?(\d{1,3}(?:,\d{2,3})*(?:\.\d{2})?)", content, re.IGNORECASE)
    gross = salary_match.group(1) if salary_match else None

    # 4. Tax Payable (Look for "Net tax payable")
    # Using 'Net tax payable' as per your document, or fallback to 'Total Tax Payable'
    tax_match = re.search(r"Net\s*tax\s*payable[\s\S]{0,50}?(\d{1,3}(?:,\d{2,3})*(?:\.\d{2})?)", content, re.IGNORECASE)
    if not tax_match:
        tax_match = re.search(r"Total\s*Tax\s*Payable[\s\S]{0,50}?(\d{1,3}(?:,\d{2,3})*(?:\.\d{2})?)", content, re.IGNORECASE)
    tax = tax_match.group(1) if tax_match else None
    
    warnings = []
    if not gross: warnings.append("Gross Salary not found")
    if not employer: warnings.append("Employer Name not found")
    
    return IdentityResponse(
        document_type="FORM_16",
        employer_name=employer,
        assessment_year=ay,
        gross_income=gross,
        tax_paid=tax,
        confidence_score=0.85,
        validation_status="VALID" if not warnings else "REVIEW_NEEDED",
        warnings=warnings
    )

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
    if not result.documents: return IdentityResponse(document_type="UNKNOWN", id_number=None, full_name=None, confidence_score=0, validation_status="INVALID")
    doc = result.documents[0]
    fields = doc.fields
    id_num = fields.get("DocumentNumber").value if fields.get("DocumentNumber") else None
    f = fields.get("FirstName").value if fields.get("FirstName") else ""
    l = fields.get("LastName").value if fields.get("LastName") else ""
    
    warnings = []
    if not id_num: warnings.append("PAN Missing")
    
    return IdentityResponse(document_type="PAN_CARD", id_number=id_num, full_name=f"{f} {l}".strip(), gender=extract_gender_fallback(result.content), date_of_birth=str(fields.get("DateOfBirth").value) if fields.get("DateOfBirth") else None, confidence_score=doc.confidence, validation_status="VALID" if not warnings else "INVALID", warnings=warnings)

def process_cheque(result: AnalyzeResult) -> IdentityResponse:
    content = result.content
    ifsc = extract_ifsc(content)
    acc = extract_account_number(content)
    warnings = []
    if not ifsc: warnings.append("IFSC Missing")
    if not acc: warnings.append("Account Number Missing")
    
    return IdentityResponse(document_type="CHEQUE", id_number=acc, full_name=None, ifsc_code=ifsc, micr_code=None, bank_name="Unknown Bank", confidence_score=0.85, validation_status="VALID" if not warnings else "REVIEW_NEEDED", warnings=warnings)

# --- CLASSIFIER ---
def classify_document(result: AnalyzeResult) -> DocType:
    content = (result.content or "").upper()
    
    if "FORM 16" in content or "FORM NO. 16" in content: return DocType.FORM16
    if "ITR-V" in content or "INDIAN INCOME TAX RETURN" in content: return DocType.ITRV
    if "PAY " in content and "RUPEES" in content and re.search(r"[A-Z]{4}0[A-Z0-9]{6}", content): return DocType.CHEQUE
    if "INCOME TAX DEPARTMENT" in content: return DocType.PAN
    if "UNIQUE IDENTIFICATION" in content or "AADHAAR" in content: return DocType.AADHAAR
    
    return DocType.PAN 

# --- ROUTER ---
@app.post("/extract/identity", response_model=IdentityResponse)
async def extract_identity(file: UploadFile = File(...), doc_type: DocType = Form(DocType.AUTO)):
    if not KEY or not ENDPOINT: raise HTTPException(500, "Missing Keys")
    await file.seek(0)
    content = await file.read()
    
    client = DocumentAnalysisClient(ENDPOINT, AzureKeyCredential(KEY))
    poller = client.begin_analyze_document("prebuilt-idDocument", document=content)
    result = poller.result()
    
    strategy = doc_type
    if strategy == DocType.AUTO: strategy = classify_document(result)
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