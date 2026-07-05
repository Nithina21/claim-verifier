import os
import sys
import uuid
import shutil
import sqlite3
import csv
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to sys.path to import verify_claims.py
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import verify_claims

app = FastAPI(title="Insurance Claim Verification API (Local Ollama)")

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = os.environ.get("DATABASE_PATH", "claims_audit.db")
UPLOADS_ROOT = Path("images/uploads")
UPLOADS_ROOT.mkdir(parents=True, exist_ok=True)

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

def get_db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = dict_factory
    return conn

# Database Initialization
def init_db():
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS claims (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            claim_object TEXT,
            user_claim TEXT,
            image_paths TEXT,
            evidence_standard_met BOOLEAN,
            evidence_standard_met_reason TEXT,
            risk_flags TEXT,
            issue_type TEXT,
            object_part TEXT,
            claim_status TEXT,
            claim_status_justification TEXT,
            supporting_image_ids TEXT,
            valid_image BOOLEAN,
            severity TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()


def save_verdict(record: dict) -> int:
    conn = get_db_conn()
    cursor = conn.cursor()
    columns = [
        "user_id", "image_paths", "user_claim", "claim_object", 
        "evidence_standard_met", "evidence_standard_met_reason", 
        "risk_flags", "issue_type", "object_part", "claim_status", 
        "claim_status_justification", "supporting_image_ids", 
        "valid_image", "severity"
    ]
    query = f"""
        INSERT INTO claims ({", ".join(columns)})
        VALUES ({", ".join(["?" for _ in columns])})
    """
    values = [
        record.get("user_id"),
        record.get("image_paths"),
        record.get("user_claim"),
        record.get("claim_object"),
        1 if record.get("evidence_standard_met") else 0,
        record.get("evidence_standard_met_reason"),
        record.get("risk_flags"),
        record.get("issue_type"),
        record.get("object_part"),
        record.get("claim_status"),
        record.get("claim_status_justification"),
        record.get("supporting_image_ids"),
        1 if record.get("valid_image") else 0,
        record.get("severity")
    ]
    cursor.execute(query, values)
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


@app.post("/verify-claim")
async def verify_claim(
    claim_object: str = Form(...),
    user_claim: str = Form(...),
    user_id: Optional[str] = Form(None),
    images: Optional[List[UploadFile]] = File(None)
):
    # Standardize user_id
    if not user_id or not user_id.strip():
        user_id = f"web_{uuid.uuid4().hex[:6]}"

    # Validate claim_object
    if claim_object not in ("car", "laptop", "package"):
        raise HTTPException(status_code=400, detail="claim_object must be one of: car, laptop, package")

    # Load history & evidence requirements (from their default locations)
    history = verify_claims.load_history("claims/user_history.csv")
    evidence_reqs = verify_claims.load_evidence_requirements("claims/evidence_requirements.csv")
    history_row = history.get(user_id)

    # 1. Handle uploaded images
    saved_paths = []
    # Clean images list from any empty UploadFiles
    valid_uploads = [img for img in images if img and img.filename] if images else []

    if not valid_uploads:
        # Zero images found -> mark INSUFFICIENT, do NOT call the local Ollama API
        record = {
            "user_id": user_id,
            "image_paths": "",
            "user_claim": user_claim,
            "claim_object": claim_object,
            "evidence_standard_met": False,
            "evidence_standard_met_reason": "No images were uploaded.",
            "risk_flags": history_row.get("history_flags") if history_row else "none",
            "issue_type": "unknown",
            "object_part": "unknown",
            "claim_status": "INSUFFICIENT",
            "claim_status_justification": "No accessible image evidence for this claim.",
            "supporting_image_ids": "",
            "valid_image": False,
            "severity": "unknown",
        }
        db_id = save_verdict(record)
        record["id"] = db_id
        return record

    # Process and save images
    claim_uuid = uuid.uuid4().hex
    claim_upload_dir = UPLOADS_ROOT / claim_uuid
    claim_upload_dir.mkdir(parents=True, exist_ok=True)

    for img in valid_uploads:
        # Ensure safe filename
        safe_filename = Path(img.filename).name
        dest_path = claim_upload_dir / safe_filename
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(img.file, buffer)
        
        # Relative path from project root
        rel_path = f"images/uploads/{claim_uuid}/{safe_filename}"
        saved_paths.append(rel_path)

    # Semicolon-separated path string
    image_paths_str = ";".join(saved_paths)

    row = {
        "user_id": user_id,
        "image_paths": image_paths_str,
        "user_claim": user_claim,
        "claim_object": claim_object,
    }

    # 2. Build user message and resolve image paths
    intro_text, loaded_image_ids, missing_images, loaded_image_paths = verify_claims.build_user_message(
        row, images_root=".", history_row=history_row, evidence_reqs=evidence_reqs
    )

    # Double check if any image was successfully loaded/decoded
    if not loaded_image_ids:
        # Zero images successfully loaded -> mark INSUFFICIENT, do NOT call the local API
        record = {
            "user_id": user_id,
            "image_paths": image_paths_str,
            "user_claim": user_claim,
            "claim_object": claim_object,
            "evidence_standard_met": False,
            "evidence_standard_met_reason": f"None of the uploaded images could be resolved on disk: {image_paths_str}",
            "risk_flags": history_row.get("history_flags") if history_row else "none",
            "issue_type": "unknown",
            "object_part": "unknown",
            "claim_status": "INSUFFICIENT",
            "claim_status_justification": "No accessible image evidence for this claim.",
            "supporting_image_ids": "",
            "valid_image": False,
            "severity": "unknown",
        }
        db_id = save_verdict(record)
        record["id"] = db_id
        return record

    # 3. Call local Ollama API
    model = os.environ.get("OLLAMA_MODEL", verify_claims.DEFAULT_MODEL)
    raw_result = verify_claims.call_ollama(model, verify_claims.SYSTEM_PROMPT, intro_text, loaded_image_paths)
    normalized = verify_claims.normalize_result(raw_result, row, loaded_image_ids, missing_images)

    # Save to DB
    db_id = save_verdict(normalized)
    normalized["id"] = db_id

    return normalized


@app.post("/verify-batch")
async def verify_batch(
    csv_path: str = Form(...),
    images_folder: str = Form(".")
):
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"CSV file not found: {csv_path}")

    # Load history & evidence requirements
    history = verify_claims.load_history("claims/user_history.csv")
    evidence_reqs = verify_claims.load_evidence_requirements("claims/evidence_requirements.csv")

    model = os.environ.get("OLLAMA_MODEL", verify_claims.DEFAULT_MODEL)

    # Read the input CSV
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {str(e)}")

    results = []
    for row in rows:
        user_id = row.get("user_id", "unknown")
        user_claim = row.get("user_claim", "")
        claim_object = row.get("claim_object", "")
        image_paths_raw = row.get("image_paths", "")

        history_row = history.get(user_id)

        # Build path verification context
        intro_text, loaded_image_ids, missing_images, loaded_image_paths = verify_claims.build_user_message(
            row, images_folder, history_row, evidence_reqs
        )

        if not loaded_image_ids:
            # Zero images found -> mark INSUFFICIENT, do NOT call the local API
            normalized = {
                "user_id": user_id,
                "image_paths": image_paths_raw,
                "user_claim": user_claim,
                "claim_object": claim_object,
                "evidence_standard_met": False,
                "evidence_standard_met_reason": f"None of the referenced images could be loaded from disk: {image_paths_raw}",
                "risk_flags": history_row.get("history_flags") if history_row else "none",
                "issue_type": "unknown",
                "object_part": "unknown",
                "claim_status": "INSUFFICIENT",
                "claim_status_justification": "No accessible image evidence for this claim.",
                "supporting_image_ids": "",
                "valid_image": False,
                "severity": "unknown",
            }
        else:
            raw_result = verify_claims.call_ollama(model, verify_claims.SYSTEM_PROMPT, intro_text, loaded_image_paths)
            normalized = verify_claims.normalize_result(raw_result, row, loaded_image_ids, missing_images)

        # Save to SQLite
        db_id = save_verdict(normalized)
        normalized["id"] = db_id
        results.append(normalized)

    # Generate Output CSV file
    out_csv_path = "output_batch.csv"
    try:
        with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=verify_claims.OUTPUT_FIELDS)
            writer.writeheader()
            for r in results:
                filtered_row = {k: r.get(k) for k in verify_claims.OUTPUT_FIELDS}
                writer.writerow(filtered_row)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate output CSV file: {str(e)}")

    return FileResponse(
        path=out_csv_path,
        filename="batch_results.csv",
        media_type="text/csv"
    )


@app.get("/claims")
async def get_claims(
    claim_status: Optional[str] = Query(None),
    claim_object: Optional[str] = Query(None)
):
    conn = get_db_conn()
    cursor = conn.cursor()
    
    query = "SELECT * FROM claims"
    conditions = []
    params = []
    
    if claim_status:
        conditions.append("claim_status = ?")
        params.append(claim_status)
    if claim_object:
        conditions.append("claim_object = ?")
        params.append(claim_object)
        
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
        
    query += " ORDER BY id DESC"
    
    cursor.execute(query, params)
    claims = cursor.fetchall()
    conn.close()
    return claims


@app.get("/claims/{claim_id}")
async def get_claim(claim_id: int):
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM claims WHERE id = ?", (claim_id,))
    claim = cursor.fetchone()
    conn.close()
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
    return claim


# Serve static web files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Serve project images (test & uploads) so web page can show thumbnails
app.mount("/images", StaticFiles(directory="images"), name="images")

@app.get("/model")
async def get_model():
    model = os.environ.get("OLLAMA_MODEL", verify_claims.DEFAULT_MODEL)
    return {"model": model}

@app.get("/")
async def root():
    return FileResponse("app/static/index.html")

