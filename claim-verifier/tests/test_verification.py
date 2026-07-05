import os
import sys
import csv
import json
import sqlite3
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Ensure sys.path includes project root
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set database env before importing app
os.environ["DATABASE_PATH"] = "test_claims_audit.db"

import verify_claims
from app.main import app, get_db_conn, init_db

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_test_db():
    # Remove existing test DB if any
    test_db = Path("test_claims_audit.db")
    if test_db.exists():
        test_db.unlink()
    # Initialize DB
    init_db()
    yield
    # Cleanup after test
    if test_db.exists():
        test_db.unlink()


# Helper to build mock Ollama chat response structure
def create_mock_message(json_data):
    return {
        "message": {
            "role": "assistant",
            "content": json.dumps(json_data)
        }
    }


# =========================================================================
# Unit Tests for CSV parsing & Image resolution
# =========================================================================

def test_load_history(tmp_path):
    history_file = tmp_path / "test_history.csv"
    with open(history_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "past_claim_count", "history_flags", "history_summary"])
        writer.writerow(["user_123", "5", "has_past_rejection", "Prior suspicious claim"])

    history = verify_claims.load_history(str(history_file))
    assert "user_123" in history
    assert history["user_123"]["past_claim_count"] == "5"
    assert history["user_123"]["history_flags"] == "has_past_rejection"


def test_load_evidence_requirements(tmp_path):
    reqs_file = tmp_path / "test_reqs.csv"
    with open(reqs_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["requirement_id", "claim_object", "minimum_image_evidence"])
        writer.writerow(["REQ_01", "car", "Bumper must be visible"])

    reqs = verify_claims.load_evidence_requirements(str(reqs_file))
    assert len(reqs) == 1
    assert reqs[0]["requirement_id"] == "REQ_01"
    assert reqs[0]["claim_object"] == "car"


def test_encode_image_missing():
    # Test encoding a file that does not exist
    data, media_type = verify_claims.encode_image("nonexistent_image.jpg")
    assert data is None
    assert media_type is None


def test_image_path_resolution(tmp_path):
    # Create a small valid JPEG file structure
    dummy_img = tmp_path / "test_img.jpg"
    dummy_img.write_bytes(
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00"
        b"\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19"
        b"\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9"
        b"=(343\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00"
        b"\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04"
        b"\x05\x06\x07\x08\t\n\x0b\xff\xca\x00\x11\x08\x00\x01\x00\x01\x03\x01\x11\x00\x02"
        b"\x11\x01\x03\x11\x02\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00\x37\xff\xd9"
    )
    
    data, media_type = verify_claims.encode_image(str(dummy_img))
    assert data is not None
    assert media_type == "image/jpeg"


# =========================================================================
# Test that Zero Images Found -> INSUFFICIENT without calling API
# =========================================================================

@patch("ollama.chat")
def test_zero_images_verdict_insufficient(mock_chat):
    # We call the FastAPI endpoint with NO images uploaded
    response = client.post(
        "/verify-claim",
        data={
            "claim_object": "car",
            "user_claim": "There is a scratch on the side door.",
            "user_id": "user_zero_test"
        }
    )
    
    assert response.status_code == 200
    res_json = response.json()
    
    # Assert result is INSUFFICIENT
    assert res_json["claim_status"] == "INSUFFICIENT"
    assert res_json["evidence_standard_met"] is False
    assert "No images" in res_json["evidence_standard_met_reason"]
    
    # Assert Ollama was NEVER called
    mock_chat.assert_not_called()

    # Assert record is written to SQLite
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM claims WHERE user_id = 'user_zero_test'")
    row = cursor.fetchone()
    conn.close()
    
    assert row is not None
    assert row["claim_status"] == "INSUFFICIENT"
    assert row["claim_object"] == "car"


# =========================================================================
# Verdict Specific Mocked API Tests (SUPPORTED, CONTRADICTED, INSUFFICIENT)
# =========================================================================

@patch("ollama.chat")
def test_verdict_supported(mock_chat, tmp_path):
    # Mocking ollama.chat response for SUPPORTED
    mock_response_payload = {
        "evidence_standard_met": True,
        "evidence_standard_met_reason": "Bumper is fully visible showing clear scraping.",
        "risk_flags": [],
        "issue_type": "scratch",
        "object_part": "front_bumper",
        "claim_status": "SUPPORTED",
        "claim_status_justification": "The image clearly shows scratches on the lower part of the front bumper.",
        "supporting_image_ids": ["img_1"],
        "valid_image": True,
        "severity": "minor"
    }
    mock_chat.return_value = create_mock_message(mock_response_payload)

    # Create dummy image to pass path validation
    img_file = tmp_path / "img_1.png"
    img_file.write_bytes(b"dummy image data")

    # Call endpoint with a file
    with open(img_file, "rb") as img_bytes:
        response = client.post(
            "/verify-claim",
            data={
                "claim_object": "car",
                "user_claim": "Scratch on my bumper.",
                "user_id": "user_supp_test"
            },
            files={"images": ("img_1.png", img_bytes, "image/png")}
        )

    assert response.status_code == 200
    res_json = response.json()

    assert res_json["claim_status"] == "SUPPORTED"
    assert res_json["evidence_standard_met"] is True
    assert res_json["issue_type"] == "scratch"
    assert res_json["object_part"] == "front_bumper"
    assert res_json["severity"] == "minor"
    assert "lower part of the front bumper" in res_json["claim_status_justification"]

    # Verify SQLite record is present
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM claims WHERE user_id = 'user_supp_test'")
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row["claim_status"] == "SUPPORTED"
    assert row["severity"] == "minor"


@patch("ollama.chat")
def test_verdict_contradicted(mock_chat, tmp_path):
    # Mocking ollama.chat response for CONTRADICTED
    mock_response_payload = {
        "evidence_standard_met": True,
        "evidence_standard_met_reason": "Laptop screen is visible and completely intact.",
        "risk_flags": ["high_claim_count"],
        "issue_type": "no_visible_damage",
        "object_part": "screen",
        "claim_status": "CONTRADICTED",
        "claim_status_justification": "The screen is powered on and shows no signs of cracks or shattered glass.",
        "supporting_image_ids": ["img_1"],
        "valid_image": True,
        "severity": "unknown"
    }
    mock_chat.return_value = create_mock_message(mock_response_payload)

    # Create dummy image to pass path validation
    img_file = tmp_path / "img_1.png"
    img_file.write_bytes(b"dummy image data")

    # Call endpoint with a file
    with open(img_file, "rb") as img_bytes:
        response = client.post(
            "/verify-claim",
            data={
                "claim_object": "laptop",
                "user_claim": "Screen is shattered.",
                "user_id": "user_contr_test"
            },
            files={"images": ("img_1.png", img_bytes, "image/png")}
        )

    assert response.status_code == 200
    res_json = response.json()

    assert res_json["claim_status"] == "CONTRADICTED"
    assert res_json["evidence_standard_met"] is True
    assert res_json["issue_type"] == "no_visible_damage"
    assert res_json["object_part"] == "screen"
    assert "high_claim_count" in res_json["risk_flags"]

    # Verify SQLite record is present
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM claims WHERE user_id = 'user_contr_test'")
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row["claim_status"] == "CONTRADICTED"


@patch("ollama.chat")
def test_verdict_insufficient_with_image(mock_chat, tmp_path):
    # Mocking ollama.chat response for INSUFFICIENT
    mock_response_payload = {
        "evidence_standard_met": False,
        "evidence_standard_met_reason": "No laptop or electronics visible in the photo.",
        "risk_flags": [],
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "INSUFFICIENT",
        "claim_status_justification": "The photo only shows a closed cardboard box, making it impossible to assess laptop screen status.",
        "supporting_image_ids": [],
        "valid_image": True,
        "severity": "unknown"
    }
    mock_chat.return_value = create_mock_message(mock_response_payload)

    # Create dummy image to pass path validation
    img_file = tmp_path / "img_1.png"
    img_file.write_bytes(b"dummy image data")

    # Call endpoint with a file
    with open(img_file, "rb") as img_bytes:
        response = client.post(
            "/verify-claim",
            data={
                "claim_object": "laptop",
                "user_claim": "The screen doesn't turn on.",
                "user_id": "user_insuff_test"
            },
            files={"images": ("img_1.png", img_bytes, "image/png")}
        )

    assert response.status_code == 200
    res_json = response.json()

    assert res_json["claim_status"] == "INSUFFICIENT"
    assert res_json["evidence_standard_met"] is False
    assert "impossible to assess laptop" in res_json["claim_status_justification"]

    # Verify SQLite record is present
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM claims WHERE user_id = 'user_insuff_test'")
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row["claim_status"] == "INSUFFICIENT"


# =========================================================================
# Test retrieval endpoints
# =========================================================================

def test_get_claims():
    # Insert some dummy records
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO claims (user_id, claim_object, user_claim, claim_status) 
        VALUES ('user_list_1', 'car', 'some claim', 'SUPPORTED')
    """)
    cursor.execute("""
        INSERT INTO claims (user_id, claim_object, user_claim, claim_status) 
        VALUES ('user_list_2', 'laptop', 'another claim', 'CONTRADICTED')
    """)
    conn.commit()
    conn.close()

    response = client.get("/claims")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 2
    
    # Test filters
    response = client.get("/claims?claim_status=SUPPORTED")
    data = response.json()
    assert len(data) == 1
    assert data[0]["user_id"] == "user_list_1"

    response = client.get("/claims?claim_object=laptop")
    data = response.json()
    assert len(data) == 1
    assert data[0]["user_id"] == "user_list_2"


def test_get_claim_by_id():
    # Insert a dummy record
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO claims (id, user_id, claim_object, user_claim, claim_status) 
        VALUES (999, 'user_999', 'package', 'package wet', 'SUPPORTED')
    """)
    conn.commit()
    conn.close()

    response = client.get("/claims/999")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "user_999"
    assert data["claim_status"] == "SUPPORTED"

    # Not found case
    response = client.get("/claims/9999")
    assert response.status_code == 404
