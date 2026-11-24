"""
Comprehensive test script for AgroTech End-to-End System.
Verifies all 6 core features using in-memory TestClient (zero network port requirements).
"""
import sys
import pathlib
import io

sys.path.insert(0, "backend")
sys.path.insert(0, ".")

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

print("=" * 60)
print("TEST 1: Health Endpoint")
print("=" * 60)
r = client.get("/health")
print("Status code:", r.status_code)
print("Response:", r.json())
assert r.status_code == 200, f"Expected 200, got {r.status_code}"

print("\n" + "=" * 60)
print("TEST 2: Plugin Status & Quick Actions")
print("=" * 60)
r = client.get("/api/v1/plugin/raithanna/status")
print("Status:", r.status_code, r.json().get("name"))
assert r.status_code == 200

r = client.get("/api/v1/plugin/raithanna/quick-actions")
print("Quick actions count:", len(r.json().get("actions", [])))
assert r.status_code == 200

# Load test assets
user_chilli_path = pathlib.Path("data_pipeline/vision_training_data/Chilli___healthy_mature/user_chilli_real.jpg")
chilli_bytes = user_chilli_path.read_bytes()

tomato_bytes = pathlib.Path("scratch/user_tomato_crop.jpg").read_bytes() if pathlib.Path("scratch/user_tomato_crop.jpg").exists() else chilli_bytes
leaf_bytes = pathlib.Path("scratch/user_leaf_crop.jpg").read_bytes() if pathlib.Path("scratch/user_leaf_crop.jpg").exists() else chilli_bytes

soil_path = list(pathlib.Path("data_pipeline/vision_training_data/Soil___black_cotton").glob("*.jpg"))[0]
soil_bytes = soil_path.read_bytes()

pest_path = list(pathlib.Path("data_pipeline/vision_training_data/Pest___fall_armyworm").glob("*.jpg"))[0]
pest_bytes = pest_path.read_bytes()

print("\n" + "=" * 60)
print("TEST 3: Feature 1 - Identify Plant Species (Real Chilli Image)")
print("=" * 60)
r = client.post(
    "/api/v1/plugin/raithanna/feature-action",
    data={"feature_id": "identify_plant", "lang": "en", "district": "Davanagere"},
    files={"file": ("chilli_sample.jpg", io.BytesIO(chilli_bytes), "image/jpeg")}
)
print("Status:", r.status_code)
res = r.json()
print("Reply Preview:\n", res["reply"])
assert r.status_code == 200
assert "chilli" in res["reply"].lower() or "capsicum" in res["reply"].lower()
assert "Likely identification" in res["reply"]
assert "Scientific name: Capsicum annuum" in res["reply"]
assert "Visible characteristics:" in res["reply"]

print("\n" + "=" * 60)
print("TEST 4: Feature 2 - Detect Disease (Real Chilli Image - MUST BE HEALTHY MATURE)")
print("=" * 60)
r = client.post(
    "/api/v1/plugin/raithanna/feature-action",
    data={"feature_id": "detect_disease", "lang": "kn", "district": "Davanagere"},
    files={"file": ("chilli_sample.jpg", io.BytesIO(chilli_bytes), "image/jpeg")}
)
print("Status:", r.status_code)
res = r.json()
print("Reply Preview:\n", res["reply"])
assert r.status_code == 200
assert "Fall Armyworm" not in res["reply"]
assert "ಸೈನಿಕ ಹುಳು" not in res["reply"]
assert "ಆರೋಗ್ಯಕರ" in res["reply"] or "Healthy" in res["reply"]

print("\n" + "=" * 60)
print("TEST 4b: Feature 2 - Detect Disease (Real Tomato Image - MUST BE HEALTHY TOMATO)")
print("=" * 60)
r = client.post(
    "/api/v1/plugin/raithanna/feature-action",
    data={"feature_id": "detect_disease", "lang": "en", "district": "Davanagere"},
    files={"file": ("tomato_sample.jpg", io.BytesIO(tomato_bytes), "image/jpeg")}
)
print("Status:", r.status_code)
res = r.json()
print("Reply Preview:\n", res["reply"])
assert r.status_code == 200
assert "Healthy Tomato" in res["reply"]
assert "Chilli" not in res["reply"]

print("\n" + "=" * 60)
print("TEST 4c: Feature 2 - Detect Disease (Real Diseased Leaf - MUST BE BLIGHT / LEAF SPOT)")
print("=" * 60)
r = client.post(
    "/api/v1/plugin/raithanna/feature-action",
    data={"feature_id": "detect_disease", "lang": "en", "district": "Davanagere"},
    files={"file": ("leaf_sample.jpg", io.BytesIO(leaf_bytes), "image/jpeg")}
)
print("Status:", r.status_code)
res = r.json()
print("Reply Preview:\n", res["reply"])
assert r.status_code == 200
assert "Blight" in res["reply"] or "Spot" in res["reply"] or "Curl" in res["reply"]
assert "Healthy Chilli Plant" not in res["reply"]

print("\n" + "=" * 60)
print("TEST 5: Feature 3 - Analyse Soil (Black Cotton Soil Image)")
print("=" * 60)
r = client.post(
    "/api/v1/plugin/raithanna/feature-action",
    data={"feature_id": "analyse_soil", "lang": "kn", "district": "Davanagere"},
    files={"file": ("soil_sample.jpg", io.BytesIO(soil_bytes), "image/jpeg")}
)
print("Status:", r.status_code)
res = r.json()
print("Reply Preview:\n", res["reply"])
assert r.status_code == 200
assert "ಕಪ್ಪು ಮಣ್ಣು" in res["reply"] or "Black Cotton" in res["reply"]

print("\n" + "=" * 60)
print("TEST 6: Feature 3b - Analyse Soil (Lab Test Card)")
print("=" * 60)
r = client.post(
    "/api/v1/plugin/raithanna/feature-action",
    data={
        "feature_id": "analyse_soil",
        "lang": "en",
        "district": "Davanagere",
        "soil_ph": "6.8",
        "soil_n": "210.0",
        "soil_p": "18.0",
        "soil_k": "190.0",
        "soil_oc": "0.45"
    }
)
print("Status:", r.status_code)
res = r.json()
print("Reply Preview:\n", res["reply"][:250])
assert r.status_code == 200
assert "Soil Health Card Interpretation" in res["reply"]

print("\n" + "=" * 60)
print("TEST 7: Feature 4 - Identify Pests (Fall Armyworm Image)")
print("=" * 60)
r = client.post(
    "/api/v1/plugin/raithanna/feature-action",
    data={"feature_id": "identify_pests", "lang": "kn", "district": "Davanagere"},
    files={"file": ("pest_sample.jpg", io.BytesIO(pest_bytes), "image/jpeg")}
)
print("Status:", r.status_code)
res = r.json()
print("Reply Preview:\n", res["reply"])
assert r.status_code == 200
assert "ಸೈನಿಕ ಹುಳು" in res["reply"] or "Fall Armyworm" in res["reply"]

print("\n" + "=" * 60)
print("TEST 8: Feature 5 - Weather Advice")
print("=" * 60)
r = client.post(
    "/api/v1/plugin/raithanna/feature-action",
    data={"feature_id": "weather_advice", "lang": "kn", "district": "Davanagere"}
)
print("Status:", r.status_code)
res = r.json()
print("Reply Preview:\n", res["reply"][:200])
assert r.status_code == 200
assert "ಹವಾಮಾನ" in res["reply"]

print("\n" + "=" * 60)
print("TEST 9: Feature 6 - Ask Agriculture AI (Kannada Chat)")
print("=" * 60)
queries = [
    ("ಯೋಜನೆಗಳು", "ಸರ್ಕಾರಿ ಯೋಜನೆ"),
    ("ಮಾರುಕಟ್ಟೆ ದರ", "ಎಪಿಎಂಸಿ"),
    ("ಮೆಕ್ಕೆಜೋಳ ಬೆಳೆ ಸಲಹೆ", "ಮೆಕ್ಕೆಜೋಳ")
]
for q, expected in queries:
    r = client.post(
        "/api/v1/plugin/raithanna/feature-action",
        data={"feature_id": "ask_ai", "message": q, "lang": "kn", "district": "Davanagere"}
    )
    print(f"Query: '{q}' -> Status:", r.status_code)
    assert r.status_code == 200
    assert len(r.json()["reply"]) > 50

print("\n" + "=" * 60)
print("🎉 ALL 9 COMPREHENSIVE END-TO-END TESTS PASSED SUCCESSFULLY!")
print("=" * 60)
