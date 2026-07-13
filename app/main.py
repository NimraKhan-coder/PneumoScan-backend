"""
FastAPI backend for the Pneumonia Detection FYP.

Endpoints:
    GET  /               -> health check
    POST /predict         -> upload an X-ray image, get prediction + explanation

Run locally with:
    uvicorn app.main:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for the interactive Swagger UI
to test uploads directly from the browser.
"""

from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.utils.model_utils import predict_pneumonia
from app.utils.gemini_gatekeeper import is_chest_xray_gemini
from app.utils.explanation import get_explanation_and_recommendation, DISCLAIMER
from app.routers.auth import router as auth_router
from app.routers.admin import router as admin_router

app = FastAPI(
    title="Pneumonia Detection API",
    description="AI-powered pneumonia detection from chest X-ray images.",
    version="1.0.0",
)


app.include_router(auth_router)
app.include_router(admin_router)


# Allow the React Native app (and Expo dev server) to call this API.
# Tighten allow_origins to your actual domain/app scheme before production deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}
MAX_FILE_SIZE_MB = 10


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Pneumonia Detection API is running."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # --- Validate file type ---
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Please upload a JPEG or PNG image.",
        )

    # --- Read and validate file size ---
    image_bytes = await file.read()
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f} MB). Maximum allowed is {MAX_FILE_SIZE_MB} MB.",
        )

    # --- Validate this actually looks like a chest X-ray (Gemini vision check) ---
    gemini_result = is_chest_xray_gemini(image_bytes, file.content_type)

    if gemini_result is None:
        raise HTTPException(
            status_code=503,
            detail="Image validation service is temporarily unavailable. Please try again shortly.",
        )

    if not gemini_result:
        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded image doesn't appear to be a chest X-ray. "
                "Please upload a valid chest X-ray image."
            ),
        )

    # --- Run prediction ---
    try:
        result = predict_pneumonia(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # --- Generate explanation + recommendation ---
    explanation, recommendation, risk_level = get_explanation_and_recommendation(
        result["label"], result["confidence"]
    )

    return {
        "prediction": result["label"],
        "confidence": result["confidence"],
        "raw_probability": result["raw_probability"],
        "risk_level": risk_level,
        "explanation": explanation,
        "recommendation": recommendation,
        "disclaimer": DISCLAIMER,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
