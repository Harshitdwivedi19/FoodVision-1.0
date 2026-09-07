"""
FoodVision 1.0 API Routes
Defines REST API endpoints for image classification, nutrition profiling,
and local LLM health diagnosis.
"""

import time
from typing import List, Optional
from fastapi import APIRouter, File, UploadFile, Query, HTTPException, Form
from pydantic import BaseModel
from foodvision.models.classifier import classifier, Prediction
from foodvision.nutrition.service import nutrition_service, NutritionProfile
from foodvision.llm.diagnosis import llm_diagnosis, HealthDiagnosisResponse, ChatMessage
from foodvision.logger import logger

router = APIRouter(prefix="/api")


class ClassificationResponse(BaseModel):
    success: bool
    latency_ms: float
    top_prediction: Prediction
    all_predictions: List[Prediction]
    nutrition: NutritionProfile


class ChatRequest(BaseModel):
    food_id: str
    portion: float = 1.0
    messages: List[ChatMessage]


class ChatResponse(BaseModel):
    reply: str
    food_name: str


class StatusResponse(BaseModel):
    app_name: str = "FoodVision 1.0"
    model_loaded: bool
    model_name: str
    device: str
    total_foods_in_db: int
    llm_provider: str


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Returns runtime status of the FoodVision AI service."""
    ollama_live = await llm_diagnosis.check_ollama_available()
    llm_str = f"Ollama ({llm_diagnosis.model_name})" if ollama_live else "Clinical Nutritionist Engine (Local Fallback)"
    
    return StatusResponse(
        model_loaded=classifier.is_loaded,
        model_name=classifier.model_name,
        device=str(classifier.device),
        total_foods_in_db=len(nutrition_service.list_all_foods()),
        llm_provider=llm_str
    )


@router.post("/classify", response_model=ClassificationResponse)
async def classify_food(
    file: UploadFile = File(...),
    portion: float = Form(1.0)
):
    """
    Receives camera frame or uploaded food image, classifies it using the DL model,
    and attaches detailed nutritional & calorie data.
    """
    t0 = time.perf_counter()
    logger.info("Received image classification request: filename=%s, content_type=%s", file.filename, file.content_type)

    # Read image bytes
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        predictions = classifier.predict(contents)
    except Exception as e:
        logger.error("Classification inference error: %s", e)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    if not predictions:
        raise HTTPException(status_code=500, detail="No predictions returned by model.")

    top_pred = predictions[0]
    nutrition = nutrition_service.get_nutrition(top_pred.food_id, portion_multiplier=portion)

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info("Classified as '%s' (Confidence: %0.2f%%) in %0.2f ms", top_pred.label, top_pred.confidence * 100, latency_ms)

    return ClassificationResponse(
        success=True,
        latency_ms=latency_ms,
        top_prediction=top_pred,
        all_predictions=predictions,
        nutrition=nutrition
    )


@router.get("/nutrition/{food_name}", response_model=NutritionProfile)
async def get_food_nutrition(
    food_name: str,
    portion: float = Query(1.0, ge=0.1, le=5.0)
):
    """Fetches full macronutrient, micronutrient, and calorie data for a given food item."""
    profile = nutrition_service.get_nutrition(food_name, portion_multiplier=portion)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Nutrition data for '{food_name}' not found.")
    return profile


@router.get("/foods")
async def get_all_foods():
    """Lists all available foods in the nutritional knowledge base."""
    return nutrition_service.list_all_foods()


@router.post("/diagnose", response_model=HealthDiagnosisResponse)
async def get_food_diagnosis(
    food_name: str = Query(...),
    portion: float = Query(1.0, ge=0.1, le=5.0)
):
    """
    Triggers local LLM dietary and clinical health diagnosis for the food item and portion.
    """
    profile = nutrition_service.get_nutrition(food_name, portion_multiplier=portion)
    diagnosis = await llm_diagnosis.generate_diagnosis(profile)
    return diagnosis


@router.post("/chat", response_model=ChatResponse)
async def chat_about_food(request: ChatRequest):
    """
    Conversational Q&A with the Local LLM regarding the identified meal.
    """
    profile = nutrition_service.get_nutrition(request.food_id, portion_multiplier=request.portion)
    reply = await llm_diagnosis.chat(profile, request.messages)
    return ChatResponse(
        reply=reply,
        food_name=profile.name
    )
