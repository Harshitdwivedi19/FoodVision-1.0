"""
Unit tests for FoodVision 1.0 Local LLM Diagnosis Module
"""

import pytest
from foodvision.nutrition.service import nutrition_service
from foodvision.llm.diagnosis import llm_diagnosis, HealthDiagnosisResponse, ChatMessage


@pytest.mark.asyncio
async def test_clinical_expert_diagnosis():
    profile = nutrition_service.get_nutrition("grilled_salmon", portion_multiplier=1.0)
    diagnosis = await llm_diagnosis.generate_diagnosis(profile)

    assert isinstance(diagnosis, HealthDiagnosisResponse)
    assert diagnosis.food_name == "Grilled Salmon"
    assert diagnosis.overall_health_grade in ["A", "B", "C", "D"]
    assert diagnosis.suitability.muscle_building in ["Excellent", "Moderate"]
    assert len(diagnosis.key_benefits) > 0
    assert len(diagnosis.healthier_swaps) > 0


@pytest.mark.asyncio
async def test_conversational_chat():
    profile = nutrition_service.get_nutrition("pizza", portion_multiplier=1.0)
    messages = [
        ChatMessage(role="user", content="Can I eat this for late dinner?")
    ]
    reply = await llm_diagnosis.chat(profile, messages)
    assert isinstance(reply, str)
    assert len(reply) > 20
    assert "dinner" in reply.lower() or "pizza" in reply.lower()
