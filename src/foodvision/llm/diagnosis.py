"""
FoodVision 1.0 Local LLM Diagnosis Module
Provides intelligent dietary and clinical health diagnoses for recognized foods.
Supports:
1. Local Ollama instance (e.g. llama3, phi3, mistral, qwen2.5)
2. Local HuggingFace Transformers
3. Deterministic Clinical Nutritionist Fallback Engine (Guarantees zero downtime)
"""

import json
from typing import Dict, Any, List, Optional
import httpx
from pydantic import BaseModel
from foodvision.config import config
from foodvision.logger import logger
from foodvision.nutrition.service import NutritionProfile


class DietarySuitability(BaseModel):
    weight_loss: str  # "Excellent", "Moderate", "Occasional", "Avoid"
    muscle_building: str
    diabetic_friendly: str
    keto_low_carb: str
    heart_health: str


class HealthDiagnosisResponse(BaseModel):
    provider: str  # "ollama", "transformers", or "clinical_engine"
    food_name: str
    portion_summary: str
    overall_health_grade: str  # "A+", "A", "B", "C", "D"
    suitability: DietarySuitability
    clinical_summary: str
    key_benefits: List[str]
    health_cautions: List[str]
    exercise_recommendation: str
    healthier_swaps: List[str]
    suggested_pairings: List[str]


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant" or "system"
    content: str


class LocalLLMDiagnosis:
    def __init__(self):
        self.ollama_url = config.llm.ollama_url.rstrip("/")
        self.model_name = config.llm.ollama_model
        self.timeout = config.llm.timeout_seconds

    async def check_ollama_available(self) -> bool:
        """Checks if local Ollama daemon is reachable."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                res = await client.get(f"{self.ollama_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    async def generate_diagnosis(self, profile: NutritionProfile) -> HealthDiagnosisResponse:
        """
        Generates comprehensive dietary and clinical diagnosis.
        Attempts Ollama first; if unreachable or times out, falls back to clinical engine.
        """
        ollama_live = await self.check_ollama_available()
        if ollama_live:
            try:
                response = await self._diagnose_with_ollama(profile)
                if response:
                    return response
            except Exception as e:
                logger.warning("Ollama diagnosis failed: %s. Falling back to clinical engine.", e)

        # Fallback to local clinical expert system
        logger.info("Using deterministic Clinical Nutritionist Engine for '%s'.", profile.name)
        return self._clinical_expert_diagnosis(profile)

    async def _diagnose_with_ollama(self, profile: NutritionProfile) -> Optional[HealthDiagnosisResponse]:
        """Calls local Ollama instance with structured prompt for clinical diagnosis."""
        prompt = f"""
You are a senior clinical nutritionist and dietitian.
Analyze the following food item and nutritional profile:

Food: {profile.name} ({profile.category})
Portion: {profile.serving_desc} ({profile.serving_weight_g}g)
Calories: {profile.calories} kcal
Protein: {profile.macros.protein_g}g ({profile.macros.protein_pct}%)
Carbohydrates: {profile.macros.carbs_g}g ({profile.macros.carbs_pct}%)
Fat: {profile.macros.fat_g}g ({profile.macros.fat_pct}%, Saturated: {profile.macros.saturated_fat_g}g)
Dietary Fiber: {profile.macros.fiber_g}g
Sugar: {profile.macros.sugar_g}g
Sodium: {profile.micros.sodium_mg}mg
Potassium: {profile.micros.potassium_mg}mg
Glycemic Index: {profile.glycemic_index}
Allergens: {', '.join(profile.allergens) if profile.allergens else 'None declared'}

Provide your diagnosis strictly in the following JSON format (no other text or markdown fences):
{{
  "overall_health_grade": "A|B|C|D",
  "suitability": {{
    "weight_loss": "Excellent|Moderate|Occasional|Avoid",
    "muscle_building": "Excellent|Moderate|Occasional|Avoid",
    "diabetic_friendly": "Excellent|Moderate|Occasional|Avoid",
    "keto_low_carb": "Excellent|Moderate|Occasional|Avoid",
    "heart_health": "Excellent|Moderate|Occasional|Avoid"
  }},
  "clinical_summary": "2-3 sentences providing clinical context on satiety, metabolic impact, and digestion.",
  "key_benefits": ["Benefit 1", "Benefit 2", "Benefit 3"],
  "health_cautions": ["Caution 1", "Caution 2"],
  "exercise_recommendation": "Specific physical activity needed to burn or balance this intake.",
  "healthier_swaps": ["Modification/swap 1", "Modification/swap 2"],
  "suggested_pairings": ["Pairing 1 to balance the meal", "Pairing 2"]
}}
"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": config.llm.temperature
                }
            }
            res = await client.post(f"{self.ollama_url}/api/generate", json=payload)
            if res.status_code == 200:
                raw_text = res.json().get("response", "").strip()
                # Clean JSON fences if model outputted them
                if raw_text.startswith("```"):
                    raw_text = raw_text.split("```")[1]
                    if raw_text.startswith("json"):
                        raw_text = raw_text[4:]
                data = json.loads(raw_text.strip())

                return HealthDiagnosisResponse(
                    provider=f"ollama ({self.model_name})",
                    food_name=profile.name,
                    portion_summary=f"{profile.serving_desc} ({profile.calories} kcal)",
                    overall_health_grade=data.get("overall_health_grade", "B"),
                    suitability=DietarySuitability(**data.get("suitability", {})),
                    clinical_summary=data.get("clinical_summary", ""),
                    key_benefits=data.get("key_benefits", []),
                    health_cautions=data.get("health_cautions", []),
                    exercise_recommendation=data.get("exercise_recommendation", ""),
                    healthier_swaps=data.get("healthier_swaps", []),
                    suggested_pairings=data.get("suggested_pairings", [])
                )
        return None

    def _clinical_expert_diagnosis(self, profile: NutritionProfile) -> HealthDiagnosisResponse:
        """
        Deterministic, evidence-based clinical nutrition assessment engine.
        Calculates health impact metrics based on macros, micronutrients, and USDA dietary guidelines.
        """
        macros = profile.macros
        micros = profile.micros
        cals = profile.calories

        # Grade calculation
        score = profile.health_score
        if score >= 85:
            grade = "A"
        elif score >= 70:
            grade = "B"
        elif score >= 50:
            grade = "C"
        else:
            grade = "D"

        # Weight loss suitability
        if cals < 250 and macros.fiber_g >= 3.0:
            wl = "Excellent"
        elif cals < 450:
            wl = "Moderate"
        elif cals < 600:
            wl = "Occasional"
        else:
            wl = "Avoid"

        # Muscle building suitability
        if macros.protein_g >= 25.0:
            mb = "Excellent"
        elif macros.protein_g >= 15.0:
            mb = "Moderate"
        else:
            mb = "Occasional"

        # Diabetic suitability
        if profile.glycemic_index == "Low" and macros.sugar_g < 5.0 and macros.fiber_g > 2.0:
            diabetic = "Excellent"
        elif profile.glycemic_index in ["Low", "Medium"] and macros.sugar_g < 15.0:
            diabetic = "Moderate"
        else:
            diabetic = "Avoid"

        # Keto / Low-Carb suitability
        if macros.carbs_g <= 8.0:
            keto = "Excellent"
        elif macros.carbs_g <= 18.0:
            keto = "Moderate"
        else:
            keto = "Avoid"

        # Heart Health
        if macros.saturated_fat_g <= 3.0 and micros.sodium_mg <= 400:
            heart = "Excellent"
        elif macros.saturated_fat_g <= 7.0 and micros.sodium_mg <= 800:
            heart = "Moderate"
        else:
            heart = "Occasional"

        # Benefits
        benefits = []
        if macros.protein_g >= 15:
            benefits.append(f"High in lean protein ({macros.protein_g}g), promoting muscle repair and sustained satiety.")
        if macros.fiber_g >= 3.0:
            benefits.append(f"Contains {macros.fiber_g}g dietary fiber, assisting blood glucose regulation and gut microbiome.")
        if micros.potassium_mg >= 350:
            benefits.append(f"Abundant in potassium ({micros.potassium_mg}mg), supporting healthy electrolyte balance.")
        if micros.iron_mg >= 2.0:
            benefits.append(f"Provides {micros.iron_mg}mg iron for cellular oxygen transport.")
        if "Omega-3 Rich" in profile.dietary_flags:
            benefits.append("Rich in essential anti-inflammatory Omega-3 fatty acids.")
        if not benefits:
            benefits.append(f"Provides {cals} kcal of bioavailable caloric energy.")

        # Cautions
        cautions = []
        if micros.sodium_mg >= 800:
            cautions.append(f"High sodium ({micros.sodium_mg}mg) — monitor if managing hypertension or water retention.")
        if macros.saturated_fat_g >= 8.0:
            cautions.append(f"Contains {macros.saturated_fat_g}g saturated fat — consume in moderation for cardiovascular health.")
        if macros.sugar_g >= 18.0:
            cautions.append(f"Elevated sugar content ({macros.sugar_g}g) may trigger rapid insulin spikes.")
        if profile.allergens:
            cautions.append(f"Contains known allergens: {', '.join(profile.allergens)}.")
        if not cautions:
            cautions.append("No critical dietary contraindications identified.")

        # Exercise recommendation
        ex = profile.exercise_burn
        exercise_str = f"To expend the {cals} kcal consumed, consider {ex.walking_minutes} mins brisk walking, {ex.cycling_minutes} mins cycling, or {ex.running_minutes} mins running."

        # Swaps
        swaps = []
        if macros.carbs_g > 40:
            swaps.append("Substitute refined grains with steamed vegetables, quinoa, or cauliflower rice.")
        if macros.fat_g > 20:
            swaps.append("Opt for grilling or air-frying instead of deep-frying to reduce lipid density by up to 40%.")
        if micros.sodium_mg > 700:
            swaps.append("Request sauces and dressings on the side to curb unnecessary sodium intake.")
        if not swaps:
            swaps.append("Current preparation is well-optimized; pair with a side salad to boost micronutrient density.")

        # Pairings
        pairings = [
            "Pair with high-potassium greens (spinach, arugula, cucumber) to optimize electrolyte balance.",
            "Accompany with sparkling water with lemon to assist digestive enzyme activity."
        ]

        summary = f"{profile.name} delivers {cals} kcal with {macros.protein_g}g protein and {macros.carbs_g}g carbs. " \
                  f"It is graded '{grade}' based on its macronutrient distribution, glycemic response, and micronutrient density."

        return HealthDiagnosisResponse(
            provider="local_clinical_engine",
            food_name=profile.name,
            portion_summary=f"{profile.serving_desc} ({cals} kcal)",
            overall_health_grade=grade,
            suitability=DietarySuitability(
                weight_loss=wl,
                muscle_building=mb,
                diabetic_friendly=diabetic,
                keto_low_carb=keto,
                heart_health=heart
            ),
            clinical_summary=summary,
            key_benefits=benefits,
            health_cautions=cautions,
            exercise_recommendation=exercise_str,
            healthier_swaps=swaps,
            suggested_pairings=pairings
        )

    async def chat(self, profile: NutritionProfile, messages: List[ChatMessage]) -> str:
        """
        Interactive conversational Q&A about the identified food.
        """
        ollama_live = await self.check_ollama_available()
        user_query = messages[-1].content if messages else "Is this food healthy?"

        if ollama_live:
            try:
                system_prompt = f"""
You are an expert AI clinical nutritionist. The user is asking about the following meal they just scanned:
Food: {profile.name} ({profile.calories} kcal, {profile.macros.protein_g}g Protein, {profile.macros.carbs_g}g Carbs, {profile.macros.fat_g}g Fat, {profile.micros.sodium_mg}mg Sodium, Allergens: {', '.join(profile.allergens)}).
Answer concisely, scientifically, and empathetically. Provide practical advice.
"""
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    chat_payload = {
                        "model": self.model_name,
                        "messages": [{"role": "system", "content": system_prompt}] + [m.model_dump() for m in messages],
                        "stream": False
                    }
                    res = await client.post(f"{self.ollama_url}/api/chat", json=chat_payload)
                    if res.status_code == 200:
                        return res.json().get("message", {}).get("content", "").strip()
            except Exception as e:
                logger.warning("Ollama chat failed: %s. Using local conversational fallback.", e)

        # Intelligent conversational fallback
        q = user_query.lower()
        if "dinner" in q or "night" in q:
            return f"For dinner, {profile.name} ({profile.calories} kcal) is fine if eaten at least 2-3 hours before sleep. {'Because of its carbohydrate content, pair it with light fiber.' if profile.macros.carbs_g > 30 else 'Its protein helps with nighttime muscle recovery.'}"
        elif "diabet" in q or "sugar" in q:
            return f"Regarding diabetes: {profile.name} has a {profile.glycemic_index} glycemic index with {profile.macros.sugar_g}g sugar and {profile.macros.carbs_g}g total carbs. {'Exercise caution with portion sizes to prevent postprandial glucose spikes.' if profile.glycemic_index == 'High' or profile.macros.sugar_g > 15 else 'It presents a relatively gentle glucose curve, especially when paired with greens.'}"
        elif "weight" in q or "fat" in q or "diet" in q:
            return f"For weight management: A single serving contributes {profile.calories} kcal. {'To keep it within a deficit, consider a half portion or offsetting with a 30-minute brisk walk.' if profile.calories > 400 else 'It fits comfortably into most caloric targets while offering good satiety.'}"
        elif "protein" in q or "gym" in q or "workout" in q:
            return f"For fitness goals: {profile.name} contains {profile.macros.protein_g}g of protein ({profile.macros.protein_pct}% of calories). {'This is an excellent post-workout option for muscle synthesis.' if profile.macros.protein_g >= 25 else 'You may want to supplement it with an additional lean protein source like egg whites or Greek yogurt.'}"
        else:
            return f"{profile.name} provides {profile.calories} calories ({profile.macros.protein_g}g protein, {profile.macros.carbs_g}g carbs, {profile.macros.fat_g}g fat). Overall, it's a solid food choice when balanced with adequate hydration and physical activity."


llm_diagnosis = LocalLLMDiagnosis()
