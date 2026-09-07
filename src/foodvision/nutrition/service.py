"""
FoodVision 1.0 Nutrition Service
Provides nutrition querying, portion scaling, macronutrient calculations,
micronutrient profiling, allergen risk assessment, and exercise burn equivalents.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from foodvision.config import config
from foodvision.logger import logger


class MacroBreakdown(BaseModel):
    protein_g: float
    carbs_g: float
    fat_g: float
    saturated_fat_g: float
    fiber_g: float
    sugar_g: float
    protein_pct: float
    carbs_pct: float
    fat_pct: float


class Micronutrients(BaseModel):
    sodium_mg: float
    potassium_mg: float
    calcium_mg: float
    iron_mg: float
    vitamin_c_mg: float


class ExerciseBurn(BaseModel):
    walking_minutes: int  # ~4 kcal/min
    running_minutes: int  # ~11.5 kcal/min
    cycling_minutes: int  # ~8.5 kcal/min
    swimming_minutes: int # ~10 kcal/min


class NutritionProfile(BaseModel):
    food_id: str
    name: str
    category: str
    portion_multiplier: float = 1.0
    serving_desc: str
    serving_weight_g: float
    calories: float
    macros: MacroBreakdown
    micros: Micronutrients
    glycemic_index: str
    allergens: List[str]
    dietary_flags: List[str]
    description: str
    exercise_burn: ExerciseBurn
    health_score: int  # 0 to 100


class NutritionService:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or config.nutrition_db_path
        self._database: Dict[str, Dict[str, Any]] = {}
        self.load_database()

    def load_database(self) -> None:
        """Loads nutrition data from JSON database file."""
        if not self.db_path.exists():
            logger.warning("Nutrition database not found at %s. Attempting to build it.", self.db_path)
            # Try to build from data/build_nutrition_db.py
            try:
                from data.build_nutrition_db import build_database
                build_database()
            except Exception as e:
                logger.error("Failed to auto-generate nutrition DB: %s", e)

        if self.db_path.exists():
            with open(self.db_path, "r", encoding="utf-8") as f:
                self._database = json.load(f)
            logger.info("Loaded nutrition database with %d items.", len(self._database))
        else:
            logger.error("Nutrition database still unavailable at %s.", self.db_path)
            self._database = {}

    def normalize_food_name(self, name: str) -> str:
        """Normalizes food name to standard database key format."""
        return name.lower().strip().replace(" ", "_").replace("-", "_")

    def get_nutrition(self, food_name: str, portion_multiplier: float = 1.0) -> Optional[NutritionProfile]:
        """
        Retrieves and calculates complete nutritional profile scaled to portion size.
        """
        key = self.normalize_food_name(food_name)
        
        # Direct lookup or fuzzy match
        item = self._database.get(key)
        if not item:
            # Fallback search for partial match
            for k, v in self._database.items():
                if k in key or key in k:
                    item = v
                    key = k
                    break

        if not item:
            logger.warning("Food item '%s' not found in nutrition database.", food_name)
            # Return a generalized healthy fallback profile
            item = {
                "name": food_name.replace("_", " ").title(),
                "category": "Prepared Dish",
                "serving_size_g": 200,
                "serving_desc": "1 standard portion (200g)",
                "calories": 350,
                "protein_g": 15.0,
                "carbs_g": 35.0,
                "fat_g": 12.0,
                "saturated_fat_g": 3.5,
                "fiber_g": 3.0,
                "sugar_g": 4.0,
                "sodium_mg": 500,
                "potassium_mg": 300,
                "calcium_mg": 80,
                "iron_mg": 2.0,
                "vitamin_c_mg": 5.0,
                "glycemic_index": "Medium",
                "allergens": [],
                "dietary_flags": ["General Food"],
                "description": f"Standard nutritional estimate for {food_name.replace('_', ' ').title()}."
            }

        mult = max(0.1, float(portion_multiplier))
        cals = round(item.get("calories", 0) * mult, 1)
        protein = round(item.get("protein_g", 0.0) * mult, 1)
        carbs = round(item.get("carbs_g", 0.0) * mult, 1)
        fat = round(item.get("fat_g", 0.0) * mult, 1)
        sat_fat = round(item.get("saturated_fat_g", 0.0) * mult, 1)
        fiber = round(item.get("fiber_g", 0.0) * mult, 1)
        sugar = round(item.get("sugar_g", 0.0) * mult, 1)

        # Calculate macro percentages (4 kcal/g protein, 4 kcal/g carb, 9 kcal/g fat)
        total_macro_cals = (protein * 4) + (carbs * 4) + (fat * 9)
        if total_macro_cals > 0:
            prot_pct = round((protein * 4 / total_macro_cals) * 100, 1)
            carb_pct = round((carbs * 4 / total_macro_cals) * 100, 1)
            fat_pct = round((fat * 9 / total_macro_cals) * 100, 1)
        else:
            prot_pct, carb_pct, fat_pct = 0.0, 0.0, 0.0

        # Exercise calculations (kcal / burning rate)
        walking_mins = int(cals / 4.5)
        running_mins = int(cals / 11.5)
        cycling_mins = int(cals / 8.5)
        swimming_mins = int(cals / 10.0)

        # Health score calculation (0 to 100)
        # Based on nutrient density: rewards protein and fiber, penalizes sugar, sat fat, sodium
        score = 70.0
        score += min(15, protein * 0.5)
        score += min(15, fiber * 2.0)
        score -= min(20, (sugar / 10.0) * 5.0)
        score -= min(15, (sat_fat / 5.0) * 4.0)
        if item.get("sodium_mg", 0) > 800:
            score -= 10
        health_score = max(10, min(99, int(score)))

        return NutritionProfile(
            food_id=key,
            name=item.get("name", food_name.replace("_", " ").title()),
            category=item.get("category", "General"),
            portion_multiplier=mult,
            serving_desc=item.get("serving_desc", "1 serving"),
            serving_weight_g=round(item.get("serving_size_g", 150) * mult, 1),
            calories=cals,
            macros=MacroBreakdown(
                protein_g=protein,
                carbs_g=carbs,
                fat_g=fat,
                saturated_fat_g=sat_fat,
                fiber_g=fiber,
                sugar_g=sugar,
                protein_pct=prot_pct,
                carbs_pct=carb_pct,
                fat_pct=fat_pct
            ),
            micros=Micronutrients(
                sodium_mg=round(item.get("sodium_mg", 0.0) * mult, 1),
                potassium_mg=round(item.get("potassium_mg", 0.0) * mult, 1),
                calcium_mg=round(item.get("calcium_mg", 0.0) * mult, 1),
                iron_mg=round(item.get("iron_mg", 0.0) * mult, 1),
                vitamin_c_mg=round(item.get("vitamin_c_mg", 0.0) * mult, 1)
            ),
            glycemic_index=item.get("glycemic_index", "Medium"),
            allergens=item.get("allergens", []),
            dietary_flags=item.get("dietary_flags", []),
            description=item.get("description", ""),
            exercise_burn=ExerciseBurn(
                walking_minutes=walking_mins,
                running_minutes=running_mins,
                cycling_minutes=cycling_mins,
                swimming_minutes=swimming_mins
            ),
            health_score=health_score
        )

    def list_all_foods(self) -> List[Dict[str, Any]]:
        """Returns summary of all foods in the database."""
        return [
            {
                "id": k,
                "name": v.get("name", k),
                "category": v.get("category", ""),
                "calories": v.get("calories", 0)
            }
            for k, v in self._database.items()
        ]


nutrition_service = NutritionService()
