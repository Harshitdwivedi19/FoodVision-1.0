"""
Unit tests for FoodVision 1.0 Nutrition Service
"""

import pytest
from foodvision.nutrition.service import nutrition_service, NutritionProfile


def test_database_loads_successfully():
    foods = nutrition_service.list_all_foods()
    assert len(foods) > 0
    assert any(f["id"] == "pizza" for f in foods)
    assert any(f["id"] == "sushi" for f in foods)


def test_get_nutrition_pizza():
    profile = nutrition_service.get_nutrition("pizza", portion_multiplier=1.0)
    assert isinstance(profile, NutritionProfile)
    assert profile.name == "Pizza"
    assert profile.calories > 400
    assert profile.macros.protein_g > 15
    assert "Dairy" in profile.allergens or "Gluten" in profile.allergens
    assert profile.exercise_burn.walking_minutes > 50


def test_portion_scaling():
    base = nutrition_service.get_nutrition("hamburger", portion_multiplier=1.0)
    double = nutrition_service.get_nutrition("hamburger", portion_multiplier=2.0)

    assert double.calories == pytest.approx(base.calories * 2.0, rel=1e-1)
    assert double.macros.protein_g == pytest.approx(base.macros.protein_g * 2.0, rel=1e-1)
    assert double.serving_weight_g == pytest.approx(base.serving_weight_g * 2.0, rel=1e-1)


def test_unknown_food_fallback():
    profile = nutrition_service.get_nutrition("unusual_mystery_dish_xyz", portion_multiplier=1.0)
    assert isinstance(profile, NutritionProfile)
    assert profile.calories > 0
    assert profile.macros.protein_g > 0
