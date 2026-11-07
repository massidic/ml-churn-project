import os
import urllib.request

import pytest
from selenium import webdriver

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

# Test E2E : exclu par défaut (pytest.ini), lancé avec "pytest -m e2e"
pytestmark = pytest.mark.e2e


def api_disponible():
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def test_application():
    if not api_disponible():
        pytest.skip(f"API non joignable sur {BASE_URL} : lancez-la avant ce test")

    driver = webdriver.Chrome()
    try:
        driver.get(f"{BASE_URL}/docs")
        # Le titre réel est "Churn Prediction API - Swagger UI"
        assert "Swagger UI" in driver.title
        assert "Churn Prediction API" in driver.title
    finally:
        driver.quit()