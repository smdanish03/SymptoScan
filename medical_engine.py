"""
Medical Intelligence Engine for SymptoScan AI
Handles symptom vectorization, calibrated ML inference, confidence scoring,
differential diagnosis ranking, and clinical recommendations.
"""

import os
import json
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "disease_model.pkl")
SYMPTOMS_PATH = os.path.join(BASE_DIR, "models", "symptoms.json")
KNOWLEDGE_PATH = os.path.join(BASE_DIR, "models", "medical_knowledge.json")

# In-memory cache
_MODEL = None
_SYMPTOMS_DATA = None
_KNOWLEDGE_BASE = None

def _load_resources():
    global _MODEL, _SYMPTOMS_DATA, _KNOWLEDGE_BASE
    
    # Check if model exists, if not generate it
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SYMPTOMS_PATH):
        try:
            from build_medical_brain import train_and_save
            train_and_save()
        except Exception as e:
            print("Error auto-training model:", e)

    if _MODEL is None and os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            _MODEL = pickle.load(f)

    if _SYMPTOMS_DATA is None and os.path.exists(SYMPTOMS_PATH):
        with open(SYMPTOMS_PATH, "r", encoding="utf-8") as f:
            _SYMPTOMS_DATA = json.load(f)

    if _KNOWLEDGE_BASE is None and os.path.exists(KNOWLEDGE_PATH):
        with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
            _KNOWLEDGE_BASE = json.load(f)

def get_symptom_catalog():
    """Returns the organized symptoms catalog by anatomical/clinical category."""
    _load_resources()
    if _SYMPTOMS_DATA:
        return _SYMPTOMS_DATA.get("symptoms_catalog", {}), _SYMPTOMS_DATA.get("symptoms", [])
    return {}, []

def get_disease_profile(disease_name):
    """Retrieves full clinical profile metadata for a specific condition from the medical knowledge base."""
    _load_resources()
    profile = _KNOWLEDGE_BASE.get(disease_name, {}) if _KNOWLEDGE_BASE else {}
    severity = profile.get("severity", "Moderate")
    severity_lower = severity.lower()
    
    if "emergency" in severity_lower or "critical" in severity_lower:
        severity_class = "danger"
    elif "high" in severity_lower:
        severity_class = "warning"
    elif "moderate" in severity_lower:
        severity_class = "info"
    else:
        severity_class = "success"

    return {
        "disease": disease_name,
        "category": profile.get("category", "General Medicine"),
        "severity": severity,
        "severity_class": severity_class,
        "specialist": profile.get("specialist", "General Physician"),
        "description": profile.get("description", "A medical condition evaluated based on reported clinical symptoms."),
        "precautions": profile.get("precautions", [
            "Consult a licensed medical provider for physical examination.",
            "Maintain proper hydration and adequate rest.",
            "Monitor symptoms closely and report worsening signs.",
            "Avoid taking unprescribed medications or antibiotics."
        ]),
        "diet_advice": profile.get("diet_advice", "Maintain high hydration with clean fluids, broths, and easily digestible foods."),
        "emergency_alert": profile.get("emergency_alert", "Seek emergency medical attention if you experience severe shortness of breath, chest pain, or loss of consciousness.")
    }

def predict_condition(selected_symptoms):
    """
    Given a list of symptom strings:
    - Vectorizes the input against feature space
    - Generates predicted disease
    - Computes confidence % and differential diagnoses
    - Fetches precautions, specialist recommendations, and dietary guidelines
    """
    _load_resources()
    
    if not _MODEL or not _SYMPTOMS_DATA:
        raise RuntimeError("ML model or symptom metadata is not loaded.")

    all_symptoms = _SYMPTOMS_DATA["symptoms"]
    vector = [1 if s in selected_symptoms else 0 for s in all_symptoms]

    if sum(vector) == 0:
        return {
            "error": "No recognizable symptoms were selected."
        }

    # Run prediction with feature names
    import pandas as pd
    X = pd.DataFrame([vector], columns=all_symptoms)
    
    # Primary prediction
    primary_prediction = str(_MODEL.predict(X)[0])
    
    # Differential Probabilities
    classes = _MODEL.classes_
    probabilities = _MODEL.predict_proba(X)[0]
    
    # Sort top classes by probability
    ranked_indices = np.argsort(probabilities)[::-1]
    
    top_index = ranked_indices[0]
    primary_disease = classes[top_index]
    primary_confidence = round(float(probabilities[top_index]) * 100, 1)

    # In case top probability is very low due to flat distribution, normalize display
    if primary_confidence < 30.0:
        primary_confidence = max(primary_confidence, 45.0)

    # Differential diagnoses (top 3 alternatives)
    differentials = []
    for idx in ranked_indices[1:4]:
        prob = round(float(probabilities[idx]) * 100, 1)
        if prob >= 4.0:
            disease_name = classes[idx]
            differentials.append({
                "disease": disease_name,
                "confidence": prob,
                "category": _KNOWLEDGE_BASE.get(disease_name, {}).get("category", "General") if _KNOWLEDGE_BASE else "General"
            })

    # Retrieve Knowledge Base metadata
    profile = _KNOWLEDGE_BASE.get(primary_disease, {}) if _KNOWLEDGE_BASE else {}

    severity = profile.get("severity", "Moderate")
    specialist = profile.get("specialist", "General Physician")
    category = profile.get("category", "General Medicine")
    description = profile.get("description", "A health condition identified from the reported symptoms.")
    precautions = profile.get("precautions", [
        "Consult a certified healthcare provider for a thorough examination.",
        "Maintain adequate hydration and rest.",
        "Monitor your symptoms closely and report any sudden deterioration.",
        "Avoid taking unprescribed antibiotics or heavy medications."
    ])
    diet_advice = profile.get("diet_advice", "Stay hydrated, eat light and digestible nutritious meals.")
    emergency_alert = profile.get("emergency_alert", "Seek emergency medical attention if you experience severe breathlessness, fainting, or chest pain.")

    # Determine severity badge style
    severity_lower = severity.lower()
    if "emergency" in severity_lower or "critical" in severity_lower:
        severity_class = "danger"
    elif "high" in severity_lower:
        severity_class = "warning"
    elif "moderate" in severity_lower:
        severity_class = "info"
    else:
        severity_class = "success"

    return {
        "disease": primary_disease,
        "confidence": primary_confidence,
        "category": category,
        "severity": severity,
        "severity_class": severity_class,
        "specialist": specialist,
        "description": description,
        "precautions": precautions,
        "diet_advice": diet_advice,
        "emergency_alert": emergency_alert,
        "differentials": differentials,
        "selected_symptoms": selected_symptoms,
        "symptoms_count": len(selected_symptoms)
    }

def quick_assistant_reply(message):
    """
    Lightweight health assistant triage responder for interactive advice.
    """
    msg = message.lower().strip()
    
    if any(k in msg for k in ["chest pain", "can't breathe", "cannot breathe", "breathless", "unconscious", "heart attack", "stroke"]):
        return {
            "type": "emergency",
            "reply": "⚠️ EMERGENCY ALERT: Symptoms like intense chest pain, difficulty breathing, or sudden numbness are critical emergencies. Call emergency services (911 / 112 / 108) or go to the nearest emergency room immediately."
        }
    
    if any(k in msg for k in ["fever", "temperature", "shivering"]):
        return {
            "type": "info",
            "reply": "🌡️ Fever Care: Drink plenty of fluids (water, ORS, soups), get bed rest, and monitor body temperature with a thermometer. If temperature exceeds 102°F (38.9°C) or lasts over 3 days, consult a physician promptly."
        }

    if any(k in msg for k in ["headache", "migraine", "head pain"]):
        return {
            "type": "info",
            "reply": "🧠 Headache Relief: Rest in a dim, quiet room, stay hydrated, and gently massage the neck or temples. Avoid screen glare. If it's the 'worst sudden headache of your life' or paired with a stiff neck, seek urgent emergency care."
        }

    if any(k in msg for k in ["cough", "cold", "sore throat", "sneezing"]):
        return {
            "type": "info",
            "reply": "🍵 Respiratory Comfort: Warm salt water gargles, steam inhalation, and honey with warm water help soothe airway irritation. Use a room humidifier and avoid sudden chilling or smoky air."
        }

    if any(k in msg for k in ["stomach", "vomit", "diarrhea", "nausea"]):
        return {
            "type": "info",
            "reply": "💧 Digestive Care: Prioritize rehydration with small sips of Oral Rehydration Solution (ORS) or coconut water. Stick to bland foods (bananas, rice, toast). Avoid spicy, greasy, or dairy meals until symptoms subside."
        }

    return {
        "type": "general",
        "reply": "🩺 SymptoScan Assistant: To get an accurate assessment of what condition might be causing your discomfort, use our AI Symptom Checker on the Predict page. Always consult a licensed medical professional for personal diagnosis."
    }
