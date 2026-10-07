import json
import os
import random
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------------------------------------
# 1. DEFINE SYMPTOMS BY CATEGORY
# -------------------------------------------------------------
SYMPTOMS_CATALOG = {
    "General & Systemic": [
        "High Fever",
        "Mild Fever",
        "Chills & Shivering",
        "Fatigue & Weakness",
        "Body Aches & Muscle Pain",
        "Joint Pain & Swelling",
        "Excessive Sweating",
        "Unexplained Weight Loss",
    ],
    "Respiratory & ENT": [
        "Dry Cough",
        "Cough with Mucus",
        "Shortness of Breath",
        "Sore Throat",
        "Runny or Stuffy Nose",
        "Chest Tightness or Pain",
        "Sneezing",
        "Loss of Taste or Smell",
    ],
    "Digestive & Gastrointestinal": [
        "Nausea",
        "Vomiting",
        "Diarrhea",
        "Abdominal Cramps & Pain",
        "Heartburn & Acid Reflux",
        "Loss of Appetite",
        "Abdominal Bloating",
    ],
    "Neurological & Sensory": [
        "Severe Throbbing Headache",
        "Dull or Tension Headache",
        "Dizziness & Lightheadedness",
        "Sensitivity to Light/Sound",
        "Stiff Neck & Confusion",
    ],
    "Skin, Eyes & Urinary": [
        "Itchy Skin Rash or Hives",
        "Red or Watery Eyes",
        "Burning Urination",
        "Frequent Urination",
        "Yellowing of Skin/Eyes (Jaundice)",
    ],
}

# Flattened list of all symptom feature names
ALL_SYMPTOMS = []
for group in SYMPTOMS_CATALOG.values():
    ALL_SYMPTOMS.extend(group)

# -------------------------------------------------------------
# 2. MEDICAL CONDITIONS CLINICAL PROFILES
# -------------------------------------------------------------
DISEASE_PROFILES = {
    "Influenza (Flu)": {
        "category": "Infectious & Viral",
        "severity": "Moderate",
        "specialist": "General Physician",
        "description": "A contagious respiratory viral infection affecting the nose, throat, and sometimes lungs.",
        "core_symptoms": ["High Fever", "Chills & Shivering", "Body Aches & Muscle Pain", "Fatigue & Weakness"],
        "optional_symptoms": ["Dry Cough", "Sore Throat", "Severe Throbbing Headache", "Loss of Appetite"],
        "precautions": [
            "Get plenty of bed rest and sleep to support immune function.",
            "Stay thoroughly hydrated with warm broths, herbal teas, and electrolytes.",
            "Use over-the-counter antipyretics (e.g. Paracetamol) if advised by a doctor.",
            "Isolate at home to prevent spreading the virus to family or colleagues."
        ],
        "diet_advice": "Warm chicken/vegetable soup, citrus fruits high in Vitamin C, herbal ginger tea, and plenty of water.",
        "emergency_alert": "Seek urgent care if you experience severe shortness of breath, chest pain, or high fever that fails to break after 72 hours."
    },
    "Common Cold": {
        "category": "Respiratory",
        "severity": "Mild",
        "specialist": "General Physician / ENT",
        "description": "A mild viral infection of the upper respiratory tract commonly caused by rhinoviruses.",
        "core_symptoms": ["Runny or Stuffy Nose", "Sneezing", "Sore Throat"],
        "optional_symptoms": ["Mild Fever", "Dry Cough", "Fatigue & Weakness", "Dull or Tension Headache"],
        "precautions": [
            "Rest adequately and gargle with warm salt water for throat comfort.",
            "Use steam inhalation or saline nasal sprays to relieve nasal congestion.",
            "Wash hands frequently and avoid sharing utensils.",
            "Avoid cold drinks and exposure to sudden temperature drops."
        ],
        "diet_advice": "Honey-lemon tea, clear soups, warm fluids, and light, easy-to-digest meals.",
        "emergency_alert": "Consult a doctor if symptoms persist past 10 days or worsen into severe wheezing."
    },
    "COVID-19": {
        "category": "Infectious & Respiratory",
        "severity": "Moderate to High",
        "specialist": "Pulmonologist / Infectious Disease Specialist",
        "description": "Coronavirus infection caused by SARS-CoV-2 targeting respiratory pathways and systemic immunity.",
        "core_symptoms": ["High Fever", "Dry Cough", "Fatigue & Weakness", "Loss of Taste or Smell"],
        "optional_symptoms": ["Shortness of Breath", "Body Aches & Muscle Pain", "Sore Throat", "Chest Tightness or Pain"],
        "precautions": [
            "Isolate immediately in a well-ventilated room.",
            "Monitor blood oxygen saturation (SpO2) and body temperature twice daily.",
            "Rest completely and maintain high fluid intake.",
            "Wear an N95 mask if interacting with caregivers."
        ],
        "diet_advice": "Antioxidant-rich foods, zinc and vitamin D rich nutrition, warm herbal fluids.",
        "emergency_alert": "Call emergency services if oxygen saturation (SpO2) drops below 94%, or if lips turn bluish."
    },
    "Acute Bronchitis": {
        "category": "Respiratory",
        "severity": "Moderate",
        "specialist": "Pulmonologist",
        "description": "Inflammation of the bronchial tubes leading to mucus buildup and persistent coughing.",
        "core_symptoms": ["Cough with Mucus", "Chest Tightness or Pain", "Fatigue & Weakness"],
        "optional_symptoms": ["Mild Fever", "Shortness of Breath", "Sore Throat", "Dull or Tension Headache"],
        "precautions": [
            "Avoid cigarette smoke, fumes, and air pollutants completely.",
            "Use a room humidifier or take warm steam showers.",
            "Drink ample fluids to help thin thick mucus.",
            "Get sufficient vocal and physical rest."
        ],
        "diet_advice": "Warm herbal teas, honey, broths, and avoid mucus-producing very cold dairy products.",
        "emergency_alert": "Seek medical help if you cough up blood or experience severe difficulty breathing."
    },
    "Pneumonia": {
        "category": "Respiratory (Urgent)",
        "severity": "High",
        "specialist": "Pulmonologist / Emergency Care",
        "description": "Infection that inflames air sacs in one or both lungs, which may fill with fluid or purulent material.",
        "core_symptoms": ["High Fever", "Chills & Shivering", "Cough with Mucus", "Shortness of Breath"],
        "optional_symptoms": ["Chest Tightness or Pain", "Fatigue & Weakness", "Loss of Appetite", "Excessive Sweating"],
        "precautions": [
            "Consult a physician immediately for diagnostic chest X-ray and prescription antibiotics/antivirals.",
            "Do not delay medical evaluation — pneumonia can escalate rapidly.",
            "Strictly follow prescribed medication courses without premature stoppage.",
            "Rest in an upright or elevated position to facilitate breathing."
        ],
        "diet_advice": "High-protein recovery foods, nutrient-dense soups, oral rehydration solutions.",
        "emergency_alert": "Go to the nearest emergency room if you experience persistent blue lips, extreme breathlessness, or confusion."
    },
    "Asthma Exacerbation": {
        "category": "Chronic Respiratory",
        "severity": "Moderate to High",
        "specialist": "Pulmonologist / Allergist",
        "description": "Airway hyper-reactivity resulting in bronchial constriction, wheezing, and breathing difficulty.",
        "core_symptoms": ["Shortness of Breath", "Chest Tightness or Pain", "Dry Cough"],
        "optional_symptoms": ["Fatigue & Weakness", "Dull or Tension Headache"],
        "precautions": [
            "Use your prescribed rescue bronchodilator inhaler as outlined in your action plan.",
            "Sit upright and stay calm; avoid lying flat.",
            "Eliminate immediate exposure to known allergens, smoke, or pet dander.",
            "Seek emergency attention if rescue inhaler does not produce relief within 15 minutes."
        ],
        "diet_advice": "Anti-inflammatory diet with omega-3 fatty acids, magnesium-rich foods, and fresh vegetables.",
        "emergency_alert": "Emergency medical treatment is required if you struggle to speak in full sentences or your chest feels clamped."
    },
    "Allergic Rhinitis": {
        "category": "Allergy & Immunology",
        "severity": "Mild",
        "specialist": "Allergist / ENT",
        "description": "Allergic response in eyes and nasal passages provoked by airborne allergens like pollen or dust.",
        "core_symptoms": ["Sneezing", "Runny or Stuffy Nose", "Red or Watery Eyes"],
        "optional_symptoms": ["Itchy Skin Rash or Hives", "Sore Throat", "Fatigue & Weakness"],
        "precautions": [
            "Identify and minimize exposure to seasonal pollens and dust mites.",
            "Rinse nasal cavities with sterile saline solution.",
            "Consider antihistamine medication as directed by a healthcare provider.",
            "Keep bedroom windows closed on windy high-pollen days."
        ],
        "diet_advice": "Quercetin-rich foods like apples, berries, onions, and green tea.",
        "emergency_alert": "Seek urgent care if allergic reaction escalates to facial swelling or difficulty swallowing."
    },
    "Sinusitis": {
        "category": "ENT / Respiratory",
        "severity": "Moderate",
        "specialist": "ENT Specialist",
        "description": "Inflammation and swelling of tissue lining the paranasal sinuses causing pressure and drainage issues.",
        "core_symptoms": ["Severe Throbbing Headache", "Runny or Stuffy Nose", "Sore Throat"],
        "optional_symptoms": ["Mild Fever", "Fatigue & Weakness", "Loss of Taste or Smell", "Cough with Mucus"],
        "precautions": [
            "Apply warm moist compresses around the eyes and forehead.",
            "Perform saline nasal irrigation once or twice daily.",
            "Stay well hydrated to encourage sinus drainage.",
            "Avoid chlorinated pools and rapid altitude fluctuations."
        ],
        "diet_advice": "Warm broths, spicy broths with cayenne/garlic to clear nasal passages, hot green tea.",
        "emergency_alert": "Visit a doctor if you experience vision changes, severe swelling around the eyes, or a stiff neck."
    },
    "Strep Throat": {
        "category": "Bacterial ENT",
        "severity": "Moderate to High",
        "specialist": "ENT Specialist / General Physician",
        "description": "Bacterial throat infection caused by Group A Streptococcus requiring antibiotics.",
        "core_symptoms": ["Sore Throat", "High Fever", "Severe Throbbing Headache"],
        "optional_symptoms": ["Fatigue & Weakness", "Loss of Appetite", "Body Aches & Muscle Pain"],
        "precautions": [
            "Obtain a throat swab rapid strep test from a doctor.",
            "Complete the full prescribed course of antibiotics to prevent rheumatic complications.",
            "Replace your toothbrush 48 hours after starting antibiotics.",
            "Gargle with warm salt water several times a day."
        ],
        "diet_advice": "Smoothies, pureed warm soups, yogurt, chilled gelatin, non-acidic soft foods.",
        "emergency_alert": "Seek emergency care if swallowing becomes impossible or breathing is obstructed."
    },
    "Migraine": {
        "category": "Neurological",
        "severity": "Moderate to Severe",
        "specialist": "Neurologist",
        "description": "Recurrent neurological disorder causing intense pulsing or throbbing headaches often with sensory auras.",
        "core_symptoms": ["Severe Throbbing Headache", "Sensitivity to Light/Sound", "Nausea"],
        "optional_symptoms": ["Vomiting", "Dizziness & Lightheadedness", "Fatigue & Weakness"],
        "precautions": [
            "Rest in a completely dark, quiet room with cool eye shades.",
            "Apply an ice pack to the back of the neck or temples.",
            "Maintain consistent sleep and meal schedules to prevent trigger episodes.",
            "Stay hydrated and avoid known dietary triggers like artificial sweeteners, aged cheese, or MSG."
        ],
        "diet_advice": "Magnesium-rich foods (almonds, spinach), ginger tea, and consistent hydration.",
        "emergency_alert": "Seek urgent medical evaluation for the 'worst headache of your life' (thunderclap headache)."
    },
    "Tension Headache": {
        "category": "Neurological",
        "severity": "Mild to Moderate",
        "specialist": "General Physician / Neurologist",
        "description": "Diffuse, mild-to-moderate band-like pain around the head linked to stress, poor posture, or eye strain.",
        "core_symptoms": ["Dull or Tension Headache", "Fatigue & Weakness"],
        "optional_symptoms": ["Sensitivity to Light/Sound", "Body Aches & Muscle Pain"],
        "precautions": [
            "Practice neck stretches, progressive muscle relaxation, and posture correction.",
            "Take scheduled screen breaks following the 20-20-20 rule.",
            "Ensure regular 7-8 hours of quality sleep nightly.",
            "Apply a warm heating pad to tight shoulder muscles."
        ],
        "diet_advice": "Hydrating fluids, herbal chamomile tea, balanced wholesome meals.",
        "emergency_alert": "Consult a doctor if headaches become daily or are accompanied by numbness."
    },
    "Gastroenteritis (Stomach Flu)": {
        "category": "Gastrointestinal",
        "severity": "Moderate",
        "specialist": "Gastroenterologist / Physician",
        "description": "Intestinal infection marked by watery diarrhea, stomach cramps, nausea, and occasional low fever.",
        "core_symptoms": ["Diarrhea", "Vomiting", "Nausea", "Abdominal Cramps & Pain"],
        "optional_symptoms": ["Mild Fever", "Fatigue & Weakness", "Loss of Appetite", "Chills & Shivering"],
        "precautions": [
            "Drink Oral Rehydration Salts (ORS) in small frequent sips to replenish electrolytes.",
            "Avoid dairy, caffeine, alcohol, fatty foods, and heavily spiced dishes.",
            "Rest your stomach for a few hours after vomiting before attempting clear liquids.",
            "Wash hands thoroughly after using the bathroom to avoid transmission."
        ],
        "diet_advice": "BRAT diet (Bananas, Rice, Applesauce, Toast), electrolyte water, light coconut water.",
        "emergency_alert": "Go to the hospital if you cannot keep fluids down for 24 hours or show severe dehydration signs."
    },
    "Food Poisoning": {
        "category": "Gastrointestinal",
        "severity": "Moderate",
        "specialist": "General Physician",
        "description": "Acute foodborne illness triggered by ingesting contaminated food with bacteria, toxins, or parasites.",
        "core_symptoms": ["Vomiting", "Diarrhea", "Abdominal Cramps & Pain", "Nausea"],
        "optional_symptoms": ["High Fever", "Chills & Shivering", "Fatigue & Weakness", "Dizziness & Lightheadedness"],
        "precautions": [
            "Prioritize rapid rehydration with electrolyte solutions.",
            "Do not immediately suppress diarrhea with anti-motility drugs unless physician advises.",
            "Rest quietly and avoid solid heavy foods until vomiting ceases completely.",
            "Discard any suspect food leftovers safely."
        ],
        "diet_advice": "Electrolyte fluids, clear broths, crackers, plain rice once stomach settles.",
        "emergency_alert": "Seek urgent medical attention if you notice bloody stools or high fever over 102°F (38.9°C)."
    },
    "GERD / Acid Reflux": {
        "category": "Gastrointestinal",
        "severity": "Mild to Moderate",
        "specialist": "Gastroenterologist",
        "description": "Chronic digestive disease where stomach acid washes back into the esophagus irritating the lining.",
        "core_symptoms": ["Heartburn & Acid Reflux", "Chest Tightness or Pain", "Abdominal Bloating"],
        "optional_symptoms": ["Nausea", "Sore Throat", "Loss of Appetite"],
        "precautions": [
            "Avoid lying down for at least 3 hours after eating a meal.",
            "Elevate the head of your bed by 6 inches.",
            "Eat smaller, more frequent meals rather than large heavy dinners.",
            "Avoid trigger foods: citrus, tomatoes, chocolate, caffeine, and deep-fried foods."
        ],
        "diet_advice": "Oatmeal, non-citrus fruits (melon, banana), lean poultry, ginger tea, steamed vegetables.",
        "emergency_alert": "Distinguish from cardiac events: if chest pain radiates to your left arm or jaw with shortness of breath, call 911/112."
    },
    "Peptic Ulcer": {
        "category": "Gastrointestinal",
        "severity": "Moderate to High",
        "specialist": "Gastroenterologist",
        "description": "Open sores that develop on the inside lining of the stomach and upper portion of the small intestine.",
        "core_symptoms": ["Abdominal Cramps & Pain", "Heartburn & Acid Reflux", "Loss of Appetite"],
        "optional_symptoms": ["Nausea", "Vomiting", "Abdominal Bloating", "Fatigue & Weakness"],
        "precautions": [
            "Consult a doctor for H. pylori testing and endoscopy evaluation.",
            "Avoid NSAID painkillers (like ibuprofen/aspirin) as they degrade gastric lining.",
            "Quit smoking and strictly avoid alcoholic beverages.",
            "Eat on a regular consistent schedule."
        ],
        "diet_advice": "Bland soothing foods: boiled potatoes, probiotic kefir, steamed greens, bone broth.",
        "emergency_alert": "Immediate emergency room visit required if vomiting coffee-ground material or passing black tarry stools."
    },
    "Appendicitis": {
        "category": "Abdominal Surgical Emergency",
        "severity": "Critical / Emergency",
        "specialist": "General Surgeon / Emergency Department",
        "description": "Acute inflammation of the appendix requiring prompt surgical evaluation to prevent rupture.",
        "core_symptoms": ["Abdominal Cramps & Pain", "Nausea", "Vomiting", "Loss of Appetite"],
        "optional_symptoms": ["High Fever", "Diarrhea", "Fatigue & Weakness"],
        "precautions": [
            "GO TO THE NEAREST EMERGENCY ROOM IMMEDIATELY.",
            "DO NOT eat or drink anything (in case emergency surgery is needed).",
            "DO NOT take laxatives or use heating pads on your abdomen as they may trigger rupture.",
            "DO NOT take pain relievers before a surgeon evaluates the tender area."
        ],
        "diet_advice": "NPO (Nothing by mouth) until evaluated by an emergency medical team.",
        "emergency_alert": "CRITICAL EMERGENCY: If lower right abdomen pain becomes excruciating, immediate surgery is required."
    },
    "Dengue Fever": {
        "category": "Tropical & Vector-Borne",
        "severity": "High",
        "specialist": "Infectious Disease Specialist / Physician",
        "description": "Mosquito-borne viral infection causing high fever, intense joint and muscle pain, and platelet drops.",
        "core_symptoms": ["High Fever", "Body Aches & Muscle Pain", "Joint Pain & Swelling", "Severe Throbbing Headache"],
        "optional_symptoms": ["Itchy Skin Rash or Hives", "Nausea", "Fatigue & Weakness", "Vomiting"],
        "precautions": [
            "Get immediate blood test for Complete Blood Count (CBC) and Dengue NS1/IgM antigen.",
            "Monitor platelet counts daily under medical supervision.",
            "DO NOT take Aspirin or Ibuprofen (they worsen bleeding risk); only use doctor-prescribed Paracetamol.",
            "Drink continuous fluids: papaya leaf extract (if advised), coconut water, ORS."
        ],
        "diet_advice": "Fresh pomegranate, kiwi, coconut water, papaya, well-cooked dal and soups.",
        "emergency_alert": "Go to ER immediately if you observe gum bleeding, nosebleeds, red petechial spots, or persistent abdominal pain."
    },
    "Malaria": {
        "category": "Parasitic Infection",
        "severity": "High",
        "specialist": "Infectious Disease Specialist",
        "description": "Life-threatening disease caused by Plasmodium parasites transmitted through infected Anopheles mosquitoes.",
        "core_symptoms": ["High Fever", "Chills & Shivering", "Excessive Sweating", "Severe Throbbing Headache"],
        "optional_symptoms": ["Body Aches & Muscle Pain", "Fatigue & Weakness", "Nausea", "Vomiting"],
        "precautions": [
            "Obtain urgent peripheral blood smear or rapid malaria test.",
            "Commence doctor-prescribed Artemisinin Combination Therapy (ACT) immediately.",
            "Sleep under insecticide-treated bed nets and use repellent.",
            "Ensure full adherence to prescribed antimalarial dosage."
        ],
        "diet_advice": "High-carbohydrate, easily digestible meals with abundant fluid and electrolyte intake.",
        "emergency_alert": "Immediate hospitalization needed if patient experiences confusion, seizures, or dark urine."
    },
    "Typhoid Fever": {
        "category": "Bacterial Infectious",
        "severity": "High",
        "specialist": "Infectious Disease Specialist / Physician",
        "description": "Bacterial infection caused by Salmonella typhi spreading through contaminated water or food.",
        "core_symptoms": ["High Fever", "Abdominal Cramps & Pain", "Fatigue & Weakness", "Severe Throbbing Headache"],
        "optional_symptoms": ["Loss of Appetite", "Diarrhea", "Body Aches & Muscle Pain", "Excessive Sweating"],
        "precautions": [
            "Consult a doctor for Widal test, blood cultures, and targeted antibiotics.",
            "Only consume boiled, purified water and strictly avoid street food.",
            "Wash hands meticulously before eating and after toilet use.",
            "Do not discontinue antibiotics early even after fever drops."
        ],
        "diet_advice": "Soft bland diet: boiled khichdi, mashed potatoes, bananas, clear broths; avoid high-fiber raw vegetables.",
        "emergency_alert": "Seek urgent hospitalization if severe abdominal swelling, delirium, or intestinal bleeding occurs."
    },
    "Urinary Tract Infection (UTI)": {
        "category": "Urological",
        "severity": "Moderate",
        "specialist": "Urologist / General Physician",
        "description": "Infection in any part of the urinary system, commonly bladder and urethra, typically caused by bacteria.",
        "core_symptoms": ["Burning Urination", "Frequent Urination"],
        "optional_symptoms": ["Abdominal Cramps & Pain", "Mild Fever", "Fatigue & Weakness", "Nausea"],
        "precautions": [
            "Consult a physician for urine routine/culture and targeted antibiotics.",
            "Drink plenty of water to frequently flush out bacteria from urinary pathways.",
            "Do not delay urination when feeling the urge.",
            "Wipe from front to back to prevent bacteria from entering the urethra."
        ],
        "diet_advice": "Pure unsweetened cranberry juice, high water volume, avoid coffee, alcohol, and spicy foods.",
        "emergency_alert": "Urgent doctor review needed if fever becomes high with flank/back pain, signaling kidney involvement."
    },
    "Kidney Stones": {
        "category": "Nephrology / Urology",
        "severity": "High",
        "specialist": "Urologist / Nephrologist",
        "description": "Hard mineral deposits that form inside the kidneys and cause excruciating pain when passing.",
        "core_symptoms": ["Abdominal Cramps & Pain", "Burning Urination", "Frequent Urination"],
        "optional_symptoms": ["Nausea", "Vomiting", "Mild Fever"],
        "precautions": [
            "Consult a urologist for ultrasound/CT KUB scan to determine stone size and location.",
            "Drink 2.5 to 3 liters of water daily to encourage stone passage if small.",
            "Manage acute pain with physician-prescribed antispasmodics.",
            "Catch passed stones in a strainer for laboratory mineral analysis."
        ],
        "diet_advice": "Citrus lemon water (citrate inhibits stones), limit sodium, avoid oxalates (spinach, chocolate, nuts).",
        "emergency_alert": "Seek ER care if unable to urinate at all or if accompanied by chills, fever, and unrelenting severe pain."
    },
    "Hypertension Warning Episode": {
        "category": "Cardiovascular",
        "severity": "High to Emergency",
        "specialist": "Cardiologist / General Physician",
        "description": "Spike in blood pressure exerting pressure on cerebral and cardiovascular vessels.",
        "core_symptoms": ["Severe Throbbing Headache", "Dizziness & Lightheadedness", "Chest Tightness or Pain"],
        "optional_symptoms": ["Shortness of Breath", "Fatigue & Weakness", "Sensitivity to Light/Sound"],
        "precautions": [
            "Measure your blood pressure immediately with a digital monitor.",
            "Sit quietly in a relaxed position; do not panic or exert yourself.",
            "Take prescribed antihypertensive medication if advised by your doctor.",
            "Avoid salt, energy drinks, nicotine, and strenuous physical stress."
        ],
        "diet_advice": "DASH diet: potassium-rich bananas, leafy greens, low sodium foods, hibiscus tea.",
        "emergency_alert": "EMERGENCY: If BP is above 180/120 with chest pain, numbness, or blurred vision, call emergency services immediately."
    },
    "Dehydration & Heat Exhaustion": {
        "category": "Environmental & Systemic",
        "severity": "Moderate",
        "specialist": "General Physician",
        "description": "Depletion of body fluids and electrolytes due to heat, inadequate intake, or heavy physical exertion.",
        "core_symptoms": ["Dizziness & Lightheadedness", "Fatigue & Weakness", "Excessive Sweating"],
        "optional_symptoms": ["Dull or Tension Headache", "Nausea", "Loss of Appetite"],
        "precautions": [
            "Move immediately to an air-conditioned room or cool shaded area.",
            "Sip electrolyte fluids, coconut water, or chilled ORS slowly.",
            "Loosen tight clothing and apply cool damp cloths to skin.",
            "Rest with feet slightly elevated."
        ],
        "diet_advice": "Watermelon, cucumber, electrolyte drinks, coconut water, buttermilk with a pinch of salt.",
        "emergency_alert": "Seek urgent ER care if sweating stops and skin turns hot and dry (signs of heat stroke), or fainting occurs."
    },
    "Iron Deficiency Anemia": {
        "category": "Hematological",
        "severity": "Moderate",
        "specialist": "Hematologist / General Physician",
        "description": "Insufficient healthy red blood cells caused by low iron levels reducing oxygen delivery to body tissues.",
        "core_symptoms": ["Fatigue & Weakness", "Dizziness & Lightheadedness", "Shortness of Breath"],
        "optional_symptoms": ["Dull or Tension Headache", "Excessive Sweating", "Unexplained Weight Loss"],
        "precautions": [
            "Have your serum ferritin and complete blood count (CBC) tested.",
            "Take iron supplements with Vitamin C (enhances absorption) on a doctor's advice.",
            "Avoid drinking tea or coffee within an hour of meals (tannins block iron absorption).",
            "Pace your daily physical activities to prevent exhaustion."
        ],
        "diet_advice": "Spinach, lentils, red meat, beetroot, pomegranate, pumpkin seeds, and citrus fruits.",
        "emergency_alert": "Visit a physician if you experience chest pain, fainting episodes, or irregular fast heartbeats."
    },
    "Allergic Urticaria (Hives)": {
        "category": "Dermatology & Allergy",
        "severity": "Mild to Moderate",
        "specialist": "Dermatologist / Allergist",
        "description": "Itchy, raised red skin welts provoked by histamines in reaction to foods, medications, or insect bites.",
        "core_symptoms": ["Itchy Skin Rash or Hives", "Red or Watery Eyes"],
        "optional_symptoms": ["Sneezing", "Mild Fever", "Fatigue & Weakness"],
        "precautions": [
            "Apply cool compresses or calamine lotion to soothe itching.",
            "Take physician-recommended second-generation antihistamines.",
            "Avoid hot baths, scratching, and rough synthetic clothing.",
            "Keep a log of recently ingested foods, medications, or cosmetic products."
        ],
        "diet_advice": "Fresh fruits, vegetables, lots of water, avoid known allergy triggers and processed foods.",
        "emergency_alert": "IMMEDIATE ER ALERT: If hives are accompanied by swelling of lips, tongue, or difficulty breathing (Anaphylaxis)."
    },
    "Conjunctivitis (Pink Eye)": {
        "category": "Ophthalmology",
        "severity": "Mild",
        "specialist": "Ophthalmologist",
        "description": "Inflammation or infection of the transparent membrane (conjunctiva) lining the eyelid and eyeball.",
        "core_symptoms": ["Red or Watery Eyes", "Sensitivity to Light/Sound"],
        "optional_symptoms": ["Dull or Tension Headache", "Sneezing", "Runny or Stuffy Nose"],
        "precautions": [
            "Avoid touching or rubbing your eyes; wash hands frequently.",
            "Apply cool or warm sterile damp compresses for comfort.",
            "Do not wear contact lenses or eye makeup until full clearance.",
            "Do not share pillows, towels, or eyedrops with anyone."
        ],
        "diet_advice": "Beta-carotene rich carrots, leafy vegetables, citrus fruits, and adequate hydration.",
        "emergency_alert": "See an eye specialist urgently if experiencing intense eye pain, severe vision loss, or cornea clouding."
    },
    "Jaundice / Hepatitis Alert": {
        "category": "Hepatic / Gastrointestinal",
        "severity": "High",
        "specialist": "Hepatologist / Gastroenterologist",
        "description": "Liver dysfunction leading to buildup of bilirubin in blood, causing yellowing of skin and eyes.",
        "core_symptoms": ["Yellowing of Skin/Eyes (Jaundice)", "Fatigue & Weakness", "Loss of Appetite"],
        "optional_symptoms": ["Nausea", "Abdominal Cramps & Pain", "Mild Fever", "Vomiting"],
        "precautions": [
            "Undergo liver function tests (LFT) and viral hepatitis panel immediately.",
            "Completely abstain from alcohol and unnecessary over-the-counter medications.",
            "Ensure complete physical rest to assist liver regeneration.",
            "Drink only boiled, safe drinking water."
        ],
        "diet_advice": "Sugarcane juice (freshly hygiene-prepared), radish, barley water, papaya, boiled low-fat foods.",
        "emergency_alert": "Hospitalization is critical if patient displays mental confusion, lethargy, or severe bleeding."
    },
    "Meningitis Alert": {
        "category": "Neurological Emergency",
        "severity": "Critical / Emergency",
        "specialist": "Neurologist / Emergency Medicine",
        "description": "Serious inflammation of protective membranes covering the brain and spinal cord.",
        "core_symptoms": ["High Fever", "Stiff Neck & Confusion", "Severe Throbbing Headache"],
        "optional_symptoms": ["Sensitivity to Light/Sound", "Vomiting", "Nausea", "Fatigue & Weakness"],
        "precautions": [
            "CALL EMERGENCY SERVICES (911/112/108) OR GO TO ER IMMEDIATELY.",
            "Meningitis is a medical emergency that can deteriorate within hours.",
            "Lumbar puncture and urgent intravenous antibiotics/antivirals are lifesaving.",
            "Avoid self-medication."
        ],
        "diet_advice": "Medical emergency: IV hospital administration only.",
        "emergency_alert": "CRITICAL EMERGENCY: Stiff neck + high fever + altered mental state requires immediate hospitalization."
    },
    "Viral Fever & Body Fatigue": {
        "category": "General Medicine",
        "severity": "Mild to Moderate",
        "specialist": "General Physician",
        "description": "General systemic viral infection causing febrile episodes, myalgia, and generalized fatigue.",
        "core_symptoms": ["High Fever", "Body Aches & Muscle Pain", "Fatigue & Weakness", "Dull or Tension Headache"],
        "optional_symptoms": ["Chills & Shivering", "Loss of Appetite", "Mild Fever"],
        "precautions": [
            "Adequate bed rest for 3 to 5 days until fever subsides.",
            "Use lukewarm sponge baths if temperature climbs above 101°F.",
            "Stay hydrated with coconut water, lemonade, and clear broths.",
            "Take antipyretics only as advised by your healthcare provider."
        ],
        "diet_advice": "Light khichdi, porridge, fresh seasonal fruits, warm milk with turmeric at night.",
        "emergency_alert": "Consult a doctor if high fever continues beyond 4 days or if rash and breathing problems appear."
    }
}

# -------------------------------------------------------------
# 3. SYNTHETIC DATASET GENERATION
# -------------------------------------------------------------
def generate_dataset(samples_per_disease=60):
    rows = []
    random.seed(42)
    np.random.seed(42)

    for disease, profile in DISEASE_PROFILES.items():
        core = profile["core_symptoms"]
        optional = profile["optional_symptoms"]

        for _ in range(samples_per_disease):
            row = {symptom: 0 for symptom in ALL_SYMPTOMS}

            # Core symptoms: present with 85-98% probability
            for s in core:
                if s in row and random.random() < 0.93:
                    row[s] = 1

            # Ensure at least 2 core symptoms are always active
            active_core = [s for s in core if row.get(s, 0) == 1]
            if len(active_core) < 2 and len(core) >= 2:
                for s in random.sample(core, min(2, len(core))):
                    row[s] = 1

            # Optional symptoms: present with 20-50% probability
            for s in optional:
                if s in row and random.random() < 0.35:
                    row[s] = 1

            # Small random noise (5% chance of 1 unrelated symptom)
            if random.random() < 0.08:
                unrelated = [s for s in ALL_SYMPTOMS if s not in core and s not in optional]
                if unrelated:
                    noise_symptom = random.choice(unrelated)
                    row[noise_symptom] = 1

            row["Disease"] = disease
            rows.append(row)

    df = pd.DataFrame(rows)
    return df

# -------------------------------------------------------------
# 4. TRAIN AND CALIBRATE MODEL
# -------------------------------------------------------------
def train_and_save():
    print("Generating comprehensive medical dataset...")
    df = generate_dataset(samples_per_disease=80)
    
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    dataset_path = "data/expanded_dataset.csv"
    df.to_csv(dataset_path, index=False)
    print(f"Dataset generated: {len(df)} rows across {df['Disease'].nunique()} diseases.")
    print(f"Saved to: {dataset_path}")

    # Features and Target
    X = df.drop("Disease", axis=1)
    y = df["Disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base_model = RandomForestClassifier(
        n_estimators=120,
        max_depth=16,
        min_samples_split=3,
        random_state=42,
        class_weight="balanced"
    )

    # Calibrate classifier for reliable probabilities
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=3
    )

    calibrated_model.fit(X_train, y_train)

    # Evaluation
    y_pred = calibrated_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Training Accuracy: {acc * 100:.2f}%")

    # Save model
    model_path = "models/disease_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(calibrated_model, f)
    print(f"Saved calibrated model to: {model_path}")

    # Save metadata
    metadata = {
        "symptoms": ALL_SYMPTOMS,
        "symptoms_catalog": SYMPTOMS_CATALOG,
        "disease_count": int(df["Disease"].nunique()),
        "accuracy": float(acc)
    }
    with open("models/symptoms.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open("models/medical_knowledge.json", "w", encoding="utf-8") as f:
        json.dump(DISEASE_PROFILES, f, indent=2)

    print("Medical knowledge base and symptom definitions saved successfully!")

if __name__ == "__main__":
    train_and_save()
