import json
import os
import random
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------------------------------------------
# 1. COMPREHENSIVE SYMPTOMS CATALOG (45 CLINICAL SYMPTOMS)
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
        "Night Sweats",
        "Unexplained Weight Loss",
        "Swollen Lymph Nodes",
    ],
    "Respiratory & ENT": [
        "Dry Cough",
        "Cough with Mucus",
        "Shortness of Breath",
        "Wheezing",
        "Sore Throat",
        "Runny or Stuffy Nose",
        "Sneezing",
        "Chest Tightness or Pain",
        "Loss of Taste or Smell",
        "Hoarseness or Voice Loss",
    ],
    "Digestive & Abdominal": [
        "Nausea",
        "Vomiting",
        "Diarrhea",
        "Abdominal Cramps & Pain",
        "Heartburn & Acid Reflux",
        "Loss of Appetite",
        "Abdominal Bloating",
        "Constipation",
        "Yellowing of Skin/Eyes (Jaundice)",
    ],
    "Neurological & Sensory": [
        "Severe Throbbing Headache",
        "Dull or Tension Headache",
        "Dizziness & Lightheadedness",
        "Sensitivity to Light/Sound",
        "Stiff Neck & Confusion",
        "Tremors or Shaking Hands",
        "Brain Fog & Poor Concentration",
        "Numbness or Tingling Sensation",
    ],
    "Skin, Eyes & Urinary": [
        "Itchy Skin Rash or Hives",
        "Red or Watery Eyes",
        "Burning Urination",
        "Frequent Urination",
        "Dark or Cloudy Urine",
        "Extreme Thirst & Dry Mouth",
        "Cold Hands & Feet",
        "Pale Skin & Brittle Nails",
    ],
}

# Flattened list of all symptom feature names
ALL_SYMPTOMS = []
for group in SYMPTOMS_CATALOG.values():
    ALL_SYMPTOMS.extend(group)

# -------------------------------------------------------------
# 2. MEDICAL CONDITIONS CLINICAL PROFILES (53 DISEASES)
# -------------------------------------------------------------
DISEASE_PROFILES = {
    "Influenza (Flu)": {
        "category": "Infectious & Viral",
        "severity": "Moderate",
        "specialist": "General Physician",
        "description": "Contagious respiratory viral infection causing acute febrile illness, intense myalgia, and prostration.",
        "core_symptoms": ["High Fever", "Chills & Shivering", "Body Aches & Muscle Pain", "Fatigue & Weakness"],
        "optional_symptoms": ["Dry Cough", "Sore Throat", "Severe Throbbing Headache", "Loss of Appetite"],
        "precautions": [
            "Get strict bed rest for 4-6 days to conserve metabolic energy.",
            "Maintain continuous oral hydration with warm broths, herbal teas, and ORS.",
            "Use antipyretics (e.g. Paracetamol) as advised by your physician; avoid aspirin in youths.",
            "Isolate at home in a well-ventilated space to prevent family transmission."
        ],
        "diet_advice": "Warm chicken/lentil soup, electrolyte solutions, citrus fruits high in Vitamin C, and herbal ginger tea.",
        "emergency_alert": "Seek urgent ER care if you develop blueish lips, severe chest tightness, or respiratory distress."
    },
    "Common Cold": {
        "category": "Respiratory",
        "severity": "Mild",
        "specialist": "General Physician / ENT",
        "description": "Mild upper respiratory viral infection primarily caused by rhinoviruses.",
        "core_symptoms": ["Runny or Stuffy Nose", "Sneezing", "Sore Throat"],
        "optional_symptoms": ["Mild Fever", "Dry Cough", "Fatigue & Weakness", "Dull or Tension Headache"],
        "precautions": [
            "Gargle with warm saline solution 3 times daily for pharyngeal comfort.",
            "Use steam inhalation or saline nasal spray to alleviate congestion.",
            "Rest adequately and stay out of cold drafts.",
            "Wash hands frequently with soap to avoid spreading virus particles."
        ],
        "diet_advice": "Honey-lemon warm water, clear broths, fresh fruits, and easily digestible home-cooked meals.",
        "emergency_alert": "Consult a physician if symptoms last more than 10 days without improvement or ear pain develops."
    },
    "COVID-19": {
        "category": "Infectious & Respiratory",
        "severity": "Moderate to High",
        "specialist": "Pulmonologist / Infectious Disease",
        "description": "Systemic coronavirus disease triggered by SARS-CoV-2 characterized by respiratory compromise and sensory loss.",
        "core_symptoms": ["High Fever", "Dry Cough", "Fatigue & Weakness", "Loss of Taste or Smell"],
        "optional_symptoms": ["Shortness of Breath", "Body Aches & Muscle Pain", "Chest Tightness or Pain", "Sore Throat"],
        "precautions": [
            "Isolate immediately in a designated room with cross-ventilation.",
            "Monitor blood oxygen saturation (SpO2) using a pulse oximeter twice daily.",
            "Get complete physical rest and practice prone breathing exercises if advised.",
            "Wear a medical-grade N95 mask during unavoidable interactions."
        ],
        "diet_advice": "Protein-rich nutrition (boiled eggs, pulses), warm herbal decoctions, Vitamin C & D rich foods.",
        "emergency_alert": "SEEK EMERGENCY HOSPITAL CARE if oxygen saturation drops below 94% or severe chest pain occurs."
    },
    "Acute Bronchitis": {
        "category": "Respiratory",
        "severity": "Moderate",
        "specialist": "Pulmonologist",
        "description": "Acute inflammation of the bronchial tree lining causing productive cough and mucosal irritation.",
        "core_symptoms": ["Cough with Mucus", "Chest Tightness or Pain", "Fatigue & Weakness"],
        "optional_symptoms": ["Mild Fever", "Shortness of Breath", "Wheezing", "Sore Throat"],
        "precautions": [
            "Avoid environmental tobacco smoke, dust, and cooking vapors entirely.",
            "Inhale moist steam or use a cool-mist room humidifier.",
            "Drink plenty of fluids to loosen thick tenacious tracheobronchial secretions.",
            "Avoid strenuous athletic exertion during recovery."
        ],
        "diet_advice": "Warm herbal teas with raw honey, warm clear soups, turmeric milk, and high fluid intake.",
        "emergency_alert": "Consult an emergency physician if you cough up blood (hemoptysis) or struggle for breath."
    },
    "Pneumonia": {
        "category": "Respiratory (Urgent)",
        "severity": "High",
        "specialist": "Pulmonologist / Critical Care",
        "description": "Infection inflaming the pulmonary alveoli in one or both lungs, leading to exudative consolidation.",
        "core_symptoms": ["High Fever", "Chills & Shivering", "Cough with Mucus", "Shortness of Breath"],
        "optional_symptoms": ["Chest Tightness or Pain", "Fatigue & Weakness", "Loss of Appetite", "Night Sweats"],
        "precautions": [
            "Consult a pulmonologist immediately for a diagnostic chest radiograph (X-ray) and targeted antimicrobial therapy.",
            "Strictly complete the prescribed antibiotic or antiviral regimen without premature cessation.",
            "Rest elevated in a 45-degree upright position to facilitate thoracic expansion.",
            "Keep close track of SpO2 and respiratory rate."
        ],
        "diet_advice": "High-calorie, protein-dense soups, oral rehydration solutions, steamed tender vegetables.",
        "emergency_alert": "IMMEDIATE ER VISIT REQUIRED if lips turn cyanotic (blue), respirations become rapid, or confusion appears."
    },
    "Asthma Exacerbation": {
        "category": "Chronic Respiratory",
        "severity": "Moderate to High",
        "specialist": "Pulmonologist / Allergist",
        "description": "Acute hyper-reactive bronchospasm leading to airway narrowing, wheezing, and dyspnea.",
        "core_symptoms": ["Shortness of Breath", "Wheezing", "Chest Tightness or Pain"],
        "optional_symptoms": ["Dry Cough", "Fatigue & Weakness", "Cold Hands & Feet"],
        "precautions": [
            "Immediately administer your prescribed quick-relief rescue inhaler (e.g. Salbutamol / Albuterol).",
            "Sit completely upright and remain calm; avoid lying flat which worsens airway collapse.",
            "Remove yourself from triggers (dust, cold air, cat/dog dander, aerosol sprays).",
            "Follow your personalized Asthma Action Plan."
        ],
        "diet_advice": "Anti-inflammatory Mediterranean diet, omega-3 rich foods, warm fluids; avoid sulfited beverages.",
        "emergency_alert": "CALL 911 / 112 IF inhaler does not produce relief within 15 minutes or you cannot speak in full sentences."
    },
    "Allergic Rhinitis": {
        "category": "Allergy & Immunology",
        "severity": "Mild",
        "specialist": "Allergist / ENT",
        "description": "Immunoglobulin E-mediated hypersensitivity reaction to airborne environmental allergens.",
        "core_symptoms": ["Sneezing", "Runny or Stuffy Nose", "Red or Watery Eyes"],
        "optional_symptoms": ["Itchy Skin Rash or Hives", "Sore Throat", "Fatigue & Weakness"],
        "precautions": [
            "Identify seasonal pollen counts and minimize outdoor exposure during peak morning winds.",
            "Perform daily saline nasal lavages with a sterile neti pot or rinse bottle.",
            "Keep domestic windows closed and use HEPA air purifiers.",
            "Use non-sedating antihistamines if prescribed."
        ],
        "diet_advice": "Foods rich in quercetin and bioflavonoids (apples, berries, onions) and green tea.",
        "emergency_alert": "Seek urgent care if facial swelling (angioedema) or throat tightness develops."
    },
    "Sinusitis": {
        "category": "ENT / Respiratory",
        "severity": "Moderate",
        "specialist": "ENT Specialist",
        "description": "Mucosal inflammation of the paranasal sinuses causing impaired ostiomeatal drainage and facial tension.",
        "core_symptoms": ["Severe Throbbing Headache", "Runny or Stuffy Nose", "Cough with Mucus"],
        "optional_symptoms": ["Mild Fever", "Sore Throat", "Fatigue & Weakness", "Loss of Taste or Smell"],
        "precautions": [
            "Apply warm moist towels around the zygomatic and frontal sinus areas.",
            "Perform isotonic saline nasal rinses twice a day.",
            "Maintain high fluid intake to thin mucosal secretions.",
            "Avoid sudden pressure changes (diving, unpressurized flying)."
        ],
        "diet_advice": "Warm spicy broths with garlic and ginger, herbal teas, and hot clear soups.",
        "emergency_alert": "Visit an ENT doctor immediately if periorbital swelling, double vision, or severe neck stiffness occurs."
    },
    "Strep Throat": {
        "category": "Bacterial ENT",
        "severity": "Moderate to High",
        "specialist": "ENT Specialist / General Physician",
        "description": "Acute pharyngeal infection provoked by Streptococcus pyogenes (Group A beta-hemolytic strep).",
        "core_symptoms": ["Sore Throat", "High Fever", "Swollen Lymph Nodes"],
        "optional_symptoms": ["Severe Throbbing Headache", "Loss of Appetite", "Body Aches & Muscle Pain"],
        "precautions": [
            "Obtain a rapid strep throat antigen swab test or throat culture.",
            "Adhere strictly to full course of prescribed oral antibiotics to avert acute rheumatic carditis.",
            "Gargle with warm salt water four times daily.",
            "Discard your old toothbrush 48 hours after starting antibiotics."
        ],
        "diet_advice": "Cool smoothies, chilled non-acidic yogurt, pureed warm soups, soft puddings, honey water.",
        "emergency_alert": "Emergency medical attention needed if inability to swallow saliva (drooling) or airway obstruction occurs."
    },
    "Acute Tonsillitis": {
        "category": "ENT",
        "severity": "Moderate",
        "specialist": "ENT Specialist",
        "description": "Acute inflammation and hypertrophy of the palatine tonsils with purulent exudate.",
        "core_symptoms": ["Sore Throat", "Swollen Lymph Nodes", "High Fever"],
        "optional_symptoms": ["Hoarseness or Voice Loss", "Difficulty Swallowing", "Severe Throbbing Headache"],
        "precautions": [
            "Gargle frequently with warm saline water to loosen tonsillar debris.",
            "Use throat lozenges containing mild local anesthetics.",
            "Rest the vocal cords and avoid whispering or shouting.",
            "Consult an ENT for evaluation of bacterial versus viral etiology."
        ],
        "diet_advice": "Soft, lukewarm foods; blended oats, yogurt, vegetable purees; avoid rough crisps and spicy chips.",
        "emergency_alert": "Seek urgent surgical consultation if one-sided severe throat swelling (peritonsillar abscess) develops."
    },
    "Laryngitis": {
        "category": "ENT",
        "severity": "Mild",
        "specialist": "ENT / General Physician",
        "description": "Inflammation of the vocal folds (larynx) causing dysphonia or complete aphonia.",
        "core_symptoms": ["Hoarseness or Voice Loss", "Sore Throat", "Dry Cough"],
        "optional_symptoms": ["Mild Fever", "Fatigue & Weakness", "Runny or Stuffy Nose"],
        "precautions": [
            "Strict vocal rest: avoid talking and especially whispering, which strains vocal cords more.",
            "Inhale steam for 10 minutes twice daily.",
            "Hydrate extensively with caffeine-free warm beverages.",
            "Avoid alcohol, menthol throat drops, and decongestants that dry mucous membranes."
        ],
        "diet_advice": "Lukewarm herbal chamomile tea with honey, room-temperature soups, avoid acidic citrus drinks.",
        "emergency_alert": "See a specialist if hoarseness lasts more than 2 weeks without a respiratory infection."
    },
    "Whooping Cough (Pertussis)": {
        "category": "Infectious Respiratory",
        "severity": "Moderate to High",
        "specialist": "Pulmonologist / Pediatrician",
        "description": "Contagious respiratory tract infection caused by Bordetella pertussis leading to paroxysmal coughing spasms.",
        "core_symptoms": ["Dry Cough", "Shortness of Breath", "Vomiting"],
        "optional_symptoms": ["Runny or Stuffy Nose", "Mild Fever", "Fatigue & Weakness"],
        "precautions": [
            "Consult a physician for PCR testing and macrolide antibiotic prescription.",
            "Ensure household members check their Tdap booster immunization status.",
            "Use a cool-mist vaporizer in the patient's sleeping quarters.",
            "Provide small, frequent meals to prevent vomiting after coughing fits."
        ],
        "diet_advice": "Plenty of fluids, warm broths, mashed bananas, and soft custards.",
        "emergency_alert": "EMERGENCY: Urgent hospitalization required if coughing spasms induce apnea, fainting, or blue face."
    },
    "COPD Flare-up": {
        "category": "Chronic Pulmonary",
        "severity": "High",
        "specialist": "Pulmonologist",
        "description": "Acute worsening of chronic obstructive pulmonary disease symptoms with increased airflow limitation.",
        "core_symptoms": ["Shortness of Breath", "Cough with Mucus", "Wheezing"],
        "optional_symptoms": ["Chest Tightness or Pain", "Fatigue & Weakness", "Swollen Lymph Nodes"],
        "precautions": [
            "Use prescribed bronchodilators and systemic corticosteroids as instructed in your COPD plan.",
            "Practice pursed-lip breathing to stabilize airways during exhalation.",
            "Avoid all exposure to tobacco smoke, smog, and fireplace ashes.",
            "Monitor oxygen levels closely."
        ],
        "diet_advice": "High-protein, lower-carbohydrate meals (carbs produce more carbon dioxide), small frequent servings.",
        "emergency_alert": "Go to the nearest emergency department if severe breathlessness occurs while resting."
    },
    "Migraine": {
        "category": "Neurological",
        "severity": "Moderate to Severe",
        "specialist": "Neurologist",
        "description": "Neurovascular disorder marked by unilateral pulsing hemicrania, photophobia, phonophobia, and nausea.",
        "core_symptoms": ["Severe Throbbing Headache", "Sensitivity to Light/Sound", "Nausea"],
        "optional_symptoms": ["Vomiting", "Dizziness & Lightheadedness", "Brain Fog & Poor Concentration"],
        "precautions": [
            "Retire immediately to a dark, sound-damped room with cool compress over the forehead.",
            "Take prescribed abortive medications (e.g. triptans) at the earliest onset of the prodrome/aura.",
            "Avoid established neurochemical triggers: aspartame, aged cheeses, nitrates, wine, skipping meals.",
            "Maintain consistent circadian sleep cycles."
        ],
        "diet_advice": "Magnesium-dense nutrition (spinach, pumpkin seeds), electrolyte water, and ginger root tea.",
        "emergency_alert": "Seek urgent emergency care for 'thunderclap' sudden maximum-intensity headaches or limb paralysis."
    },
    "Tension Headache": {
        "category": "Neurological",
        "severity": "Mild to Moderate",
        "specialist": "General Physician / Neurologist",
        "description": "Bilateral band-like constrictive cephalalgia related to pericranial myofascial tension and psychological stress.",
        "core_symptoms": ["Dull or Tension Headache", "Fatigue & Weakness"],
        "optional_symptoms": ["Sensitivity to Light/Sound", "Body Aches & Muscle Pain", "Brain Fog & Poor Concentration"],
        "precautions": [
            "Perform gentle cervical stretches and trapezial myofascial massage.",
            "Enforce the 20-20-20 rule during prolonged computer and display screen use.",
            "Apply a warm therapeutic heating pad across tight posterior shoulder muscles.",
            "Practice mindfulness or progressive Jacobson muscle relaxation."
        ],
        "diet_advice": "Consistent balanced meals, adequate hydration, herbal peppermint tea; eliminate excess caffeine.",
        "emergency_alert": "Consult a neurologist if headaches become daily or accompanied by focal neurological deficits."
    },
    "Cluster Headache": {
        "category": "Neurological (Severe)",
        "severity": "Severe",
        "specialist": "Neurologist",
        "description": "Excruciating strictly unilateral periorbital autonomic cephalalgia occurring in cyclical cluster periods.",
        "core_symptoms": ["Severe Throbbing Headache", "Red or Watery Eyes", "Runny or Stuffy Nose"],
        "optional_symptoms": ["Sensitivity to Light/Sound", "Dizziness & Lightheadedness", "Excessive Sweating"],
        "precautions": [
            "Seek specialist evaluation for high-flow normobaric oxygen therapy (100% O2 via non-rebreather mask).",
            "Strictly avoid any alcohol intake during active cluster cycles.",
            "Consult a neurologist regarding verapamil or subcutaneous triptan therapies.",
            "Maintain an exact symptom diary recording attack durations and times."
        ],
        "diet_advice": "Maintain strict meal timing, eliminate vasodilating foods, ensure optimal hydration.",
        "emergency_alert": "Emergency medical evaluation needed if initial attack presents with altered sensorium."
    },
    "Meningitis Alert": {
        "category": "Neurological Emergency",
        "severity": "Critical / Emergency",
        "specialist": "Neurologist / Emergency Medicine",
        "description": "Life-threatening acute infection and inflammation of the leptomeninges surrounding the brain and spinal cord.",
        "core_symptoms": ["High Fever", "Stiff Neck & Confusion", "Severe Throbbing Headache"],
        "optional_symptoms": ["Sensitivity to Light/Sound", "Vomiting", "Nausea", "Brain Fog & Poor Concentration"],
        "precautions": [
            "DIAL EMERGENCY SERVICES (911 / 112 / 108) OR GO TO THE NEAREST ER IMMEDIATELY.",
            "Do not attempt any home remedies — bacterial meningitis can cause mortality within hours.",
            "Prompt diagnostic lumbar puncture and intravenous broad-spectrum antibiotics are critical.",
            "Isolate the patient from close contacts pending diagnosis."
        ],
        "diet_advice": "Nothing by mouth (NPO) in anticipation of emergency IV hospitalization.",
        "emergency_alert": "CRITICAL EMERGENCY: Stiff neck + high fever + altered consciousness is a medical emergency."
    },
    "Vertigo & Labyrinthitis": {
        "category": "Vestibular / ENT",
        "severity": "Moderate",
        "specialist": "ENT / Neurologist",
        "description": "Vestibular nerve inflammation causing false sensation of spinning movement, balance instability, and nausea.",
        "core_symptoms": ["Dizziness & Lightheadedness", "Nausea", "Brain Fog & Poor Concentration"],
        "optional_symptoms": ["Vomiting", "Sensitivity to Light/Sound", "Mild Fever"],
        "precautions": [
            "Lie completely still in a stable bed during acute episodes; avoid rapid head turning.",
            "Consult a doctor or physical therapist for canalith repositioning maneuvers (e.g. Epley maneuver).",
            "Keep pathways well-lit to prevent falls.",
            "Do not drive or operate heavy machinery while experiencing vertigo."
        ],
        "diet_advice": "Low-sodium diet to decrease inner ear endolymphatic fluid pressure; avoid caffeine and tobacco.",
        "emergency_alert": "Seek ER evaluation if dizziness is accompanied by slurred speech, double vision, or arm weakness (stroke alert)."
    },
    "Concussion / Head Injury": {
        "category": "Trauma & Neurological",
        "severity": "High",
        "specialist": "Neurologist / Emergency Care",
        "description": "Mild traumatic brain injury causing temporary disruption of normal neurofunctional operations.",
        "core_symptoms": ["Dizziness & Lightheadedness", "Dull or Tension Headache", "Brain Fog & Poor Concentration"],
        "optional_symptoms": ["Nausea", "Sensitivity to Light/Sound", "Fatigue & Weakness"],
        "precautions": [
            "Undergo immediate clinical evaluation to rule out intracranial hemorrhage.",
            "Strict cognitive and physical rest: eliminate screen time, reading, and sports for 48 hours.",
            "Have a companion monitor the patient for the first 24 hours.",
            "Avoid alcohol, sedatives, and driving until cleared by a physician."
        ],
        "diet_advice": "Brain-supporting omega-3 fatty acids, blueberries, dark leafy greens, plenty of water.",
        "emergency_alert": "EMERGENCY: Unequal pupils, repetitive vomiting, worsening confusion, or seizures requires immediate ER care."
    },
    "Gastroenteritis (Stomach Flu)": {
        "category": "Gastrointestinal",
        "severity": "Moderate",
        "specialist": "Gastroenterologist / Physician",
        "description": "Acute viral or bacterial inflammation of the gastric and intestinal mucosa with rapid fluid loss.",
        "core_symptoms": ["Diarrhea", "Vomiting", "Nausea", "Abdominal Cramps & Pain"],
        "optional_symptoms": ["Mild Fever", "Fatigue & Weakness", "Loss of Appetite", "Chills & Shivering"],
        "precautions": [
            "Consume Oral Rehydration Salts (ORS) in slow, frequent teaspoon sips to avert hypovolemia.",
            "Avoid milk products, high-fat foods, spicy dishes, and sodas until gut flora recovers.",
            "Do not immediately use anti-motility drugs unless physician approved (gut must flush pathogens).",
            "Meticulously disinfect contaminated bathroom surfaces."
        ],
        "diet_advice": "BRAT protocol (Bananas, Rice, Applesauce, Toast), coconut water, clear rice kanji/water.",
        "emergency_alert": "Go to hospital if unable to retain fluids for 24 hours or signs of dehydration (sunken eyes, no urination) occur."
    },
    "Food Poisoning": {
        "category": "Gastrointestinal",
        "severity": "Moderate",
        "specialist": "General Physician",
        "description": "Acute toxin-mediated gastrointestinal distress resulting from consumption of contaminated food items.",
        "core_symptoms": ["Vomiting", "Diarrhea", "Abdominal Cramps & Pain", "Nausea"],
        "optional_symptoms": ["High Fever", "Chills & Shivering", "Fatigue & Weakness", "Extreme Thirst & Dry Mouth"],
        "precautions": [
            "Prioritize rehydration with balanced glucose-electrolyte solutions.",
            "Allow the stomach to rest for 2-3 hours following active vomiting before re-challenging fluids.",
            "Secure any remaining food samples if public health tracing is required.",
            "Wash hands thoroughly after all bathroom visits."
        ],
        "diet_advice": "Electrolyte fluids, diluted clear apple juice, salted crackers, plain boiled white rice.",
        "emergency_alert": "Immediate hospital visit if stool contains visible blood or fever exceeds 102°F (38.9°C)."
    },
    "GERD / Acid Reflux": {
        "category": "Gastrointestinal",
        "severity": "Mild to Moderate",
        "specialist": "Gastroenterologist",
        "description": "Chronic retrograde flow of gastric acid into the esophagus injuring esophageal epithelium.",
        "core_symptoms": ["Heartburn & Acid Reflux", "Chest Tightness or Pain", "Abdominal Bloating"],
        "optional_symptoms": ["Nausea", "Sore Throat", "Dry Cough", "Loss of Appetite"],
        "precautions": [
            "Remain strictly upright for at least 3 hours following completion of any meal.",
            "Elevate the head of your bed by 6 to 8 inches using bed risers.",
            "Consume smaller, frequent meals rather than heavy saturated feasts.",
            "Avoid trigger substances: peppermint, caffeine, chocolate, citrus, and deep-fried dishes."
        ],
        "diet_advice": "Alkaline foods: oatmeal, bananas, melons, steamed green beans, chamomile tea, lean grilled poultry.",
        "emergency_alert": "IMPORTANT: If retrosternal chest pain radiates to left arm or jaw with cold sweating, treat as cardiac emergency."
    },
    "Peptic Ulcer Disease": {
        "category": "Gastrointestinal",
        "severity": "Moderate to High",
        "specialist": "Gastroenterologist",
        "description": "Focal mucosal excavations in the stomach or duodenum typically caused by H. pylori or chronic NSAID use.",
        "core_symptoms": ["Abdominal Cramps & Pain", "Heartburn & Acid Reflux", "Loss of Appetite"],
        "optional_symptoms": ["Nausea", "Vomiting", "Abdominal Bloating", "Fatigue & Weakness"],
        "precautions": [
            "Consult a gastroenterologist for Helicobacter pylori diagnostic testing and upper endoscopy.",
            "Immediately stop all non-steroidal anti-inflammatory drugs (NSAIDs like ibuprofen, aspirin).",
            "Completely avoid ethanol consumption and nicotine smoking.",
            "Maintain consistent meal schedules to avoid empty stomach acid spikes."
        ],
        "diet_advice": "Bland non-irritating diet: boiled potatoes, probiotic kefir, cabbage juice, steamed leafy greens.",
        "emergency_alert": "EMERGENCY: Seek hospital care immediately if vomiting 'coffee-ground' blood or passing pitch-black tarry stools."
    },
    "Irritable Bowel Syndrome (IBS)": {
        "category": "Functional GI",
        "severity": "Moderate",
        "specialist": "Gastroenterologist",
        "description": "Functional bowel disorder marked by abdominal discomfort associated with altered defecation habits.",
        "core_symptoms": ["Abdominal Cramps & Pain", "Abdominal Bloating", "Diarrhea"],
        "optional_symptoms": ["Constipation", "Nausea", "Fatigue & Weakness"],
        "precautions": [
            "Follow a Low-FODMAP dietary protocol under guidance of a clinical dietitian.",
            "Keep a daily food, stress, and symptom journal to uncover individualized triggers.",
            "Engage in regular moderate aerobic exercise to stimulate healthy bowel motility.",
            "Incorporate stress-reduction practices like yoga or cognitive behavioral therapy."
        ],
        "diet_advice": "Low-FODMAP meals: rice, oats, carrots, cucumbers, blueberries, lactose-free milk.",
        "emergency_alert": "Consult a physician to rule out inflammatory bowel disease if unexplained weight loss occurs."
    },
    "Appendicitis": {
        "category": "Surgical Emergency",
        "severity": "Critical / Emergency",
        "specialist": "General Surgeon / Emergency Department",
        "description": "Acute luminal obstruction and inflammation of the vermiform appendix risking perforation and peritonitis.",
        "core_symptoms": ["Abdominal Cramps & Pain", "Nausea", "Vomiting", "Loss of Appetite"],
        "optional_symptoms": ["High Fever", "Constipation", "Diarrhea", "Fatigue & Weakness"],
        "precautions": [
            "PROCEED TO THE NEAREST HOSPITAL EMERGENCY DEPARTMENT IMMEDIATELY.",
            "Do NOT consume any food, liquids, or oral medications (pre-operative fasting required).",
            "Do NOT apply heat pads or hot compresses to the abdomen (accelerates rupture).",
            "Do NOT ingest laxatives or enemas."
        ],
        "diet_advice": "Strictly NPO (Nothing by mouth) pending urgent surgical evaluation.",
        "emergency_alert": "CRITICAL: Excruciating right lower quadrant pain with rebound tenderness requires emergency surgery."
    },
    "Gallstones / Cholecystitis": {
        "category": "Hepatobiliary",
        "severity": "High",
        "specialist": "Gastroenterologist / General Surgeon",
        "description": "Biliary calculi blocking cystic duct causing acute gallbladder inflammation and biliary colic.",
        "core_symptoms": ["Abdominal Cramps & Pain", "Nausea", "Vomiting"],
        "optional_symptoms": ["Yellowing of Skin/Eyes (Jaundice)", "Mild Fever", "Heartburn & Acid Reflux"],
        "precautions": [
            "Consult a surgeon or gastroenterologist for abdominal ultrasonography.",
            "Avoid high-fat, deep-fried, and greasy meals that trigger cholecystokinin gallbladder contraction.",
            "Manage pain only with physician-prescribed analgesics.",
            "Monitor body temperature and skin color."
        ],
        "diet_advice": "Very low-fat diet: clear vegetable broths, steamed fish, plain rice, apples, leafy greens.",
        "emergency_alert": "Go to ER if severe right upper quadrant abdominal pain radiates to right shoulder with fever and jaundice."
    },
    "Acute Pancreatitis": {
        "category": "Gastrointestinal Emergency",
        "severity": "Critical / Emergency",
        "specialist": "Gastroenterologist / ICU Team",
        "description": "Premature enzymatic autodigestion of pancreatic parenchyma causing systemic inflammatory cascade.",
        "core_symptoms": ["Abdominal Cramps & Pain", "Nausea", "Vomiting"],
        "optional_symptoms": ["High Fever", "Abdominal Bloating", "Dizziness & Lightheadedness"],
        "precautions": [
            "SEEK IMMEDIATE EMERGENCY HOSPITAL ADMISSION.",
            "Patients require intensive intravenous fluid resuscitation and pain control.",
            "Do not consume any oral food or fluids.",
            "Completely avoid all alcohol."
        ],
        "diet_advice": "Strict medical hospital fasting (NPO) until enzyme levels stabilize.",
        "emergency_alert": "CRITICAL EMERGENCY: Severe epigastric pain boring through to the back with vomiting requires immediate ER."
    },
    "Chronic Constipation & Dyspepsia": {
        "category": "Digestive",
        "severity": "Mild",
        "specialist": "Gastroenterologist / Physician",
        "description": "Infrequent or difficult evacuation of feces accompanied by abdominal distension and discomfort.",
        "core_symptoms": ["Constipation", "Abdominal Bloating", "Abdominal Cramps & Pain"],
        "optional_symptoms": ["Loss of Appetite", "Nausea", "Fatigue & Weakness"],
        "precautions": [
            "Gradually increase soluble and insoluble dietary fiber to 25-35 grams daily.",
            "Drink at least 2.5 to 3 liters of water daily to soften stool consistency.",
            "Establish a regular unhurried morning bathroom routine.",
            "Engage in daily brisk walking to promote intestinal peristalsis."
        ],
        "diet_advice": "Prunes, flaxseeds, psyllium husk in water, kiwi fruit, whole grains, steamed vegetables.",
        "emergency_alert": "Seek urgent medical care if complete inability to pass gas or stool is accompanied by severe vomiting."
    },
    "Dengue Fever": {
        "category": "Tropical & Vector-Borne",
        "severity": "High",
        "specialist": "Infectious Disease / Physician",
        "description": "Flavivirus infection transmitted by Aedes mosquitoes causing break-bone fever and thrombocytopenia.",
        "core_symptoms": ["High Fever", "Body Aches & Muscle Pain", "Joint Pain & Swelling", "Severe Throbbing Headache"],
        "optional_symptoms": ["Itchy Skin Rash or Hives", "Nausea", "Vomiting", "Fatigue & Weakness"],
        "precautions": [
            "Obtain urgent Complete Blood Count (CBC) and Dengue NS1/IgM antigen tests.",
            "Track platelet counts daily under direct physician supervision.",
            "NEVER take Aspirin, Ibuprofen, or NSAIDs as they precipitate severe hemorrhagic bleeding.",
            "Maintain rigorous fluid intake with ORS, fresh coconut water, and soups."
        ],
        "diet_advice": "Pomegranate, fresh kiwi fruit, coconut water, well-cooked lentils, light vegetable soups.",
        "emergency_alert": "GO TO ER IMMEDIATELY if you experience gum bleeding, epistaxis, black stools, or severe abdominal pain."
    },
    "Malaria": {
        "category": "Parasitic Infectious",
        "severity": "High",
        "specialist": "Infectious Disease Specialist",
        "description": "Protozoan parasitic infection transmitted by female Anopheles mosquitoes causing paroxysmal fevers.",
        "core_symptoms": ["High Fever", "Chills & Shivering", "Excessive Sweating", "Severe Throbbing Headache"],
        "optional_symptoms": ["Body Aches & Muscle Pain", "Fatigue & Weakness", "Nausea", "Vomiting"],
        "precautions": [
            "Obtain immediate peripheral blood smear microscopy or Rapid Diagnostic Test (RDT).",
            "Commence doctor-prescribed Artemisinin Combination Therapy (ACT) without delay.",
            "Sleep under long-lasting insecticide-treated bed nets.",
            "Complete all days of the anti-malarial medication course."
        ],
        "diet_advice": "High-carbohydrate easily digestible meals with abundant fluids, coconut water, and citrus juices.",
        "emergency_alert": "Immediate hospitalization needed if patient experiences confusion, seizures, or dark tea-colored urine."
    },
    "Typhoid Fever": {
        "category": "Bacterial Infectious",
        "severity": "High",
        "specialist": "Infectious Disease / Physician",
        "description": "Systemic bacterial infection caused by Salmonella enterica serovar Typhi through contaminated water/food.",
        "core_symptoms": ["High Fever", "Abdominal Cramps & Pain", "Fatigue & Weakness", "Severe Throbbing Headache"],
        "optional_symptoms": ["Loss of Appetite", "Constipation", "Diarrhea", "Night Sweats"],
        "precautions": [
            "Consult a physician for blood culture, Typhidot, and targeted antibiotic therapy.",
            "Drink only rigorously boiled or bottled water; avoid street beverages and ice cubes.",
            "Scrupulously wash hands before meal preparation and following defecation.",
            "Finish the full antibiotic duration to avoid asymptomatic carrier state."
        ],
        "diet_advice": "Soft bland diet: boiled khichdi, mashed potatoes, bananas, clear broths; avoid rough raw salads.",
        "emergency_alert": "Seek urgent hospitalization if sudden acute abdominal rigidity, delirium, or intestinal bleeding occurs."
    },
    "Chikungunya": {
        "category": "Tropical & Vector-Borne",
        "severity": "Moderate to High",
        "specialist": "Rheumatologist / Physician",
        "description": "Alphavirus infection transmitted by Aedes mosquitoes characterized by debilitating polyarthralgia.",
        "core_symptoms": ["High Fever", "Joint Pain & Swelling", "Body Aches & Muscle Pain"],
        "optional_symptoms": ["Itchy Skin Rash or Hives", "Fatigue & Weakness", "Severe Throbbing Headache"],
        "precautions": [
            "Get serological confirmation via IgM ELISA or RT-PCR.",
            "Rest painful joints and apply cold compresses to swollen areas.",
            "Take acetaminophen for fever and pain control as prescribed.",
            "Perform gentle passive joint mobilizations once fever resolves."
        ],
        "diet_advice": "Anti-inflammatory foods: cherries, turmeric milk, omega-3 rich nutrition, plenty of fluids.",
        "emergency_alert": "Consult a doctor if joint swelling impairs walking or signs of secondary bacterial infection appear."
    },
    "Viral Fever & Body Fatigue": {
        "category": "General Medicine",
        "severity": "Mild to Moderate",
        "specialist": "General Physician",
        "description": "Common self-limiting non-specific viral febrile illness with generalized myalgia and malaise.",
        "core_symptoms": ["High Fever", "Body Aches & Muscle Pain", "Fatigue & Weakness", "Dull or Tension Headache"],
        "optional_symptoms": ["Chills & Shivering", "Loss of Appetite", "Mild Fever"],
        "precautions": [
            "Complete physical bed rest for 3 to 5 days until afebrile for 24 hours.",
            "Lukewarm sponge baths if body temperature surpasses 101°F.",
            "Sip oral rehydration fluids and warm water throughout the day.",
            "Take antipyretics only as guided by a licensed medical provider."
        ],
        "diet_advice": "Light porridge, mung dal khichdi, fresh seasonal fruits, warm milk with turmeric before sleep.",
        "emergency_alert": "Consult a doctor if fever persists beyond 96 hours or skin petechiae appear."
    },
    "Infectious Mononucleosis": {
        "category": "Infectious & Viral",
        "severity": "Moderate",
        "specialist": "Infectious Disease / Physician",
        "description": "Epstein-Barr virus (EBV) infection causing prolonged fatigue, tonsillopharyngitis, and lymphadenopathy.",
        "core_symptoms": ["Fatigue & Weakness", "Sore Throat", "Swollen Lymph Nodes", "High Fever"],
        "optional_symptoms": ["Body Aches & Muscle Pain", "Night Sweats", "Loss of Appetite"],
        "precautions": [
            "Obtain a Monospot or EBV antibody blood panel.",
            "STRICTLY AVOID contact sports and heavy lifting for 4-6 weeks to prevent splenic rupture.",
            "Get abundant bed rest and avoid sharing cups or kissing.",
            "Gargle with warm salt water for severe throat discomfort."
        ],
        "diet_advice": "Soft nutritious meals, smoothies, high fluid volume, easily digestible proteins.",
        "emergency_alert": "Go to the emergency department if sudden sharp left upper quadrant abdominal pain develops."
    },
    "Urinary Tract Infection (UTI)": {
        "category": "Urological",
        "severity": "Moderate",
        "specialist": "Urologist / General Physician",
        "description": "Bacterial colonization of the urothelium causing dysuria, frequency, and suprapubic pain.",
        "core_symptoms": ["Burning Urination", "Frequent Urination", "Dark or Cloudy Urine"],
        "optional_symptoms": ["Abdominal Cramps & Pain", "Mild Fever", "Fatigue & Weakness"],
        "precautions": [
            "Consult a physician for urinalysis and targeted antibiotic prescription.",
            "Drink copious amounts of plain water to mechanically flush bacteria from the bladder.",
            "Do not delay urination when feeling the micturition urge.",
            "Always wipe from front to back to prevent perineal bacterial translocation."
        ],
        "diet_advice": "Pure unsweetened cranberry juice, water, barley water; avoid caffeine, alcohol, and spicy condiments.",
        "emergency_alert": "Urgent review required if high fever, rigors, and flank/back pain occur (signaling acute pyelonephritis)."
    },
    "Kidney Stones (Nephrolithiasis)": {
        "category": "Nephrology / Urology",
        "severity": "High",
        "specialist": "Urologist / Nephrologist",
        "description": "Mineral precipitates forming urinary calculi causing severe spasmodic renal colic and hematuria.",
        "core_symptoms": ["Abdominal Cramps & Pain", "Burning Urination", "Frequent Urination"],
        "optional_symptoms": ["Nausea", "Vomiting", "Dark or Cloudy Urine"],
        "precautions": [
            "Consult a urologist for non-contrast CT KUB scan to ascertain stone diameter and site.",
            "Drink 2.5 to 3 liters of water per day to promote spontaneous stone passage if under 5mm.",
            "Utilize prescribed medical expulsive therapy (e.g. alpha-blockers) under supervision.",
            "Filter urine through a fine mesh strainer to collect stones for crystallographic analysis."
        ],
        "diet_advice": "Citrus lemon water (citrate inhibits crystal precipitation), restrict sodium, avoid high-oxalate nuts.",
        "emergency_alert": "Seek urgent ER care if completely unable to void urine or experiencing unbearable pain with fever."
    },
    "Acute Pyelonephritis": {
        "category": "Urological (Urgent)",
        "severity": "High",
        "specialist": "Nephrologist / Urologist",
        "description": "Ascending bacterial infection reaching the renal parenchyma requiring prompt medical management.",
        "core_symptoms": ["High Fever", "Chills & Shivering", "Burning Urination", "Abdominal Cramps & Pain"],
        "optional_symptoms": ["Nausea", "Vomiting", "Dark or Cloudy Urine", "Fatigue & Weakness"],
        "precautions": [
            "Consult a physician immediately for urine culture and parenteral or oral antibiotics.",
            "Complete full 10-14 day antibiotic duration without skipping doses.",
            "Monitor core temperature every 4 hours.",
            "Rest in bed and maintain high fluid volume."
        ],
        "diet_advice": "High-volume water intake, light nutrient broths, easily digestible porridge.",
        "emergency_alert": "Immediate hospitalization needed if vomiting prevents keeping oral antibiotic tablets down."
    },
    "Hypertension Warning Episode": {
        "category": "Cardiovascular",
        "severity": "High to Emergency",
        "specialist": "Cardiologist / Physician",
        "description": "Acute elevation in systemic arterial blood pressure exerting acute strain on vascular end-organs.",
        "core_symptoms": ["Severe Throbbing Headache", "Dizziness & Lightheadedness", "Chest Tightness or Pain"],
        "optional_symptoms": ["Shortness of Breath", "Brain Fog & Poor Concentration", "Sensitivity to Light/Sound"],
        "precautions": [
            "Measure arterial blood pressure immediately using an accurate calibrated cuff.",
            "Sit quietly in an uncrossed, resting position in a quiet environment for 10 minutes.",
            "Take prescribed antihypertensive medications if indicated by your doctor.",
            "Completely avoid sodium, caffeine, energy drinks, and tobacco."
        ],
        "diet_advice": "DASH diet: potassium-rich bananas, leafy greens, unsalted nuts, hibiscus tea.",
        "emergency_alert": "EMERGENCY: If systolic BP > 180 or diastolic > 120 with chest pain or blurred vision, call 911/112."
    },
    "Angina / Chest Pain Alert": {
        "category": "Cardiovascular Emergency",
        "severity": "Critical / Emergency",
        "specialist": "Cardiologist / ER Team",
        "description": "Coronary artery disease leading to myocardial ischemia and retrosternal constriction.",
        "core_symptoms": ["Chest Tightness or Pain", "Shortness of Breath", "Fatigue & Weakness"],
        "optional_symptoms": ["Dizziness & Lightheadedness", "Excessive Sweating", "Nausea"],
        "precautions": [
            "CALL EMERGENCY SERVICES (911 / 112 / 108) IMMEDIATELY.",
            "Chew an uncoated adult aspirin (325mg) if advised by emergency dispatch and no allergy.",
            "Cease all physical exertion and sit upright in a comfortable position.",
            "Loosen tight collars and restrictive clothing."
        ],
        "diet_advice": "Emergency cardiac situation: do not consume food.",
        "emergency_alert": "CRITICAL EMERGENCY: Crushing chest pressure radiating to left arm, neck, or jaw requires immediate ER."
    },
    "Iron Deficiency Anemia": {
        "category": "Hematological",
        "severity": "Moderate",
        "specialist": "Hematologist / Physician",
        "description": "Depleted iron reserves impairing erythropoiesis and tissue oxygen delivery.",
        "core_symptoms": ["Fatigue & Weakness", "Pale Skin & Brittle Nails", "Dizziness & Lightheadedness"],
        "optional_symptoms": ["Shortness of Breath", "Cold Hands & Feet", "Dull or Tension Headache"],
        "precautions": [
            "Obtain Complete Blood Count (CBC) and serum ferritin/iron saturation panel.",
            "Take prescribed oral iron supplements alongside Vitamin C to optimize intestinal absorption.",
            "Do not consume black tea, coffee, or calcium pills within 2 hours of iron doses (blocks absorption).",
            "Pace your daily physical routines."
        ],
        "diet_advice": "Spinach, lentils, red meat, pomegranates, beets, dates, pumpkin seeds, citrus fruits.",
        "emergency_alert": "Consult a doctor urgently if fainting (syncope), palpitations, or shortness of breath while seated occurs."
    },
    "Dehydration & Heat Exhaustion": {
        "category": "Environmental & Fluid Balance",
        "severity": "Moderate",
        "specialist": "General Physician",
        "description": "Depletion of body water and sodium volume due to thermal exposure, diaphoresis, or restricted intake.",
        "core_symptoms": ["Dizziness & Lightheadedness", "Extreme Thirst & Dry Mouth", "Fatigue & Weakness"],
        "optional_symptoms": ["Excessive Sweating", "Dull or Tension Headache", "Nausea", "Dark or Cloudy Urine"],
        "precautions": [
            "Relocate immediately to an air-conditioned room or cool shaded shelter.",
            "Sip cool electrolyte solutions, ORS, or coconut water slowly.",
            "Loosen tight restrictive garments and apply damp cool towels to the neck and axillae.",
            "Lie flat with legs slightly elevated to promote venous return."
        ],
        "diet_advice": "Watermelon, cucumbers, chilled buttermilk with salt, oral rehydration solutions.",
        "emergency_alert": "CALL EMERGENCY SERVICES if the person stops sweating, exhibits hot dry skin, or loses consciousness (Heat Stroke)."
    },
    "Type 2 Diabetes Flare": {
        "category": "Endocrine & Metabolic",
        "severity": "Moderate to High",
        "specialist": "Endocrinologist / Diabetologist",
        "description": "Marked glycemic dysregulation with progressive insulin resistance and hyperglycemia.",
        "core_symptoms": ["Extreme Thirst & Dry Mouth", "Frequent Urination", "Fatigue & Weakness"],
        "optional_symptoms": ["Unexplained Weight Loss", "Brain Fog & Poor Concentration", "Burning Urination"],
        "precautions": [
            "Measure capillary blood glucose immediately using a glucometer.",
            "Take physician-prescribed oral hypoglycemic agents or insulin as directed.",
            "Drink plenty of water to assist renal clearance of excess glucose.",
            "Undergo Glycated Hemoglobin (HbA1c) testing."
        ],
        "diet_advice": "Low-glycemic-index whole grains, bitter gourd, fenugreek water, leafy vegetables; eliminate refined sugars.",
        "emergency_alert": "Seek urgent ER care if blood glucose > 300 mg/dL with persistent vomiting or fruity breath (Ketoacidosis alert)."
    },
    "Hypoglycemia Episode": {
        "category": "Endocrine Emergency",
        "severity": "High",
        "specialist": "Endocrinologist / Emergency Care",
        "description": "Dangerous drop in plasma glucose concentration (< 70 mg/dL) starving cerebral tissues of glucose.",
        "core_symptoms": ["Tremors or Shaking Hands", "Dizziness & Lightheadedness", "Excessive Sweating"],
        "optional_symptoms": ["Brain Fog & Poor Concentration", "Fatigue & Weakness", "Extreme Thirst & Dry Mouth"],
        "precautions": [
            "Execute the 15-15 Rule: Consume 15 grams of fast-acting sugar (half cup fruit juice or 3 glucose tablets).",
            "Re-check capillary blood glucose in exactly 15 minutes.",
            "Repeat treatment if glucose remains below 70 mg/dL.",
            "Once normalized, eat a complex carbohydrate snack with protein (e.g. peanut butter toast)."
        ],
        "diet_advice": "Immediate: fruit juice, glucose candy. Follow-up: whole wheat crackers with cheese or nuts.",
        "emergency_alert": "CALL 911 / 112 if the patient is unable to swallow, becomes unresponsive, or experiences seizures."
    },
    "Hypothyroidism": {
        "category": "Endocrine",
        "severity": "Moderate",
        "specialist": "Endocrinologist",
        "description": "Underactive thyroid gland resulting in systemic metabolic slowing and fatigue.",
        "core_symptoms": ["Fatigue & Weakness", "Cold Hands & Feet", "Brain Fog & Poor Concentration"],
        "optional_symptoms": ["Constipation", "Pale Skin & Brittle Nails", "Unexplained Weight Loss"],
        "precautions": [
            "Consult an endocrinologist for serum Thyroid Stimulating Hormone (TSH) and Free T4 panel.",
            "Take prescribed levothyroxine consistently on an empty stomach with water 30 minutes before breakfast.",
            "Do not consume calcium or iron pills within 4 hours of thyroid hormone.",
            "Maintain regular laboratory check-ups every 6-12 weeks until stable."
        ],
        "diet_advice": "Iodine and selenium-containing foods (brazil nuts, eggs), cooked cruciferous vegetables.",
        "emergency_alert": "Seek urgent care if severe lethargy, hypothermia, or marked facial swelling occurs."
    },
    "Allergic Urticaria (Hives)": {
        "category": "Dermatology & Allergy",
        "severity": "Mild to Moderate",
        "specialist": "Dermatologist / Allergist",
        "description": "Cutaneous vascular reaction marked by pruritic transient wheals provoked by histamine release.",
        "core_symptoms": ["Itchy Skin Rash or Hives", "Red or Watery Eyes"],
        "optional_symptoms": ["Sneezing", "Mild Fever", "Fatigue & Weakness"],
        "precautions": [
            "Apply cool damp compresses or calamine lotion to minimize itching.",
            "Take physician-recommended non-sedating H1-antihistamines.",
            "Wear loose-fitting, soft cotton clothing; avoid wool and synthetic tight garments.",
            "Keep a diary tracking recently ingested foods, medications, or contactants."
        ],
        "diet_advice": "Fresh antioxidant-rich produce, green tea, anti-inflammatory foods; avoid fermented products.",
        "emergency_alert": "CALL EMERGENCY SERVICES IMMEDIATELY if hives are accompanied by lip/tongue swelling or breathing difficulty."
    },
    "Contact Dermatitis": {
        "category": "Dermatology",
        "severity": "Mild",
        "specialist": "Dermatologist",
        "description": "Localized inflammatory skin reaction triggered by direct exposure to irritants or allergens.",
        "core_symptoms": ["Itchy Skin Rash or Hives"],
        "optional_symptoms": ["Red or Watery Eyes", "Fatigue & Weakness"],
        "precautions": [
            "Thoroughly cleanse the affected skin area with lukewarm water and mild hypoallergenic soap.",
            "Apply pure colloidal oatmeal creams or zinc oxide barrier pastes.",
            "Avoid scratching to prevent secondary bacterial staphylococcal impetiginization.",
            "Wear protective nitrile gloves when handling detergents, nickel jewelry, or harsh cleaners."
        ],
        "diet_advice": "High water intake, skin-nourishing foods with Vitamin E and zinc.",
        "emergency_alert": "Consult a doctor if rash spreads near eyes or develops yellow honey-colored crusting."
    },
    "Atopic Eczema Flare": {
        "category": "Dermatology",
        "severity": "Mild to Moderate",
        "specialist": "Dermatologist",
        "description": "Chronic relapsing pruritic dermatitis associated with epidermal skin barrier dysfunction.",
        "core_symptoms": ["Itchy Skin Rash or Hives", "Pale Skin & Brittle Nails"],
        "optional_symptoms": ["Fatigue & Weakness", "Red or Watery Eyes"],
        "precautions": [
            "Apply thick ceramide-based moisturizing ointments within 3 minutes of bathing.",
            "Take short, lukewarm showers instead of prolonged hot baths.",
            "Use fragrance-free detergents and mild soap-free cleansers.",
            "Maintain home relative humidity between 40-50%."
        ],
        "diet_advice": "Anti-inflammatory diet: flaxseed oil, fatty fish rich in omega-3s, probiotic yogurt.",
        "emergency_alert": "See a dermatologist if skin lesions develop painful vesicles or fever."
    },
    "Conjunctivitis (Pink Eye)": {
        "category": "Ophthalmology",
        "severity": "Mild",
        "specialist": "Ophthalmologist",
        "description": "Hyperemia and inflammation of the transparent conjunctival membrane of the eye.",
        "core_symptoms": ["Red or Watery Eyes", "Sensitivity to Light/Sound"],
        "optional_symptoms": ["Dull or Tension Headache", "Sneezing", "Runny or Stuffy Nose"],
        "precautions": [
            "Avoid rubbing or touching eyes; wash hands rigorously before and after eye care.",
            "Apply sterile cool or warm damp compresses to alleviate discomfort.",
            "Do not wear contact lenses or apply cosmetic eye makeup until completely resolved.",
            "Never share towels, pillowcases, or ophthalmic drops with family members."
        ],
        "diet_advice": "Foods rich in lutein, zeaxanthin, beta-carotene (carrots, bell peppers, spinach).",
        "emergency_alert": "Seek urgent eye care if experiencing severe ocular pain, photophobia, or significant vision reduction."
    },
    "Jaundice / Hepatitis Alert": {
        "category": "Hepatic / Gastrointestinal",
        "severity": "High",
        "specialist": "Hepatologist / Gastroenterologist",
        "description": "Hepatic parenchymal injury causing impaired bilirubin metabolism and hyperbilirubinemia.",
        "core_symptoms": ["Yellowing of Skin/Eyes (Jaundice)", "Fatigue & Weakness", "Loss of Appetite"],
        "optional_symptoms": ["Dark or Cloudy Urine", "Nausea", "Abdominal Cramps & Pain", "Vomiting"],
        "precautions": [
            "Undergo immediate comprehensive Liver Function Tests (LFT) and viral hepatitis serology.",
            "Strictly avoid all alcoholic drinks and unnecessary hepatotoxic medications (including excess paracetamol).",
            "Ensure complete physical rest to assist hepatic regeneration.",
            "Consume only boiled, purified drinking water and hygienically prepared food."
        ],
        "diet_advice": "Fresh sugarcane juice (prepared in sterile conditions), radish, barley water, boiled vegetables.",
        "emergency_alert": "IMMEDIATE HOSPITALIZATION REQUIRED if mental confusion, flapping hand tremors, or bleeding occurs."
    },
    "Shingles (Herpes Zoster)": {
        "category": "Infectious & Dermatological",
        "severity": "Moderate to High",
        "specialist": "Dermatologist / Neurologist",
        "description": "Reactivation of latent varicella-zoster virus within dorsal root ganglia causing unilateral dermatomal pain.",
        "core_symptoms": ["Itchy Skin Rash or Hives", "Numbness or Tingling Sensation", "Body Aches & Muscle Pain"],
        "optional_symptoms": ["Mild Fever", "Fatigue & Weakness", "Severe Throbbing Headache"],
        "precautions": [
            "Consult a physician within 72 hours of rash onset to start antiviral therapy (e.g. Valacyclovir).",
            "Keep the blistering rash clean, dry, and loosely covered to avoid spreading virus to unvaccinated individuals.",
            "Wear loose, smooth cotton garments.",
            "Apply cool calamine compresses to calm neuropathic burning."
        ],
        "diet_advice": "Foods rich in lysine (poultry, dairy, legumes); avoid excess arginine (chocolate, nuts, gelatin).",
        "emergency_alert": "EMERGENCY: Urgent ophthalmologic consultation needed if rash appears near or involves the eye."
    },
    "Gout Acute Attack": {
        "category": "Rheumatology / Metabolic",
        "severity": "Moderate to High",
        "specialist": "Rheumatologist",
        "description": "Monosodium urate crystal deposition in joints triggering severe acute inflammatory arthritis.",
        "core_symptoms": ["Joint Pain & Swelling", "Body Aches & Muscle Pain"],
        "optional_symptoms": ["Mild Fever", "Fatigue & Weakness", "Excessive Sweating"],
        "precautions": [
            "Elevate and rest the affected inflamed joint; do not walk on it during acute attacks.",
            "Apply wrapped ice packs for 20 minutes at a time to reduce acute swelling.",
            "Consult a physician for colchicine, NSAIDs, or corticosteroid therapy.",
            "Strictly avoid beer, distilled spirits, shellfish, organ meats, and high-fructose corn syrup."
        ],
        "diet_advice": "Tart cherry juice, low-fat dairy products, plenty of water, fresh cherries, and celery seeds.",
        "emergency_alert": "Consult a doctor if joint erythema and heat are accompanied by high fever (rule out septic arthritis)."
    },
    "Panic Attack & Hyperventilation": {
        "category": "Psychological & Autonomic",
        "severity": "Moderate",
        "specialist": "Psychiatrist / General Physician",
        "description": "Sudden surge of overwhelming autonomic fear triggering hyperventilation and intense somatic sensations.",
        "core_symptoms": ["Chest Tightness or Pain", "Shortness of Breath", "Dizziness & Lightheadedness"],
        "optional_symptoms": ["Tremors or Shaking Hands", "Numbness or Tingling Sensation", "Extreme Thirst & Dry Mouth"],
        "precautions": [
            "Practice Box Breathing: Inhale for 4 seconds, hold for 4 seconds, exhale for 4 seconds, hold for 4 seconds.",
            "Ground yourself using the 5-4-3-2-1 sensory technique (5 things you see, 4 you touch, 3 you hear, etc.).",
            "Remind yourself that panic attacks are temporary and do not cause physical harm.",
            "Avoid caffeine, nicotine, and stimulants."
        ],
        "diet_advice": "Warm herbal chamomile tea, complex whole grain snacks, magnesium-rich foods.",
        "emergency_alert": "Seek emergency evaluation if this is your first episode of chest pain to rule out cardiac pathology."
    }
}

# -------------------------------------------------------------
# 3. SYNTHETIC DATASET GENERATION
# -------------------------------------------------------------
def generate_dataset(samples_per_disease=75):
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
                if s in row and random.random() < 0.94:
                    row[s] = 1

            # Ensure at least 2 core symptoms are always active
            active_core = [s for s in core if row.get(s, 0) == 1]
            if len(active_core) < 2 and len(core) >= 2:
                for s in random.sample(core, min(2, len(core))):
                    row[s] = 1

            # Optional symptoms: present with 20-45% probability
            for s in optional:
                if s in row and random.random() < 0.35:
                    row[s] = 1

            # Small random noise (5% chance of 1 unrelated symptom)
            if random.random() < 0.06:
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
    print("Generating comprehensive medical dataset (53 conditions, 45 symptoms)...")
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

    model = RandomForestClassifier(
        n_estimators=60,
        max_depth=14,
        min_samples_split=2,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Training Accuracy: {acc * 100:.2f}%")

    # Save model (lightweight, ~8.9 MB, perfectly within GitHub's 100 MB limit)
    model_path = "models/disease_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved optimized model to: {model_path} ({round(os.path.getsize(model_path) / (1024 * 1024), 2)} MB)")

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
