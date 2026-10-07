# SymptoScan AI 🩺🤖
> **Next-Generation AI Clinical Triage & Symptom Assessment Platform**

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Flask Framework](https://img.shields.io/badge/Framework-Flask%203.x-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20Random%20Forest-orange.svg)](https://scikit-learn.org/)
[![Render Deploy](https://img.shields.io/badge/Deploy-Render%20Ready-green.svg)](https://render.com/)

SymptoScan AI is a full-stack, machine learning-powered educational healthcare application designed to evaluate patient-reported symptoms, deliver calibrated disease predictions with differential diagnoses, route patients to appropriate medical specialists, and generate printable clinical triage summaries.

---

## ✨ Key Features

- **🧠 Multi-Class Machine Learning Engine:**
  - Trained on 29+ distinct clinical conditions and 30+ categorized symptoms.
  - Calibrated Classifier (Random Forest + Sigmoid calibration) returning precise confidence probabilities.
  - Generates top **Differential Diagnoses** to account for overlapping symptoms.

- **👨‍⚕️ Clinical Triage & Guidance:**
  - Dynamic severity indicator: **Mild**, **Moderate**, **High**, **Critical / Emergency**.
  - Direct specialist recommendation (e.g. *Pulmonologist, Neurologist, Gastroenterologist, General Physician*).
  - 4-point evidence-based care plan & self-care precautions.
  - Nutrition, dietary, and hydration recommendations.
  - Emergency Red-Flag alerts for immediate ER attention.

- **🎨 Modern Clinical UI / UX:**
  - Interactive Symptom Checker with categorized filter tabs and instant live search.
  - One-click presets (*Flu-like, Common Cold, Stomach Upset, Migraine, Allergy*).
  - Responsive layout with sticky live symptom counter dock.
  - Floating interactive AI Health Assistant chat widget.
  - Printable medical triage summary report with PDF export formatting.

- **🗄️ Resilient Multi-Engine Database Architecture:**
  - **Zero-config SQLite** by default: Runs seamlessly locally and on Render with zero setup.
  - **PostgreSQL** support for Render Postgres via `DATABASE_URL`.
  - **MySQL** support for local development or remote instances.
  - Automated database table creation on startup.

- **⚡ RESTful API Endpoints:**
  - `GET /api/symptoms` — Retrieve categorized symptom lists.
  - `POST /api/predict` — Headless disease prediction for web/mobile apps.
  - `POST /api/assistant-chat` — Symptom triage chat assistant.
  - `GET /health` — Health check endpoint for cloud uptime monitors.

---

## 🛠️ Tech Stack

| Domain | Technology |
|---|---|
| **Backend** | Python 3, Flask, Gunicorn, Werkzeug |
| **Machine Learning** | Scikit-Learn (Random Forest, CalibratedClassifierCV), Pandas, NumPy |
| **Database** | SQLite (Default/Zero-Config), PostgreSQL, MySQL |
| **Frontend** | HTML5, CSS3, Vanilla JS, Plus Jakarta Sans, SVG Icons |
| **Cloud Deployment** | Render, Gunicorn WSGI |

---

## 🚀 Quick Start (Local Development)

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/smdanish03/SymptoScan.git
cd SymptoScan

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the AI Model
```bash
python train_model.py
```

### 4. Run the Application
```bash
python app.py
```
Open your browser at **`http://localhost:5000`**.

---

## ☁️ Deployment on Render

Deploying to Render takes under 3 minutes:

1. Push your repository to GitHub.
2. In [Render Dashboard](https://dashboard.render.com/), click **New +** → **Web Service**.
3. Select your repository.
4. Set:
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt && python train_model.py`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT`
5. Add Environment Variable:
   - `SECRET_KEY`: *(Generate secure key)*
6. Click **Create Web Service**.

> For comprehensive deployment instructions, check [DEPLOYMENT.md](file:///d:/SymptoScan/DEPLOYMENT.md).

---

## ⚠️ Medical Disclaimer

**SymptoScan AI is an educational demonstration and triage simulation tool.** It does **NOT** provide official medical diagnosis, treatment, or clinical prescriptions. Always consult a licensed medical physician or emergency services for any actual health conditions or emergencies.
