# 🚀 Render Deployment Guide for SymptoScan AI

SymptoScan AI is configured for seamless, 1-click deployment on **[Render](https://render.com/)**.

---

## ⚡ Option 1: Quick Web Service Deployment (Free Tier / Zero-Config)

With SymptoScan AI's dual-engine database layer, SQLite is used automatically if no external database is configured. This lets you deploy a fully functional live instance on Render in under 3 minutes for free.

### Step 1: Push Code to GitHub
Ensure all files are committed to your GitHub repository:
```bash
git add .
git commit -m "SymptoScan AI v2.0 - Production ready"
git push origin main
```

### Step 2: Create a Web Service on Render
1. Log in to your [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** → **Web Service**.
3. Connect your GitHub repository (`SymptoScan`).
4. Configure the settings:
   - **Name:** `symptoscan-ai` (or your preferred name)
   - **Region:** Choose the region closest to you (e.g., *Oregon (US West)* or *Frankfurt (EU)*)
   - **Branch:** `main`
   - **Runtime:** `Python 3`
   - **Build Command:**
     ```bash
     pip install -r requirements.txt && python train_model.py
     ```
   - **Start Command:**
     ```bash
     gunicorn app:app --bind 0.0.0.0:$PORT
     ```
   - **Instance Type:** `Free`

### Step 3: Add Environment Variables
Under the **Environment Variables** section on Render, add:

| Key | Value | Notes |
|---|---|---|
| `SECRET_KEY` | *(Click "Generate" or enter a random secure string)* | Required for session security |
| `PYTHON_VERSION` | `3.11.9` | Ensures consistent Python build |

*(Optional)* If you want to connect a MySQL or PostgreSQL database, add:
- For PostgreSQL: `DATABASE_URL` = `your_render_postgres_internal_url`
- For MySQL: `DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`

### Step 4: Deploy!
Click **Create Web Service**. Render will:
1. Clone your repository.
2. Install dependencies from `requirements.txt`.
3. Train and calibrate the Random Forest model (`train_model.py`).
4. Start Gunicorn WSGI server.
5. Provide you with your live URL: `https://symptoscan-ai.onrender.com`.

---

## 🌐 Option 2: Render Blueprint (render.yaml)

Render supports Infrastructure-as-Code blueprints:
1. Go to [Render Blueprints](https://dashboard.render.com/blueprints).
2. Connect your repository.
3. Render will automatically read `render.yaml` and configure the build command, start command, health check (`/health`), and environment variables automatically!
4. Click **Apply**.

---

## 🔍 Verifying Health & APIs
Once deployed, check your live endpoints:
- **Web App:** `https://your-service.onrender.com/`
- **Health Check:** `https://your-service.onrender.com/health` (Returns `{"status": "healthy"}`)
- **API Symptoms:** `https://your-service.onrender.com/api/symptoms`
- **API Predict:** `POST https://your-service.onrender.com/api/predict`
  - Payload: `{"symptoms": ["High Fever", "Dry Cough", "Fatigue & Weakness"]}`

---

## 🛠️ Local Testing
To test locally on Windows before deploying:
```powershell
# Activate venv
.\venv\Scripts\activate

# Train model (if not already trained)
python train_model.py

# Run application
python app.py
```
Open `http://localhost:5000` in your web browser.
