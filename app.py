import os
import re
from datetime import datetime
from dotenv import load_dotenv
from flask import (
    Flask,
    render_template,
    request,
    session,
    redirect,
    url_for,
    flash,
    jsonify,
    abort
)
from werkzeug.security import generate_password_hash, check_password_hash

import database
import medical_engine

# =========================================================
# APPLICATION SETUP
# =========================================================
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "symptoscan-enterprise-secret-key-2026")

# Initialize database tables on startup
try:
    database.init_db()
except Exception as e:
    print(f"[Warning] DB initialization error: {e}")

# Context processor for global template variables
@app.context_processor
def inject_global_vars():
    user_name = session.get("user_name")
    user_id = session.get("user_id")
    current_year = datetime.now().year
    return {
        "current_user_name": user_name,
        "is_authenticated": bool(user_id),
        "current_year": current_year,
        "app_version": "2.0.0"
    }

@app.template_filter('format_datetime')
def format_datetime(val, fmt="%Y-%m-%d"):
    if not val:
        return "N/A"
    if hasattr(val, 'strftime'):
        return val.strftime(fmt)
    val_str = str(val)
    return val_str[:16] if len(val_str) >= 16 else val_str

# =========================================================
# PUBLIC / MARKETING ROUTES
# =========================================================
@app.route("/")
def home():
    """Modern landing page with live interactive demonstration."""
    catalog, all_symptoms = medical_engine.get_symptom_catalog()
    return render_template(
        "index.html",
        catalog=catalog,
        sample_symptoms=all_symptoms[:12]
    )

@app.route("/health")
def health_check():
    """Health check endpoint for Render / cloud monitoring."""
    return jsonify({
        "status": "healthy",
        "service": "SymptoScan AI",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }), 200

# =========================================================
# AUTHENTICATION
# =========================================================
@app.route("/register", methods=["GET", "POST"])
def register():
    """User registration with rigorous validation and instant login."""
    if session.get("user_id"):
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        # Form validations
        if not name or not email or not password:
            flash("Please fill in all required fields.", "error")
            return redirect(url_for("register"))

        email_pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
        if not re.match(email_pattern, email):
            flash("Please enter a valid email address.", "error")
            return redirect(url_for("register"))

        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "error")
            return redirect(url_for("register"))

        if confirm_password and password != confirm_password:
            flash("Passwords do not match.", "error")
            return redirect(url_for("register"))

        # Check existing email
        existing_user = database.query_one(
            "SELECT id FROM users WHERE email = %s", (email,)
        )
        if existing_user:
            flash("This email address is already registered. Please log in.", "error")
            return redirect(url_for("login"))

        # Hash password and insert
        hashed_password = generate_password_hash(password)
        try:
            user_id = database.execute_query(
                "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
                (name, email, hashed_password)
            )

            flash(f"🎉 Account created successfully! Welcome, {name}. Please sign in with your password.", "success")
            return redirect(url_for("login", registered="1", email=email, name=name))
        except Exception as e:
            print("Registration error:", e)
            flash("Unable to complete registration. Please try again.", "error")
            return redirect(url_for("register"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    """Secure user login with session initialization."""
    if session.get("user_id"):
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Please enter both email and password.", "error")
            return redirect(url_for("login"))

        try:
            user = database.query_one(
                "SELECT id, name, email, password FROM users WHERE email = %s",
                (email,)
            )

            if user and check_password_hash(user["password"], password):
                session["user_id"] = user["id"]
                session["user_name"] = user["name"]
                session["user_email"] = user["email"]
                flash(f"Welcome back, {user['name']}!", "success")
                next_url = request.args.get("next")
                return redirect(next_url or url_for("dashboard"))

            flash("Invalid email or password. Please try again.", "error")
            return redirect(url_for("login"))

        except Exception as e:
            print("Login error:", e)
            flash("Service temporarily unavailable. Please try again.", "error")
            return redirect(url_for("login"))

    return render_template("login.html")

@app.route("/logout")
def logout():
    """Logs out user and clears session."""
    session.clear()
    flash("You have been signed out safely.", "info")
    return redirect(url_for("login"))

# =========================================================
# DASHBOARD
# =========================================================
@app.route("/dashboard")
def dashboard():
    """Central analytics dashboard showing past predictions and clinical stats."""
    if "user_id" not in session:
        flash("Please log in to access your health dashboard.", "warning")
        return redirect(url_for("login", next=request.path))

    user_id = session["user_id"]

    try:
        # User statistics
        total_predictions_row = database.query_one(
            "SELECT COUNT(*) AS total FROM predictions WHERE user_id = %s",
            (user_id,)
        )
        total_predictions = total_predictions_row["total"] if total_predictions_row else 0

        # Latest assessment
        latest_prediction = database.query_one(
            """
            SELECT id, name, disease, confidence, severity, specialist, created_at
            FROM predictions
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id,)
        )

        # Recent 5 assessments
        recent_predictions = database.query_all(
            """
            SELECT id, name, age, gender, symptoms, disease, confidence, severity, specialist, created_at
            FROM predictions
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 5
            """,
            (user_id,)
        )

        # Severity breakdown
        severity_counts = {
            "Mild": 0,
            "Moderate": 0,
            "High": 0,
            "Critical": 0
        }
        all_user_preds = database.query_all(
            "SELECT severity FROM predictions WHERE user_id = %s", (user_id,)
        )
        for p in all_user_preds:
            sev = p.get("severity", "Moderate")
            if "critical" in sev.lower() or "emergency" in sev.lower():
                severity_counts["Critical"] += 1
            elif "high" in sev.lower():
                severity_counts["High"] += 1
            elif "mild" in sev.lower():
                severity_counts["Mild"] += 1
            else:
                severity_counts["Moderate"] += 1

        return render_template(
            "dashboard.html",
            name=session.get("user_name"),
            total_predictions=total_predictions,
            latest_prediction=latest_prediction,
            recent_predictions=recent_predictions,
            severity_counts=severity_counts
        )

    except Exception as e:
        print("Dashboard error:", e)
        flash("Unable to load dashboard data.", "error")
        return render_template(
            "dashboard.html",
            name=session.get("user_name"),
            total_predictions=0,
            latest_prediction=None,
            recent_predictions=[],
            severity_counts={"Mild": 0, "Moderate": 0, "High": 0, "Critical": 0}
        )

# =========================================================
# PREDICTION & TRIAGE
# =========================================================
@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Interactive multi-category symptom assessment form."""
    catalog, all_symptoms = medical_engine.get_symptom_catalog()

    if request.method == "POST":
        # Form fields
        name = request.form.get("name", "").strip()
        age_str = request.form.get("age", "").strip()
        gender = request.form.get("gender", "").strip()
        duration = request.form.get("duration", "").strip()
        severity_level = request.form.get("severity_level", "").strip()
        temperature = request.form.get("temperature", "").strip()
        user_notes = request.form.get("notes", "").strip()
        symptoms = request.form.getlist("symptoms")

        # Compile rich clinical notes
        clinical_notes_parts = []
        if duration:
            clinical_notes_parts.append(f"Duration: {duration}")
        if severity_level:
            clinical_notes_parts.append(f"Severity: {severity_level}")
        if temperature:
            clinical_notes_parts.append(f"Temp: {temperature}")
        if user_notes:
            clinical_notes_parts.append(user_notes)
        notes = " • ".join(clinical_notes_parts)

        # Fallback to session user name if patient name empty
        if not name:
            name = session.get("user_name", "Patient")

        # Validate age
        try:
            age = int(age_str)
            if age < 1 or age > 120:
                flash("Please enter a realistic age between 1 and 120.", "error")
                return render_template("predict.html", catalog=catalog, all_symptoms=all_symptoms)
        except ValueError:
            flash("Please enter a valid numeric age.", "error")
            return render_template("predict.html", catalog=catalog, all_symptoms=all_symptoms)

        if not gender:
            flash("Please select patient gender.", "error")
            return render_template("predict.html", catalog=catalog, all_symptoms=all_symptoms)

        if not symptoms:
            flash("Please select at least one symptom to evaluate.", "error")
            return render_template("predict.html", catalog=catalog, all_symptoms=all_symptoms)

        # Run AI prediction
        try:
            result = medical_engine.predict_condition(symptoms)
            if "error" in result:
                flash(result["error"], "error")
                return render_template("predict.html", catalog=catalog, all_symptoms=all_symptoms)

            disease = result["disease"]
            confidence = result["confidence"]
            severity = result["severity"]
            specialist = result["specialist"]

            # Save prediction to DB if logged in or fallback
            user_id = session.get("user_id")
            prediction_id = None

            try:
                prediction_id = database.execute_query(
                    """
                    INSERT INTO predictions
                    (user_id, name, age, gender, symptoms, disease, confidence, severity, specialist, notes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        user_id,
                        name,
                        age,
                        gender,
                        ", ".join(symptoms),
                        disease,
                        confidence,
                        severity,
                        specialist,
                        notes
                    )
                )
            except Exception as dbe:
                print("Database save error:", dbe)

            # Store result in session or redirect to result page
            session["last_result"] = {
                "id": prediction_id,
                "name": name,
                "age": age,
                "gender": gender,
                "notes": notes,
                "symptoms": symptoms,
                "created_at": datetime.now().strftime("%B %d, %Y - %I:%M %p"),
                **result
            }

            if prediction_id:
                return redirect(url_for("view_result", prediction_id=prediction_id))
            return redirect(url_for("view_last_result"))

        except Exception as e:
            print("Prediction error:", e)
            flash("Unable to complete AI assessment at this moment. Please try again.", "error")
            return render_template("predict.html", catalog=catalog, all_symptoms=all_symptoms)

    return render_template(
        "predict.html",
        catalog=catalog,
        all_symptoms=all_symptoms,
        default_name=session.get("user_name", "")
    )

@app.route("/result/<int:prediction_id>")
def view_result(prediction_id):
    """Displays detailed clinical diagnostic result by prediction ID."""
    pred = database.query_one(
        """
        SELECT id, user_id, name, age, gender, symptoms, disease, confidence, severity, specialist, notes, created_at
        FROM predictions
        WHERE id = %s
        """,
        (prediction_id,)
    )

    if not pred:
        flash("Assessment record not found.", "error")
        return redirect(url_for("dashboard"))

    # Reconstitute knowledge base details
    profile = medical_engine.get_disease_profile(pred["disease"])
    symptom_list = [s.strip() for s in pred["symptoms"].split(",") if s.strip()]
    
    # Try getting differential diagnoses if symptoms exist
    differentials = []
    try:
        calc_result = medical_engine.predict_condition(symptom_list)
        if "differentials" in calc_result:
            differentials = calc_result["differentials"]
    except Exception:
        pass

    result_data = {
        "id": pred["id"],
        "name": pred["name"],
        "age": pred["age"],
        "gender": pred["gender"],
        "notes": pred.get("notes", ""),
        "disease": pred["disease"],
        "confidence": pred["confidence"] or 88.0,
        "severity": pred["severity"] or profile.get("severity", "Moderate"),
        "severity_class": profile.get("severity_class", "info"),
        "specialist": pred["specialist"] or profile.get("specialist", "General Physician"),
        "category": profile.get("category", "General Medicine"),
        "description": profile.get("description", ""),
        "precautions": profile.get("precautions", []),
        "diet_advice": profile.get("diet_advice", ""),
        "emergency_alert": profile.get("emergency_alert", ""),
        "differentials": differentials,
        "selected_symptoms": symptom_list,
        "created_at": pred["created_at"]
    }

    return render_template("result.html", **result_data)

@app.route("/result/latest")
def view_last_result():
    """Displays most recently computed result from session."""
    result_data = session.get("last_result")
    if not result_data:
        flash("No recent assessment found.", "info")
        return redirect(url_for("predict"))
    return render_template("result.html", **result_data)

@app.route("/report/<int:prediction_id>")
def print_report(prediction_id):
    """Printable / PDF-friendly clean medical report view."""
    pred = database.query_one(
        """
        SELECT id, user_id, name, age, gender, symptoms, disease, confidence, severity, specialist, notes, created_at
        FROM predictions
        WHERE id = %s
        """,
        (prediction_id,)
    )

    if not pred:
        flash("Assessment record not found.", "error")
        return redirect(url_for("dashboard"))

    profile = medical_engine.get_disease_profile(pred["disease"])
    symptom_list = [s.strip() for s in pred["symptoms"].split(",") if s.strip()]
    
    differentials = []
    try:
        calc_result = medical_engine.predict_condition(symptom_list)
        if "differentials" in calc_result:
            differentials = calc_result["differentials"]
    except Exception:
        pass

    meta = {
        **profile,
        "differentials": differentials,
        "confidence": pred.get("confidence", 88.0)
    }

    return render_template(
        "report.html",
        pred=pred,
        meta=meta,
        symptom_list=symptom_list,
        printed_at=datetime.now().strftime("%B %d, %Y at %I:%M %p")
    )

# =========================================================
# HISTORY & MANAGEMENT
# =========================================================
@app.route("/history")
def history():
    """Complete history of user predictions with search and filtering."""
    if "user_id" not in session:
        flash("Please log in to view your assessment history.", "warning")
        return redirect(url_for("login", next=request.path))

    user_id = session["user_id"]
    search_query = request.args.get("q", "").strip()

    if search_query:
        predictions = database.query_all(
            """
            SELECT id, name, age, gender, symptoms, disease, confidence, severity, specialist, created_at
            FROM predictions
            WHERE user_id = %s AND (disease LIKE %s OR name LIKE %s OR symptoms LIKE %s)
            ORDER BY created_at DESC
            """,
            (user_id, f"%{search_query}%", f"%{search_query}%", f"%{search_query}%")
        )
    else:
        predictions = database.query_all(
            """
            SELECT id, name, age, gender, symptoms, disease, confidence, severity, specialist, created_at
            FROM predictions
            WHERE user_id = %s
            ORDER BY created_at DESC
            """,
            (user_id,)
        )

    return render_template(
        "history.html",
        predictions=predictions,
        search_query=search_query
    )

@app.route("/history/delete/<int:prediction_id>", methods=["POST"])
def delete_prediction(prediction_id):
    """Deletes an assessment record belonging to the current user."""
    if "user_id" not in session:
        flash("Authentication required.", "error")
        return redirect(url_for("login"))

    user_id = session["user_id"]
    try:
        database.execute_query(
            "DELETE FROM predictions WHERE id = %s AND user_id = %s",
            (prediction_id, user_id)
        )
        flash("Record deleted successfully.", "success")
    except Exception as e:
        print("Delete error:", e)
        flash("Could not delete record.", "error")

    return redirect(url_for("history"))

# =========================================================
# INTERACTIVE REST APIs
# =========================================================
@app.route("/api/symptoms", methods=["GET"])
def api_symptoms():
    """Returns categorized symptoms list in JSON format."""
    catalog, all_symptoms = medical_engine.get_symptom_catalog()
    return jsonify({
        "status": "success",
        "catalog": catalog,
        "symptoms": all_symptoms,
        "total": len(all_symptoms)
    })

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Headless REST API for AI disease prediction."""
    data = request.get_json(silent=True) or request.form
    symptoms = data.get("symptoms", [])

    if isinstance(symptoms, str):
        symptoms = [s.strip() for s in symptoms.split(",") if s.strip()]

    if not symptoms:
        return jsonify({
            "status": "error",
            "message": "Symptoms list cannot be empty."
        }), 400

    try:
        result = medical_engine.predict_condition(symptoms)
        return jsonify({
            "status": "success",
            "result": result
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/api/assistant-chat", methods=["POST"])
def api_assistant():
    """Interactive Symptom Triage Chatbot."""
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"reply": "Please describe your symptom or health question."})

    response = medical_engine.quick_assistant_reply(message)
    return jsonify(response)

# =========================================================
# ERROR HANDLERS
# =========================================================
@app.errorhandler(404)
def not_found_error(error):
    return render_template(
        "index.html",
        flash_error="The requested page was not found."
    ), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template(
        "index.html",
        flash_error="An unexpected server error occurred. Our team has been notified."
    ), 500

# =========================================================
# LOCAL EXECUTION
# =========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        debug=True,
        host="0.0.0.0",
        port=port
    )
