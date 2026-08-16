from flask import (
    Flask,
    render_template,
    request,
    session,
    redirect,
    url_for,
    flash
)

import pickle
import mysql.connector
from mysql.connector import Error
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

import os
import re


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)

app.secret_key = os.getenv("SECRET_KEY")


# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_db_connection():

    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )


# =========================================================
# LOAD MACHINE LEARNING MODEL
# =========================================================

with open("models/disease_model.pkl", "rb") as file:
    model = pickle.load(file)


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template("index.html")


# =========================================================
# REGISTER
# =========================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        # -------------------------------------------------
        # Get form data
        # -------------------------------------------------

        name = request.form.get("name", "").strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )


        # -------------------------------------------------
        # Basic validation
        # -------------------------------------------------

        if not name or not email or not password:

            flash(
                "Please fill all fields.",
                "error"
            )

            return redirect(
                url_for("register")
            )


        # -------------------------------------------------
        # Email validation
        # -------------------------------------------------

        email_pattern = (
            r"^[A-Za-z0-9._%+-]+"
            r"@[A-Za-z0-9.-]+\."
            r"[A-Za-z]{2,}$"
        )

        if not re.match(
            email_pattern,
            email
        ):

            flash(
                "Please enter a valid email address.",
                "error"
            )

            return redirect(
                url_for("register")
            )


        # -------------------------------------------------
        # Password validation
        # -------------------------------------------------

        if len(password) < 6:

            flash(
                "Password must contain at least 6 characters.",
                "error"
            )

            return redirect(
                url_for("register")
            )


        db = None
        cursor = None


        try:

            # -------------------------------------------------
            # Connect to MySQL
            # -------------------------------------------------

            db = get_db_connection()

            cursor = db.cursor()


            # -------------------------------------------------
            # Check existing email
            # -------------------------------------------------

            cursor.execute(
                """
                SELECT id
                FROM users
                WHERE email = %s
                """,
                (email,)
            )

            existing_user = cursor.fetchone()


            if existing_user:

                flash(
                    "This email is already registered.",
                    "error"
                )

                return redirect(
                    url_for("register")
                )


            # -------------------------------------------------
            # Hash password
            # -------------------------------------------------

            hashed_password = generate_password_hash(
                password
            )


            # -------------------------------------------------
            # Insert user
            # -------------------------------------------------

            cursor.execute(
                """
                INSERT INTO users
                (name, email, password)
                VALUES (%s, %s, %s)
                """,
                (
                    name,
                    email,
                    hashed_password
                )
            )


            db.commit()


            # -------------------------------------------------
            # Success message
            # -------------------------------------------------

            flash(
                "Registration successful! You can now login.",
                "success"
            )

            return redirect(
                url_for("login")
            )


        except Error as e:

            print(
                "Registration error:",
                e
            )

            if db:
                db.rollback()

            flash(
                "Registration failed. Please try again.",
                "error"
            )

            return redirect(
                url_for("register")
            )


        finally:

            if cursor:
                cursor.close()

            if db:
                db.close()


    return render_template(
        "register.html"
    )


# =========================================================
# LOGIN
# =========================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )


        if not email or not password:

            flash(
                "Please enter email and password.",
                "error"
            )

            return redirect(
                url_for("login")
            )


        db = None
        cursor = None


        try:

            # -------------------------------------------------
            # Connect to database
            # -------------------------------------------------

            db = get_db_connection()

            cursor = db.cursor(
                dictionary=True
            )


            # -------------------------------------------------
            # Find user
            # -------------------------------------------------

            cursor.execute(
                """
                SELECT id, name, email, password
                FROM users
                WHERE email = %s
                """,
                (email,)
            )


            user = cursor.fetchone()


            # -------------------------------------------------
            # Verify password
            # -------------------------------------------------

            if user and check_password_hash(
                user["password"],
                password
            ):

                session["user_id"] = user["id"]

                session["user_name"] = user["name"]

                return redirect(
                    url_for("dashboard")
                )


            flash(
                "Invalid email or password.",
                "error"
            )

            return redirect(
                url_for("login")
            )


        except Error as e:

            print(
                "Login error:",
                e
            )

            flash(
                "Unable to connect to the database.",
                "error"
            )

            return redirect(
                url_for("login")
            )


        finally:

            if cursor:
                cursor.close()

            if db:
                db.close()


    return render_template(
        "login.html"
    )


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/dashboard")
def dashboard():

    # -------------------------------------------------
    # Login required
    # -------------------------------------------------

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    db = None
    cursor = None


    try:

        db = get_db_connection()

        cursor = db.cursor(
            dictionary=True
        )


        # -------------------------------------------------
        # Total predictions
        # -------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(*) AS total_predictions
            FROM predictions
            WHERE user_id = %s
            """,
            (session["user_id"],)
        )


        total_result = cursor.fetchone()

        total_predictions = (
            total_result["total_predictions"]
        )


        # -------------------------------------------------
        # Latest prediction
        # -------------------------------------------------

        cursor.execute(
            """
            SELECT disease, created_at
            FROM predictions
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (session["user_id"],)
        )


        latest_prediction = cursor.fetchone()


        return render_template(
            "dashboard.html",
            name=session["user_name"],
            total_predictions=total_predictions,
            latest_prediction=latest_prediction
        )


    except Error as e:

        print(
            "Dashboard error:",
            e
        )

        flash(
            "Unable to load dashboard.",
            "error"
        )

        return redirect(
            url_for("login")
        )


    finally:

        if cursor:
            cursor.close()

        if db:
            db.close()


# =========================================================
# PREDICTION
# =========================================================

@app.route(
    "/predict",
    methods=["GET", "POST"]
)
def predict():

    # -------------------------------------------------
    # Login required
    # -------------------------------------------------

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    if request.method == "POST":

        # -------------------------------------------------
        # Get form data
        # -------------------------------------------------

        name = request.form.get(
            "name",
            ""
        ).strip()


        age = request.form.get(
            "age",
            ""
        ).strip()


        gender = request.form.get(
            "gender",
            ""
        ).strip()


        symptoms = request.form.getlist(
            "symptoms"
        )


        # -------------------------------------------------
        # Validate name
        # -------------------------------------------------

        if not name:

            flash(
                "Please enter your name.",
                "error"
            )

            return redirect(
                url_for("predict")
            )


        # -------------------------------------------------
        # Validate age
        # -------------------------------------------------

        if not age:

            flash(
                "Please enter your age.",
                "error"
            )

            return redirect(
                url_for("predict")
            )


        try:

            age = int(age)

        except ValueError:

            flash(
                "Age must be a valid number.",
                "error"
            )

            return redirect(
                url_for("predict")
            )


        if age < 1 or age > 120:

            flash(
                "Please enter a valid age between 1 and 120.",
                "error"
            )

            return redirect(
                url_for("predict")
            )


        # -------------------------------------------------
        # Validate gender
        # -------------------------------------------------

        if not gender:

            flash(
                "Please select your gender.",
                "error"
            )

            return redirect(
                url_for("predict")
            )


        # -------------------------------------------------
        # Validate symptoms
        # -------------------------------------------------

        if not symptoms:

            flash(
                "Please select at least one symptom.",
                "error"
            )

            return redirect(
                url_for("predict")
            )


        # -------------------------------------------------
        # Convert symptoms to model values
        # -------------------------------------------------

        fever = (
            1 if "Fever" in symptoms else 0
        )

        cough = (
            1 if "Cough" in symptoms else 0
        )

        headache = (
            1 if "Headache" in symptoms else 0
        )

        vomiting = (
            1 if "Vomiting" in symptoms else 0
        )

        fatigue = (
            1 if "Fatigue" in symptoms else 0
        )

        bodypain = (
            1 if "Body Pain" in symptoms else 0
        )


        # -------------------------------------------------
        # AI prediction
        # -------------------------------------------------

        try:

            prediction = model.predict(
                [[
                    fever,
                    cough,
                    headache,
                    vomiting,
                    fatigue,
                    bodypain
                ]]
            )

            disease = str(
                prediction[0]
            )


        except Exception as e:

            print(
                "Prediction error:",
                e
            )

            flash(
                "Unable to make prediction.",
                "error"
            )

            return redirect(
                url_for("predict")
            )


        # -------------------------------------------------
        # Save prediction
        # -------------------------------------------------

        db = None
        cursor = None


        try:

            db = get_db_connection()

            cursor = db.cursor()


            cursor.execute(
                """
                INSERT INTO predictions
                (
                    name,
                    age,
                    gender,
                    symptoms,
                    disease,
                    user_id
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    name,
                    age,
                    gender,
                    ", ".join(symptoms),
                    disease,
                    session["user_id"]
                )
            )


            db.commit()


        except Error as e:

            print(
                "Prediction database error:",
                e
            )

            if db:
                db.rollback()

            flash(
                "Prediction was made but could not be saved.",
                "error"
            )

            return redirect(
                url_for("predict")
            )


        finally:

            if cursor:
                cursor.close()

            if db:
                db.close()


        # -------------------------------------------------
        # Show prediction result
        # -------------------------------------------------

        return render_template(
            "result.html",
            disease=disease,
            name=name,
            age=age,
            gender=gender
        )


    return render_template(
        "predict.html"
    )


# =========================================================
# HISTORY
# =========================================================

@app.route("/history")
def history():

    # -------------------------------------------------
    # Login required
    # -------------------------------------------------

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    db = None
    cursor = None


    try:

        db = get_db_connection()

        cursor = db.cursor(
            dictionary=True
        )


        cursor.execute(
            """
            SELECT
                id,
                name,
                age,
                gender,
                symptoms,
                disease,
                created_at
            FROM predictions
            WHERE user_id = %s
            ORDER BY created_at DESC
            """,
            (session["user_id"],)
        )


        predictions = cursor.fetchall()


        return render_template(
            "history.html",
            predictions=predictions
        )


    except Error as e:

        print(
            "History error:",
            e
        )

        flash(
            "Unable to load prediction history.",
            "error"
        )

        return redirect(
            url_for("dashboard")
        )


    finally:

        if cursor:
            cursor.close()

        if db:
            db.close()


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    session.clear()

    flash(
        "You have been logged out successfully.",
        "success"
    )

    return redirect(
        url_for("login")
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
    from flask import (
    Flask,
    render_template,
    request,
    session,
    redirect,
    url_for,
    flash
)

import pickle
import mysql.connector
from mysql.connector import Error
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

import os
import re


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)

app.secret_key = os.getenv("SECRET_KEY")


# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_db_connection():

    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )


# =========================================================
# LOAD MACHINE LEARNING MODEL
# =========================================================

with open("models/disease_model.pkl", "rb") as file:
    model = pickle.load(file)


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template("index.html")


# =========================================================
# REGISTER
# =========================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        # -------------------------------------------------
        # Get form data
        # -------------------------------------------------

        name = request.form.get("name", "").strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )


        # -------------------------------------------------
        # Basic validation
        # -------------------------------------------------

        if not name or not email or not password:

            flash(
                "Please fill all fields.",
                "error"
            )

            return redirect(
                url_for("register")
            )


        # -------------------------------------------------
        # Email validation
        # -------------------------------------------------

        email_pattern = (
            r"^[A-Za-z0-9._%+-]+"
            r"@[A-Za-z0-9.-]+\."
            r"[A-Za-z]{2,}$"
        )

        if not re.match(
            email_pattern,
            email
        ):

            flash(
                "Please enter a valid email address.",
                "error"
            )

            return redirect(
                url_for("register")
            )


        # -------------------------------------------------
        # Password validation
        # -------------------------------------------------

        if len(password) < 6:

            flash(
                "Password must contain at least 6 characters.",
                "error"
            )

            return redirect(
                url_for("register")
            )


        db = None
        cursor = None


        try:

            # -------------------------------------------------
            # Connect to MySQL
            # -------------------------------------------------

            db = get_db_connection()

            cursor = db.cursor()


            # -------------------------------------------------
            # Check existing email
            # -------------------------------------------------

            cursor.execute(
                """
                SELECT id
                FROM users
                WHERE email = %s
                """,
                (email,)
            )

            existing_user = cursor.fetchone()


            if existing_user:

                flash(
                    "This email is already registered.",
                    "error"
                )

                return redirect(
                    url_for("register")
                )


            # -------------------------------------------------
            # Hash password
            # -------------------------------------------------

            hashed_password = generate_password_hash(
                password
            )


            # -------------------------------------------------
            # Insert user
            # -------------------------------------------------

            cursor.execute(
                """
                INSERT INTO users
                (name, email, password)
                VALUES (%s, %s, %s)
                """,
                (
                    name,
                    email,
                    hashed_password
                )
            )


            db.commit()


            # -------------------------------------------------
            # Success message
            # -------------------------------------------------

            flash(
                "Registration successful! You can now login.",
                "success"
            )

            return redirect(
                url_for("login")
            )


        except Error as e:

            print(
                "Registration error:",
                e
            )

            if db:
                db.rollback()

            flash(
                "Registration failed. Please try again.",
                "error"
            )

            return redirect(
                url_for("register")
            )


        finally:

            if cursor:
                cursor.close()

            if db:
                db.close()


    return render_template(
        "register.html"
    )


# =========================================================
# LOGIN
# =========================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )


        if not email or not password:

            flash(
                "Please enter email and password.",
                "error"
            )

            return redirect(
                url_for("login")
            )


        db = None
        cursor = None


        try:

            # -------------------------------------------------
            # Connect to database
            # -------------------------------------------------

            db = get_db_connection()

            cursor = db.cursor(
                dictionary=True
            )


            # -------------------------------------------------
            # Find user
            # -------------------------------------------------

            cursor.execute(
                """
                SELECT id, name, email, password
                FROM users
                WHERE email = %s
                """,
                (email,)
            )


            user = cursor.fetchone()


            # -------------------------------------------------
            # Verify password
            # -------------------------------------------------

            if user and check_password_hash(
                user["password"],
                password
            ):

                session["user_id"] = user["id"]

                session["user_name"] = user["name"]

                return redirect(
                    url_for("dashboard")
                )


            flash(
                "Invalid email or password.",
                "error"
            )

            return redirect(
                url_for("login")
            )


        except Error as e:

            print(
                "Login error:",
                e
            )

            flash(
                "Unable to connect to the database.",
                "error"
            )

            return redirect(
                url_for("login")
            )


        finally:

            if cursor:
                cursor.close()

            if db:
                db.close()


    return render_template(
        "login.html"
    )


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/dashboard")
def dashboard():

    # -------------------------------------------------
    # Login required
    # -------------------------------------------------

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    db = None
    cursor = None


    try:

        db = get_db_connection()

        cursor = db.cursor(
            dictionary=True
        )


        # -------------------------------------------------
        # Total predictions
        # -------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(*) AS total_predictions
            FROM predictions
            WHERE user_id = %s
            """,
            (session["user_id"],)
        )


        total_result = cursor.fetchone()

        total_predictions = (
            total_result["total_predictions"]
        )


        # -------------------------------------------------
        # Latest prediction
        # -------------------------------------------------

        cursor.execute(
            """
            SELECT disease, created_at
            FROM predictions
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (session["user_id"],)
        )


        latest_prediction = cursor.fetchone()


        return render_template(
            "dashboard.html",
            name=session["user_name"],
            total_predictions=total_predictions,
            latest_prediction=latest_prediction
        )


    except Error as e:

        print(
            "Dashboard error:",
            e
        )

        flash(
            "Unable to load dashboard.",
            "error"
        )

        return redirect(
            url_for("login")
        )


    finally:

        if cursor:
            cursor.close()

        if db:
            db.close()


# =========================================================
# PREDICTION
# =========================================================

@app.route(
    "/predict",
    methods=["GET", "POST"]
)
def predict():

    # -------------------------------------------------
    # Login required
    # -------------------------------------------------

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    if request.method == "POST":

        # -------------------------------------------------
        # Get form data
        # -------------------------------------------------

        name = request.form.get(
            "name",
            ""
        ).strip()


        age = request.form.get(
            "age",
            ""
        ).strip()


        gender = request.form.get(
            "gender",
            ""
        ).strip()


        symptoms = request.form.getlist(
            "symptoms"
        )


        # -------------------------------------------------
        # Validate name
        # -------------------------------------------------

        if not name:

            flash(
                "Please enter your name.",
                "error"
            )

            return redirect(
                url_for("predict")
            )


        # -------------------------------------------------
        # Validate age
        # -------------------------------------------------

        if not age:

            flash(
                "Please enter your age.",
                "error"
            )

            return redirect(
                url_for("predict")
            )


        try:

            age = int(age)

        except ValueError:

            flash(
                "Age must be a valid number.",
                "error"
            )

            return redirect(
                url_for("predict")
            )


        if age < 1 or age > 120:

            flash(
                "Please enter a valid age between 1 and 120.",
                "error"
            )

            return redirect(
                url_for("predict")
            )


        # -------------------------------------------------
        # Validate gender
        # -------------------------------------------------

        if not gender:

            flash(
                "Please select your gender.",
                "error"
            )

            return redirect(
                url_for("predict")
            )


        # -------------------------------------------------
        # Validate symptoms
        # -------------------------------------------------

        if not symptoms:

            flash(
                "Please select at least one symptom.",
                "error"
            )

            return redirect(
                url_for("predict")
            )


        # -------------------------------------------------
        # Convert symptoms to model values
        # -------------------------------------------------

        fever = (
            1 if "Fever" in symptoms else 0
        )

        cough = (
            1 if "Cough" in symptoms else 0
        )

        headache = (
            1 if "Headache" in symptoms else 0
        )

        vomiting = (
            1 if "Vomiting" in symptoms else 0
        )

        fatigue = (
            1 if "Fatigue" in symptoms else 0
        )

        bodypain = (
            1 if "Body Pain" in symptoms else 0
        )


        # -------------------------------------------------
        # AI prediction
        # -------------------------------------------------

        try:

            prediction = model.predict(
                [[
                    fever,
                    cough,
                    headache,
                    vomiting,
                    fatigue,
                    bodypain
                ]]
            )

            disease = str(
                prediction[0]
            )


        except Exception as e:

            print(
                "Prediction error:",
                e
            )

            flash(
                "Unable to make prediction.",
                "error"
            )

            return redirect(
                url_for("predict")
            )


        # -------------------------------------------------
        # Save prediction
        # -------------------------------------------------

        db = None
        cursor = None


        try:

            db = get_db_connection()

            cursor = db.cursor()


            cursor.execute(
                """
                INSERT INTO predictions
                (
                    name,
                    age,
                    gender,
                    symptoms,
                    disease,
                    user_id
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    name,
                    age,
                    gender,
                    ", ".join(symptoms),
                    disease,
                    session["user_id"]
                )
            )


            db.commit()


        except Error as e:

            print(
                "Prediction database error:",
                e
            )

            if db:
                db.rollback()

            flash(
                "Prediction was made but could not be saved.",
                "error"
            )

            return redirect(
                url_for("predict")
            )


        finally:

            if cursor:
                cursor.close()

            if db:
                db.close()


        # -------------------------------------------------
        # Show prediction result
        # -------------------------------------------------

        return render_template(
            "result.html",
            disease=disease,
            name=name,
            age=age,
            gender=gender
        )


    return render_template(
        "predict.html"
    )


# =========================================================
# HISTORY
# =========================================================

@app.route("/history")
def history():

    # -------------------------------------------------
    # Login required
    # -------------------------------------------------

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    db = None
    cursor = None


    try:

        db = get_db_connection()

        cursor = db.cursor(
            dictionary=True
        )


        cursor.execute(
            """
            SELECT
                id,
                name,
                age,
                gender,
                symptoms,
                disease,
                created_at
            FROM predictions
            WHERE user_id = %s
            ORDER BY created_at DESC
            """,
            (session["user_id"],)
        )


        predictions = cursor.fetchall()


        return render_template(
            "history.html",
            predictions=predictions
        )


    except Error as e:

        print(
            "History error:",
            e
        )

        flash(
            "Unable to load prediction history.",
            "error"
        )

        return redirect(
            url_for("dashboard")
        )


    finally:

        if cursor:
            cursor.close()

        if db:
            db.close()


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    session.clear()

    flash(
        "You have been logged out successfully.",
        "success"
    )

    return redirect(
        url_for("login")
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )