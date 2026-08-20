# SymptoScan AI 🩺🤖

> **AI-Powered Health Assistant for Educational Disease Prediction**

[🌐 Live Demo](https://symptoscan-production-ab95.up.railway.app) · [💻 GitHub Repository](https://github.com/smdanish03/SymptoScan)

SymptoScan is a web-based educational application that uses **Machine Learning** to analyze selected symptoms and predict a possible health condition.

The application is built using **Python Flask**, **MySQL**, and a **Decision Tree Classifier**. Users can create an account, log in securely, enter patient information, select symptoms, receive an AI-generated possible condition, and view their previous prediction history.

> ⚠️ **Disclaimer:** SymptoScan is an educational and demonstration project. It is **not a medical diagnosis system** and should not be used as a replacement for professional medical advice.

---

# 🚀 Live Demo

🌐 **Live Application:**
https://symptoscan-production-ab95.up.railway.app

The application is deployed using **Railway** with a **MySQL database**.

---

# ✨ Features

* 👤 User Registration
* 🔐 Secure Login System
* 🔑 Password Hashing using Werkzeug
* 🧠 Machine Learning-Based Disease Prediction
* 🌳 Decision Tree Classifier
* 🩺 Symptom-Based Analysis
* 📊 Prediction History
* 📈 Dashboard with Recent Prediction Information
* 🗄️ MySQL Database Integration
* 🔒 Environment Variables for Sensitive Credentials
* 📱 Responsive Web Interface
* ☁️ Cloud Deployment using Railway

---

# 🛠️ Technologies Used

## Backend

* Python
* Flask
* MySQL
* mysql-connector-python
* Werkzeug
* python-dotenv
* Gunicorn

## Machine Learning

* NumPy
* Pandas
* Scikit-learn
* Decision Tree Classifier

## Frontend

* HTML5
* CSS3
* Google Fonts
* Jinja2 Templates

## Deployment

* Railway
* Railway MySQL

---

# 🧠 Machine Learning Model

SymptoScan uses a **Decision Tree Classifier** to predict a possible health condition based on selected symptoms.

## Input Features

The current model uses six symptoms:

1. Fever
2. Cough
3. Headache
4. Vomiting
5. Fatigue
6. Body Pain

Each symptom is converted into a binary value:

```text
Selected     → 1
Not Selected → 0
```

The trained model is stored at:

```text
models/disease_model.pkl
```

The model is trained using:

```text
train_model.py
```

---

# 🩺 Example Conditions

The educational dataset contains example conditions such as:

* Flu
* Common Cold
* Food Poisoning
* Dengue
* Typhoid
* Allergy
* Viral Fever
* Migraine

> These predictions are based on a small educational dataset and should not be interpreted as clinically accurate diagnoses.

---

# 📁 Project Structure

```text
SymptoScan/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── dataset.csv
│
├── models/
│   └── disease_model.pkl
│
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── predict.html
│   ├── result.html
│   └── history.html
│
└── static/
    └── css/
        └── style.css
```

> The `.env` file and `venv` folder should not be uploaded to GitHub.

---

# ⚙️ Requirements

Make sure you have installed:

* Python 3.13+
* MySQL Server
* Git

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/smdanish03/SymptoScan.git
```

Move into the project folder:

```bash
cd SymptoScan
```

---

## 2. Create a Virtual Environment

### Windows

```bash
py -3.13 -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🗄️ MySQL Database Setup

Create a database:

```sql
CREATE DATABASE symptoscan;
```

Select the database:

```sql
USE symptoscan;
```

Create the `users` table:

```sql
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL
);
```

Create the `predictions` table:

```sql
CREATE TABLE predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    age INT NOT NULL,
    gender VARCHAR(20) NOT NULL,
    symptoms TEXT NOT NULL,
    disease VARCHAR(100) NOT NULL,
    user_id INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

---

# 🔐 Environment Variables

Create a `.env` file in the project root:

```env
SECRET_KEY=your_secret_key

DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_NAME=symptoscan
```

> ⚠️ **Never upload your `.env` file to GitHub.**

For deployment, configure the environment variables directly in your hosting platform.

---

# 🧠 Train the Machine Learning Model

If the model file does not exist, run:

```bash
python train_model.py
```

After successful training:

```text
Model trained successfully!
```

The trained model will be created at:

```text
models/disease_model.pkl
```

The training script loads the dataset, separates the `Disease` column as the target, trains a `DecisionTreeClassifier`, and saves the trained model.

---

# ▶️ Run the Application Locally

Activate the virtual environment:

```bash
venv\Scripts\activate
```

Run the application:

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

# 🔄 Application Flow

```text
User
 │
 ▼
Home Page
 │
 ▼
Register / Login
 │
 ▼
Dashboard
 │
 ▼
Enter Patient Information
 │
 ▼
Select Symptoms
 │
 ▼
Machine Learning Model
 │
 ▼
Possible Condition
 │
 ▼
Save Result in MySQL
 │
 ▼
Prediction History
```

---

# 🔒 Security Features

SymptoScan includes basic security practices:

* Password hashing using Werkzeug
* Environment variables for database credentials
* Session-based authentication
* Login-protected pages
* SQL parameterized queries
* Sensitive credentials excluded from GitHub using `.gitignore`

---

# 📱 Main Pages

## 🏠 Home

Introduces SymptoScan and provides access to prediction and account features.

## 📝 Register

Allows new users to create an account.

## 🔐 Login

Authenticates registered users.

## 📊 Dashboard

Displays information such as:

* Total predictions
* Latest prediction
* Account status
* Quick actions
* Recent activity

## 🩺 Prediction

Users provide:

* Name
* Age
* Gender
* Symptoms

Selected symptoms are converted into machine learning model features.

## 📋 Result

Displays the predicted possible condition and submitted patient information.

## 📜 History

Displays previous prediction records for the logged-in user.

---

# ⚠️ Important Disclaimer

SymptoScan is **not a medical device** and does not provide medical diagnosis, treatment, or professional medical advice.

Predictions are generated from a small educational dataset and are intended only to demonstrate how a machine-learning-based web application can work.

For real health concerns, consult a qualified healthcare professional.

---

# 🔮 Future Improvements

Possible future improvements include:

* Larger and clinically validated datasets
* More symptoms and conditions
* Multiple machine learning algorithms
* Model accuracy comparison
* Prediction confidence visualization
* Doctor consultation module
* Appointment booking
* Medical information resources
* Admin dashboard
* Better data privacy controls
* REST API
* Improved mobile UI

---

# 👨‍💻 Author

**Mohd Danish Shaikh**

Computer Engineering Student
Aspiring Software Engineer

* GitHub: https://github.com/smdanish03
* LinkedIn: https://www.linkedin.com/in/danish-shaikh-6544a9361

---

# 📄 License

This project is created for **educational and academic purposes**.

You are welcome to study, modify, and improve the project for learning and demonstration purposes.
