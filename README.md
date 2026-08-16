# SymptoScan AI 🩺🤖

> **AI-Powered Health Assistant for Educational Disease Prediction**

SymptoScan is a web-based educational project that uses **Machine Learning** to analyze selected symptoms and predict a possible health condition.

The application is built using **Python Flask**, **MySQL**, and a **Decision Tree Classifier**. Users can create an account, log in, enter basic information, select symptoms, receive an AI-generated possible condition, and view their previous prediction history.

⚠️ **Disclaimer:** SymptoScan is an educational and demonstration project. It is **not a medical diagnosis system** and should not be used as a replacement for professional medical advice.

---

## 🚀 Features

* 👤 User Registration
* 🔐 Secure Login
* 🔑 Password Hashing using Werkzeug
* 🧠 Machine Learning Disease Prediction
* 🌳 Decision Tree Classifier
* 🩺 Symptom-Based Analysis
* 📊 Prediction History
* 🗄️ MySQL Database Integration
* 📱 Responsive Web Interface
* 🔒 Environment Variables using `.env`
* 📈 Dashboard with recent prediction information

---

## 🛠️ Technologies Used

### Backend

* Python
* Flask
* MySQL
* mysql-connector-python
* Werkzeug
* python-dotenv

### Machine Learning

* NumPy
* Pandas
* Scikit-learn
* Decision Tree Classifier

### Frontend

* HTML5
* CSS3
* Google Fonts
* Jinja2 Templates

---

## 🧠 Machine Learning Model

SymptoScan uses a **Decision Tree Classifier** to predict a possible condition from selected symptoms.

### Input Features

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

The trained model is stored as:

```text
models/disease_model.pkl
```

The model is trained using `train_model.py`.

---

## 🩺 Example Conditions

The current educational dataset contains examples such as:

* Flu
* Common Cold
* Food Poisoning
* Dengue
* Typhoid
* Allergy
* Viral Fever
* Migraine

> These predictions are based only on the small educational dataset included in this project and should not be interpreted as clinically accurate diagnoses.

---

## 📁 Project Structure

```text
SymptoScan/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env
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
├── static/
│   └── css/
│       └── style.css
│
└── venv/
```

> `venv/` and `.env` should **not** be uploaded to GitHub.

---

## ⚙️ Requirements

Make sure you have installed:

* Python 3.13+
* MySQL Server
* Git
* VS Code

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/SymptoScan.git
```

Go inside the project:

```bash
cd SymptoScan
```

---

### 2. Create a Virtual Environment

Windows:

```bash
py -3.13 -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The main dependencies are:

```text
Flask
mysql-connector-python
Werkzeug
numpy
pandas
scikit-learn
python-dotenv
```

---

## 🗄️ MySQL Database Setup

Create a MySQL database:

```sql
CREATE DATABASE symptoscan;
```

Create the users table:

```sql
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(150) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL
);
```

Create the predictions table:

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

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
SECRET_KEY=your_secret_key

DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_NAME=symptoscan
```

**Never upload `.env` to GitHub.**

The application reads these values using `python-dotenv`.

---

## 🧠 Train the Machine Learning Model

If the model file does not exist, run:

```bash
python train_model.py
```

You should see:

```text
Model trained successfully!
```

This creates:

```text
models/disease_model.pkl
```

The training script loads `data/dataset.csv`, separates the `Disease` column as the target, trains a `DecisionTreeClassifier`, and saves the model in the `models` directory.

---

## ▶️ Run the Application

Activate the virtual environment:

```bash
venv\Scripts\activate
```

Run Flask:

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## 🔄 Application Flow

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

## 🔒 Security

SymptoScan includes basic security practices such as:

* Password hashing using Werkzeug
* Environment variables for database credentials
* Session-based authentication
* Login-protected prediction pages
* Login-protected prediction history
* SQL parameterized queries

---

## 📸 Main Pages

### Home

Introduces SymptoScan and provides access to prediction and account creation.

### Register

Allows a new user to create an account.

### Login

Authenticates registered users.

### Dashboard

Displays:

* Total predictions
* Latest prediction
* Account status
* Quick actions
* Recent activity

### Prediction

Users enter:

* Name
* Age
* Gender
* Symptoms

The selected symptoms are converted into model features before prediction.

### Result

Displays the predicted possible condition and patient information.

### History

Displays previous prediction records saved for the logged-in user.

---

## ⚠️ Important Disclaimer

SymptoScan is **not a medical device** and does not provide medical diagnosis, treatment, or professional medical advice.

The predictions are generated from a small educational dataset and are intended only to demonstrate how a machine-learning-based web application can work.

For real health concerns, consult a qualified healthcare professional.

---

## 🔮 Future Improvements

Possible future improvements include:

* Larger and clinically validated dataset
* More symptoms
* More machine learning algorithms
* Model accuracy comparison
* Prediction confidence visualization
* Doctor consultation module
* Appointment booking
* Medical information resources
* Admin dashboard
* Better data privacy controls
* Cloud deployment
* REST API
* Improved mobile UI

---

## 👨‍💻 Author

## Mohd Danish Shaikh **

Computer Engineering Student
Aspiring Software Engineer

---

## 📄 License

This project is created for **educational and academic purposes**.

You may modify and improve it for learning and demonstration.
