import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle
import os

# Load dataset
data = pd.read_csv("data/dataset.csv")

# Features
X = data.drop("Disease", axis=1)

# Target
y = data["Disease"]

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save model
with open("models/disease_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model trained successfully!")