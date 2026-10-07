#!/usr/bin/env bash
# Exit on any error
set -o errexit

echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🧠 Training AI Disease Prediction Model & Calibrator..."
python train_model.py

echo "✅ Build completed successfully!"
