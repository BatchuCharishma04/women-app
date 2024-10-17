from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('risk_model.pkl')

# Function to predict risk percentage
@app.route('/predict_risk', methods=['POST'])
def predict_risk():
    data = request.json
    latitude = data['latitude']
    longitude = data['longitude']
    
    # For simplicity, we'll assume the risk model uses lat/lon to predict
    # Create the feature vector as per the model
    input_features = np.array([[latitude, longitude]])

    # Predict risk using the model
    risk_percentage = model.predict_proba(input_features)[0][1] * 100
    
    return jsonify({"risk_percentage": risk_percentage})

if __name__ == '_main_':
    app.run(debug=True)