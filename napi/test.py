import pickle
import numpy as np

# Load models and scaler
scaler = pickle.load(open("scaler.pkl", "rb"))
random_forest_model = pickle.load(open("random_forest_model.pkl", "rb"))

# Example input
features = [[75, 30, 10, 6.5, 150, 40, 35, 25]]
scaled_features = scaler.transform(features)

# Predict
prediction = random_forest_model.predict(scaled_features)
print(f"Prediction: {prediction[0]}")
