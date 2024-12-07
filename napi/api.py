from flask import Flask, request, jsonify
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)

# Load the pre-trained models and scaler
scaler = pickle.load(open("scaler.pkl", "rb"))
decision_tree_model = pickle.load(open("decision_tree_model.pkl", "rb"))
logistic_regression_model = pickle.load(open("logistic_regression_model.pkl", "rb"))
random_forest_model = pickle.load(open("random_forest_model.pkl", "rb"))


@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Fertilizer API"})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        input_data = request.get_json()
        model_type = input_data.get("model", "random_forest")  # Default model
        features = input_data.get("features")  # List of features

        # Validate input
        if not features or len(features) != 8:
            return jsonify({"error": "Invalid input. Provide a list of 8 features."}), 400

        # Transform features using the scaler
        features = scaler.transform([features])  # Ensure input is a 2D array

        # Select the model and make prediction
        if model_type == "decision_tree":
            prediction = decision_tree_model.predict(features)
        elif model_type == "logistic_regression":
            prediction = logistic_regression_model.predict(features)
        elif model_type == "random_forest":
            prediction = random_forest_model.predict(features)
        else:
            return jsonify({"error": "Invalid model type. Choose 'decision_tree', 'logistic_regression', or 'random_forest'."}), 400

        # Return the prediction
        return jsonify({"model": model_type, "prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
