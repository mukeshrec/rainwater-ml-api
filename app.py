# app.py

from flask import Flask, request, jsonify
from flasgger import Swagger
import joblib
import json
import numpy as np

app = Flask(__name__)
swagger = Swagger(app)

# ==============================
# Load models, encoders & features
# ==============================
clf_feasibility = joblib.load("model_feasibility.pkl")
clf_structure = joblib.load("model_structure.pkl")
reg_runoff = joblib.load("model_runoff.pkl")
reg_infil = joblib.load("model_infiltration.pkl")
label_encoders = joblib.load("label_encoders.pkl")

with open("features.json", "r") as f:
    FEATURES = json.load(f)

# Encoders
le_roof_type = label_encoders["roof_type"]
le_structure = label_encoders["recommended_structure"]
le_feas = label_encoders["feasibility_classification"]


# ==============================
# Helper for feature vector
# ==============================
def build_feature_vector(data: dict):
    missing = [f for f in FEATURES if f not in data and f != "roof_type"]
    if "roof_type" not in data:
        missing.append("roof_type")

    if missing:
        return None, {"error": f"Missing fields: {', '.join(missing)}"}

    # Encode roof_type
    try:
        roof_type_encoded = int(le_roof_type.transform([str(data["roof_type"])])[0])
    except ValueError:
        return None, {
            "error": f"Unknown roof_type '{data['roof_type']}'. "
                     f"Valid: {list(le_roof_type.classes_)}"
        }

    feature_values = []
    for feat in FEATURES:
        if feat == "roof_type":
            feature_values.append(roof_type_encoded)
        else:
            feature_values.append(float(data[feat]))

    return np.array(feature_values).reshape(1, -1), None


# ==============================
# Health Check Route
# ==============================
@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "message": "Rainwater ML API Running"})


# ==============================
# PREDICT API with Swagger Docs
# ==============================
@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict Rainwater Harvesting Outputs
    ---
    tags:
      - Rainwater ML API
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            latitude:
              type: number
            longitude:
              type: number
            roof_area:
              type: number
            open_space:
              type: number
            roof_type:
              type: string
            annual_rainfall:
              type: number
            max_daily_rainfall:
              type: number
            clay:
              type: number
            sand:
              type: number
            silt:
              type: number
            elevation:
              type: number
            evaporation:
              type: number
    responses:
      200:
        description: Predictions returned successfully
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    X, err = build_feature_vector(data)
    if err:
        return jsonify(err), 400

    # Predictions
    fea_pred = le_feas.inverse_transform([clf_feasibility.predict(X)[0]])[0]
    struct_pred = le_structure.inverse_transform([clf_structure.predict(X)[0]])[0]
    runoff_pred = float(reg_runoff.predict(X)[0])
    infil_pred = float(reg_infil.predict(X)[0])

    return jsonify({
        "feasibility": fea_pred,
        "recommended_structure": struct_pred,
        "annual_runoff": runoff_pred,
        "infiltration": infil_pred
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

