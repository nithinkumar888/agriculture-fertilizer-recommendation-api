
import os
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

API_SECRET_KEY = os.environ.get("API_SECRET_KEY")

# Load dataset
data = pd.read_csv("fertilizer_recommendation.csv")

# Encode categorical features
le_soil = LabelEncoder()
data["Soil Type"] = le_soil.fit_transform(data["Soil Type"])

le_crop = LabelEncoder()
data["Crop Type"] = le_crop.fit_transform(data["Crop Type"])

# Prepare and train model
X = data.iloc[:, :8]
y = data.iloc[:, -1]

model = DecisionTreeClassifier(random_state=0)
model.fit(X, y)

@app.route("/")
def home():
    return jsonify({"message": "Fertilizer Recommendation API is running"})

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model_loaded": True})

@app.route("/predict", methods=["POST"])
def predict():
    if not API_SECRET_KEY or request.headers.get("X-API-Key") != API_SECRET_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        values = request.get_json(force=True)

        soil = le_soil.transform([values["soil"]])[0]
        crop = le_crop.transform([values["crop"]])[0]

        user_input = [[
            float(values["temperature"]),
            float(values["humidity"]),
            float(values["soilMoisture"]),
            soil,
            crop,
            float(values["n"]),
            float(values["k"]),
            float(values["p"])
        ]]

        result = model.predict(user_input)[0]

        return jsonify({"recommended_fertilizer": str(result)})

    except ValueError:
        return jsonify({"error": "Invalid input or unknown soil/crop type"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
