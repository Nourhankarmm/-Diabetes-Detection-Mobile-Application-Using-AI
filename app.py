from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

model_file_path = 'diabetes_model.sav'
with open(model_file_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "Welcome to the Diabetes Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    print(data["gender"], data["age"], data["hypertension"], data["heart_disease"], data["smoking_history"], data["bmi"], data["HbA1c_level"], data["blood_glucose_level"])
    features = [data['gender'], data['age'], data['hypertension'], data['heart_disease'],
                data['smoking_history'], data['bmi'], data['HbA1c_level'], data['blood_glucose_level']]
    prediction = model.predict([features])
    return jsonify({'diabetes_prediction': int(prediction[0])})


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
