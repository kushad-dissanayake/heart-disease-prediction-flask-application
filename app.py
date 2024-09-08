from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the model and scaler
with open('best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the dataset to fit the scaler
df = pd.read_csv('simulated_heart_attack_risk.csv')
X = df.drop(columns='target')
scaler = StandardScaler()
scaler.fit(X)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Gather inputs from the form
    data = {
        'age': float(request.form['age']),
        'sex': int(request.form['sex']),
        'cp': int(request.form['cp']),
        'trestbps': float(request.form['trestbps']),
        'chol': float(request.form['chol']),
        'fbs': int(request.form['fbs']),
        'restecg': int(request.form['restecg']),
        'thalach': float(request.form['thalach']),
        'exang': int(request.form['exang']),
        'oldpeak': float(request.form['oldpeak']),
        'slope': int(request.form['slope']),
        'ca': int(request.form['ca']),
        'thal': int(request.form['thal']),
        'bmi': float(request.form['bmi']),
        'smoking_status': int(request.form['smoking_status']),
        'exercise': int(request.form['exercise']),
        'family_history': int(request.form['family_history']),
        'diabetes': int(request.form['diabetes'])
    }
    
    # Convert data to DataFrame
    df_input = pd.DataFrame([data])

    # Scale the input data
    df_input_scaled = scaler.transform(df_input)

    # Predict using the loaded model
    prediction = model.predict(df_input_scaled)
    risk_probability = model.predict_proba(df_input_scaled)[0][1]

    result = {
        'prediction': 'Heart Disease Detected' if prediction[0] == 1 else 'No Heart Disease',
        'probability': f'{risk_probability:.2f}'
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
