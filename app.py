from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input data from form
        features = [float(request.form[key]) for key in request.form]
        
        # Scale the input data using the loaded scaler
        scaled_features = scaler.transform([features])

        # Predict using the model
        prediction = model.predict(scaled_features)[0]
        
        # Map prediction to result
        class_names = ["No Heart Disease Detected", "Heart Disease Detected"]
        result = class_names[int(prediction)]

        return render_template('result.html', result=result)

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
