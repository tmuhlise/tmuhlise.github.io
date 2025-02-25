
import numpy as np


import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the model once when the app starts
model = pickle.load(open('lgbm_glucose_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index .html')

@app.route('/predict', methods=['POST'])
def predict():
    glucose_value = float(request.form['feature1'])  # Get the input value and convert to float
    # Prepare the input for the model (adjust based on your model's input requirements)
    features = [[glucose_value]]  # Example for a single feature

    # Make a prediction
    prediction = model.predict(features)
    
    # Create a prediction message
    prediction_text = f"Predicted glucose level: {prediction[0]:.2f}"
    
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
