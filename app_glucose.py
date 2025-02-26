import numpy as np
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('lgbm_glucose_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs from the form
        age = int(request.form['age'])            # Age input
        hba1c = float(request.form['hba1c'])      # HbA1c input
        gender = request.form['gender']           # Gender input ("Male" or "Female")

        # Convert Gender to a numerical value (1 for Woman, 0 for Man)
        gender_woman = 1 if gender == "Female" else 0  

        # Prepare input array (matching model feature order: Age, HbA1c, Gender_woman)
        features = np.array([[age, hba1c, gender_woman]])  # Shape (1, 3)

        # Make prediction
        prediction = model.predict(features)

        # Format output
        prediction_text = f"Predicted Value: {prediction[0]:.2f}"

        return render_template('index.html', prediction_text=prediction_text)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
