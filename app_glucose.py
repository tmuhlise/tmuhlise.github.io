
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
        glucose = float(request.form['feature1'])  # Glucose value
        age = int(request.form['age'])            # Age value
        gender = request.form['gender']           # Gender value ("Male" or "Female")

        # Convert Gender to a numerical value (if your model was trained this way)
        gender_numeric = 1 if gender == "Male" else 0  # Male → 1, Female → 0

        # Prepare the input data (3 features)
        features = np.array([[glucose, age, gender_numeric]])  # Shape (1, 3)

        # Make the prediction
        prediction = model.predict(features)

        # Format the prediction output
        prediction_text = f"Predicted Value: {prediction[0]:.2f}"

        return render_template('index.html', prediction_text=prediction_text)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

