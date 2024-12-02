from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Modeli yükleyin
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict(np.array(features).reshape(1, -1))
    return render_template('index.html', prediction_text=f'Tahmin: {prediction[0]}')

if __name__ == '__main__':
    app.run(debug=True)
