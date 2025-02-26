import numpy as np
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Modeli yükle
model = pickle.load(open('lgbm_glucose_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Kullanıcıdan alınan girişleri al
        age = float(request.form['age'])
        hba1c = float(request.form['hba1c'])
        gender = request.form['gender']

        # Kadın = 1, Erkek = 0 olarak kodlayalım
        gender_woman = 1 if gender == "Female" else 0

        # Model için giriş dizisini oluştur
        features = np.array([[age, hba1c, gender_woman]])

        # Model tahmini yap (log[Glucose])
        log_glucose_pred = model.predict(features)[0]

        # Normal Glucose değerine dönüştür (anti-log)
        glucose_pred = 10 ** log_glucose_pred

        # Kullanıcıya sonucu göster
        return render_template('index.html', prediction_text=f'Predicted Glucose Level: {glucose_pred:.2f} mg/dL')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
