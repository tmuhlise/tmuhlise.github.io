
import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np
import lazypredict
import lazypredict.Supervised
from lazypredict.Supervised import LazyClassifier
from lazypredict.Supervised import LazyRegressor
from lightgbm import LGBMRegressor


# Modeli yükle
model = joblib.load('lgbm_glucose_model.pkl')
import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.models import load_model
import numpy as np
model = joblib.load('lgbm_glucose_model.pkl')
# Modeli yükle
#model = load_model('lgbm_glucose_model.pkl')

# Tahmin fonksiyonu
def predict_glucose():
    try:
        # Girişten HbA1c değerini al
        hba1c_value = entry.get()
        
        # Tahmin yapmak için numpy array'e çevir
        prediction = model.predict(np.array([[hba1c_value]]))
        
        # Sonucu göster
        glucose_value = prediction[0][0]  # Model çıktısını al
        messagebox.showinfo("Tahmin Sonucu", f"Tahmini Glucose Değeri: {glucose_value:.2f}")
    except ValueError:
        messagebox.showerror("Hata", "Lütfen geçerli bir HbA1c değeri girin.")
    except Exception as e:
        messagebox.showerror("Hata", f"Bir hata oluştu: {e}")

# Tkinter ana pencere
root = tk.Tk()
root.title("HbA1c'den Glucose Tahmini")

# Etiket ve giriş kutusu
label = tk.Label(root, text="HbA1c Değerini Girin:")
label.pack(pady=5)

entry = tk.Entry(root)
entry.pack(pady=5)

# Tahmin butonu
predict_button = tk.Button(root, text="Tahmin Et", command=predict_glucose)
predict_button.pack(pady=10)

# Ana döngüyü başlat
root.mainloop()

'''''
# Tahmin fonksiyonu
def predict_glucose():
    try:
        # Kullanıcıdan HbA1c değerini al
        hba1c_value = float(entry_hba1c.get())
        
        # Gerekli özellikleri hazırlama
        input_data = np.array([[hba1c_value]])  # Modelin beklediği şekle uygun hale getir
        
        # Tahmin yap
        glucose_prediction = model.predict(input_data)
        
        # Sonucu göster
        messagebox.showinfo("Tahmin", f'Tahmin Edilen Glukoz Değeri: {glucose_prediction[0]:.2f} mg/dL')
    except ValueError:
        messagebox.showerror("Hata", "Lütfen geçerli bir HbA1c değeri girin.")

# GUI oluşturma
root = tk.Tk()
root.title("Glukoz Tahmin Uygulaması")

# HbA1c girişi
label_hba1c = tk.Label(root, text="HbA1c Değerini Girin:")
label_hba1c.pack()

entry_hba1c = tk.Entry(root)
entry_hba1c.pack()

# Tahmin butonu
button_predict = tk.Button(root, text="Tahmin Et", command=predict_glucose)
button_predict.pack()

# Uygulamayı başlat
root.mainloop()
'''''