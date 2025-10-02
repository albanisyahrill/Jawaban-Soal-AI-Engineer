import requests
import json

# URL endpoint API
url = "http://localhost:5000/predict"

# Contoh data input untuk satu pasien
patient_data = {
        "heart_rate": 123,
        "respiratory_rate": 22,
        "temperature": 42.1,
        "wbc_count": 20,
        "lactate_level": 7.6,
        "age": 34,
        "num_comorbidities": 2	
}

# Mengirim permintaan POST
response = requests.post(url, json=patient_data)

# Mencetak hasil
if response.status_code == 200:
    result = response.json()
    print("Prediksi Sepsis Berhasil:")
    print(json.dumps(result, indent=4))
else:
    print(f"Error: {response.status_code}")
    print(response.text)