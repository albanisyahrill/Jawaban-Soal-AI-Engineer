import pandas as pd
from flask import Flask, request, jsonify
from utils.utils import load_ann_model, load_xgb_model, load_scaler, preprocessing, ensemble_predict

# Inisialisasi Flask app
app = Flask(__name__)

# Path ke model dan scaler artifacts
ann_model = '/app/artifacts/ann_model.keras'
xgb_model = '/app/artifacts/xgb_model.pkl'
scaler_path = '/app/artifacts/scaler.pkl'

# Load model dan scaler saat startup
ann_model = load_ann_model(ann_model)
xgb_model = load_xgb_model(xgb_model)
scaler = load_scaler(scaler_path)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint untuk prediksi menggunakan ensemble model.
    
    Method: POST
    Endpoint: /predict
    
    Request Body (JSON):
        - Bisa berupa single object atau array of objects
        - Setiap object harus memiliki fitur-fitur yang dibutuhkan model
    
    Returns:
        JSON: Hasil prediksi dan probabilitas atau error message
    """
    try:
        # Ambil data JSON dari request
        json_data = request.get_json()
        
        # Konversi single object ke list untuk konsistensi
        if not isinstance(json_data, list):
            json_data = [json_data]
        
        # Convert ke DataFrame
        df = pd.DataFrame(json_data)
        
        # Preprocessing data dengan scaler
        preprocess_data = preprocessing(df, scaler)
        
        # Prediksi menggunakan ensemble model
        prediction, prediction_proba = ensemble_predict(ann_model, xgb_model, preprocess_data)
        
        # Return hasil prediksi dalam format JSON
        return jsonify({
            'prediction': prediction.tolist(),
            'prediction_probability': prediction_proba.tolist()
        })
        
    except Exception as e:
        # Handle error dan return pesan error
        return jsonify({'error': str(e)})


# Run Flask development server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)