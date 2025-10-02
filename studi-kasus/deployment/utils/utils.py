import pickle
import tensorflow as tf


def load_ann_model(ann_model):
    """
    Load model ANN dari file Keras.
    
    Args:
        ann_model (str): Path ke file model ANN
        
    Returns:
        Model: Model ANN yang sudah di-load
    """
    # Load model Keras dari file
    model = tf.keras.models.load_model(ann_model)
    return model


def load_xgb_model(xgb_model):
    """
    Load model XGBoost dari file pickle.
    
    Args:
        xgb_model (str): Path ke file model XGBoost
        
    Returns:
        Model: Model XGBoost yang sudah di-load
    """
    # Buka dan load model dari pickle file
    with open(xgb_model, 'rb') as model_file:
        model = pickle.load(model_file)
    return model


def load_scaler(scaler_path):
    """
    Load scaler dari file pickle.
    
    Args:
        scaler_path (str): Path ke file scaler
        
    Returns:
        Scaler: Scaler object yang sudah di-fit
    """
    # Buka dan load scaler dari pickle file
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return scaler


def preprocessing(data, scaler):
    """
    Transform data menggunakan scaler.
    
    Args:
        data (DataFrame/array): Data yang akan di-scale
        scaler (Scaler): Scaler object untuk transformasi
        
    Returns:
        array: Data yang sudah di-scale
    """
    # Transform data dengan scaler yang sudah di-fit
    data_scaled = scaler.transform(data)
    return data_scaled


def ensemble_predict(ann_model, xgb_model, data):
    """
    Prediksi ensemble dengan averaging ANN dan XGBoost.
    
    Args:
        ann_model (Model): Model ANN
        xgb_model (Model): Model XGBoost
        data (array): Data yang sudah di-preprocessing
        
    Returns:
        tuple: (prediksi_class, probabilitas_ensemble)
    """
    # Prediksi dengan ANN dan flatten hasilnya
    ann_pred = ann_model.predict(data).flatten()
    
    # Prediksi dengan XGBoost
    xgb_pred = xgb_model.predict(data)
    
    # Rata-rata probabilitas dari kedua model
    ensemble_pred = (ann_pred + xgb_pred) / 2
    
    # Konversi probabilitas ke class (threshold 0.5)
    final_pred = (ensemble_pred > 0.5).astype(int)
    
    return final_pred, ensemble_pred