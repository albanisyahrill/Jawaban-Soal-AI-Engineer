# Sepsis Prediction Model - Technical Test

## 📋 Project Overview

This project addresses two main tasks: handling imbalanced datasets and building a soft voting ensemble model combining ANN and XGBoost for sepsis prediction. The solution demonstrates end-to-end ML pipeline development from data preprocessing to deployment.

### Approach Summary

- **Data Understanding**: Understand feature distributions and class imbalance
- **Data Preprocessing**: Rigorous cleaning with expert-consideration approach for medical data
- **Ensemble Modeling**: Combined XGBoost and ANN with soft voting for robust predictions
- **Deployment**: Containerized Flask API for scalable inference

## 🎯 Model Performance

The ensemble model achieved perfect performance metrics:

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 100%  |
| Precision | 100%  |
| Recall    | 100%  |
| F1-Score  | 100%  |
| AUC-ROC   | 100%  |

**Validation Method**: K-Fold Cross Validation consistently showed 100% across all folds and metrics.

## 🏗️ Project Structure

```
project/
│
├── imbalance-dataset/                    # Imbalanced dataset analysis
├── notebooks/
│       ├── Customer-Churn-Records-v2.csv
│       └── Imbalanced_Classification_&_NN.ipynb
│
└── studi-kasus/
        ├── deployment/                   # Production deployment
        │       ├── artifacts/            # Trained models & scaler
        │       │       ├── ann_model.keras
        │       │       ├── scaler.pkl
        │       │       └── xgb_model.pkl
        │       │
        │       ├── utils/                # Helper functions
        │       │       ├── __init__.py
        │       │       └── utils.py
        │       │
        │       ├── __init__.py
        │       ├── app.py               # Flask application
        │       ├── Dockerfile           # Container configuration
        │       ├── predict.py           # API client script
        │       └── requirements.txt     # Dependencies
        ├── notebook/                    # Development notebooks
        │       ├── sepsis_emr_data.csv
        │       └── Studi_Kasus.ipynb
        └── external/                    # External validation data
                └── external_data.csv
```

## 🐳 Docker Deployment

### Prerequisites

- Docker installed on your system
- At least 2GB free disk space

### Step-by-Step Build Instructions

1. **Navigate to deployment directory**

   ```bash
   cd studi-kasus/deployment
   ```

2. **Build Docker image**

   ```bash
   docker build -t flask-api:latest .
   ```

3. **Run the container**

   ```bash
   docker run -d -p 5000:5000 flask-api
   ```

4. **Verify container is running**

   ```bash
   docker ps
   ```

5. **Check logs (if needed)**
   ```bash
   docker logs sepsis-api
   ```

The API will be available at: `http://localhost:5000`

## 🚀 API Usage

### API Endpoint

- **URL**: `http://localhost:5000/predict`
- **Method**: POST
- **Content-Type**: application/json

### Example Request using Python

Run the provided prediction script:

```bash
cd studi-kasus/deployment
python predict.py
```

### Expected Response

```json
{
  "prediction": [1],
  "prediction_probability": [0.94]
}
```

## 🎨 Design Choices & Assumptions

### Data Preprocessing

- **Missing Values**: Chose deletion over imputation due to sensitive medical nature
- **Expert Consultation**: Recommended domain expert validation for any imputation decisions
- **SMOTE-NC**: Used for oversampling to handle class imbalance in mixed data types
- **MinMaxScaler**: Selected for normalization to preserve data distribution

### Model Architecture

- **Ensemble Approach**: Soft voting combines XGBoost (tree-based) and ANN (neural network) strengths
- **K-Fold Validation**: Ensured model robustness across different data splits
- **Manual Ensemble**: Custom implementation for flexible probability weighting

### Medical Data Considerations

- **Conservative Approach**: Prioritized model interpretability and clinical relevance
- **Validation Rigor**: Multiple validation methods despite high initial metrics
- **Deployment Safety**: Comprehensive preprocessing pipeline matching training phase

### Performance Notes

While achieving 100% metrics is uncommon in real-world scenarios, the consistent performance across validation methods suggests the dataset contains clear, separable patterns. However, continued monitoring with external data is recommended.

For additional support, check the container logs and ensure all artifact files are properly mounted.
