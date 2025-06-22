
# CreditScore Prediction

**CreditScore Prediction** is a full-stack machine learning pipeline for classifying customer creditworthiness. It walks through a complete ML lifecycle—from data cleaning, preprocessing, EDA, feature engineering, model training, and evaluation, to experiment tracking with MLflow and deployment with Streamlit in a Docker container.

> _Built for modularity, reproducibility, and real-world integration._

---

## 📌 Project Highlights

- 🔍 **Data Cleaning & Preprocessing**  
  Missing values handled, categorical features encoded, and outliers treated.

- 📊 **Exploratory Data Analysis (EDA)**  
  Visualization of key trends and feature importance using Plotly and Matplotlib.

- 🧠 **Model Training & Selection**  
  Trains and compares Logistic Regression, Random Forest, and XGBoost classifiers using a consistent pipeline.

- 📈 **Evaluation Metrics**  
  Accuracy, Precision, Recall, F1-Score, ROC-AUC, and confusion matrices logged via MLflow.

- 🧪 **MLflow Experiment Tracking**  
  All training runs are logged with parameters, metrics, artifacts, and models for comparison.

- 🚀 **Streamlit App**  
  Interactive web interface for predicting credit scores based on user input.

- 🐳 **Dockerized Deployment**  
  Easy, reproducible deployment using Docker and Docker Compose.

---

## 📁 Directory Structure

```
creditscore/
├── data/                  # Raw and cleaned data
├── src/
│   ├── artifacts/         # Plots, scalers, and split datasets
│   ├── models/            # Trained models and scaler.pkl
│   ├── mlflow_tracking/
│   │   ├── app.py         # Streamlit app
│   │   └── pipeline.py    # Model training and MLflow logging
│   └── train_model.py     # Data preprocessing and splitting
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 📊 Model Performance Summary

| Model               | Accuracy | Precision | Recall  | F1-Score | ROC-AUC |
|--------------------|----------|-----------|---------|----------|---------|
| Logistic Regression| 0.605    | 0.602     | 0.605   | 0.586    | 0.760   |
| Random Forest       | 0.765    | 0.764     | 0.765   | 0.764    | 0.897   |
| XGBoost             | 0.727    | 0.727     | 0.727   | 0.727    | 0.869   |

📌 _Random Forest was selected as the best-performing model._

---

## 🚀 Run Locally

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/yourusername/creditscore.git
cd creditscore
pip install -r requirements.txt
```

### 2. Run Training & Experiment Tracking

```bash
python src/train_model.py
python src/mlflow_tracking/pipeline.py
mlflow ui  # optional: view metrics at http://localhost:5000
```

### 3. Launch Streamlit App

```bash
streamlit run src/mlflow_tracking/app.py
```

👉 **[View Live App](http://localhost:8501)**

---

## 🐳 Dockerized Deployment

Build and run the app in a container:

```bash
docker-compose up --build
```

Access the app at: [http://localhost:8501](http://localhost:8501)

---

##  Future Improvements

- Model explainability using SHAP or LIME
- Batch upload support via CSV in Streamlit
- REST API endpoint using FastAPI

---

## Author

**Ifeoma Adigwe**  
_Powered by Python • Streamlit • MLflow • Docker_

---


