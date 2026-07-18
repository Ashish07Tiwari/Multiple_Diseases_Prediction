### Multiple Diseases Prediction

# 🏥 Multiple Disease Prediction System using Machine Learning

A comprehensive Machine Learning web application that predicts the likelihood of multiple diseases using patient medical data. The project integrates trained ML models with an interactive Streamlit interface, enabling users to obtain predictions in real time.

---

## 📌 Project Overview

The Multiple Disease Prediction System is an end-to-end Machine Learning application developed to assist in the early detection of common diseases. Users can enter relevant medical parameters through a user-friendly interface, and the system predicts whether the patient is at risk for a specific disease.

The application currently supports prediction for:

- 🩺 Diabetes
- ❤️ Heart Disease
- 🧠 Kidney Disease
- 🎗️ Breast Cancer

The project demonstrates the complete Machine Learning workflow including:

- Data Collection
- Data Preprocessing
- Feature Engineering
- Model Training
- Model Evaluation
- Model Serialization
- Streamlit Web Deployment

---

# 🚀 Features

- Interactive Streamlit Dashboard
- Predict Multiple Diseases
- User-Friendly Interface
- Real-Time Predictions
- Pre-trained Machine Learning Models
- Fast and Lightweight Application
- Modular Project Structure
- Easy to Extend with New Disease Models

---

# 🛠️ Tech Stack

## Programming Language

- Python

## Machine Learning

- Scikit-Learn
- Pandas
- NumPy

## Data Visualization

- Matplotlib
- Seaborn

## Model Serialization

- Joblib
- Pickle

## Web Framework

- Streamlit

---

# 📂 Project Structure

```
MLProject/
│
├── artifacts/
│   ├── trained_model.pkl
│   ├── preprocessor.pkl
│   ├── scaler.pkl
│
├── data/
│   ├── diabetes.csv
│   ├── heart.csv
│   ├── kidney.csv
│   ├── breast_cancer.csv
│
├── notebook/
│   ├── EDA.ipynb
│   ├── Model_Training.ipynb
│
├── src/
│   ├── components/
│   ├── pipeline/
│   ├── utils.py
│   ├── logger.py
│   ├── exception.py
│
├── app.py
├── requirements.txt
├── README.md
└── setup.py
```

---

# ⚙️ Installation

## Clone the Repository

```bash
git clone https://github.com/your-username/MLProject.git
```

```bash
cd MLProject
```

---

## Create Virtual Environment

Windows

```bash
python -m venv venv
```

Activate

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv
```

Activate

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

After running, open:

```
http://localhost:8501
```

---

# 📊 Machine Learning Workflow

The project follows the standard Machine Learning pipeline:

```
Data Collection
        │
        ▼
Data Cleaning
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Engineering
        │
        ▼
Data Preprocessing
        │
        ▼
Train-Test Split
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Model Serialization
        │
        ▼
Streamlit Deployment
```

---

# 🤖 Machine Learning Algorithms

Different supervised learning algorithms can be used depending on the disease dataset, such as:

- Logistic Regression
- Random Forest
- Decision Tree
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gradient Boosting
- XGBoost (optional)

The best-performing model is selected based on evaluation metrics.

---

# 📈 Model Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

---

# 📋 Input Features

Each disease prediction module accepts different medical parameters.

Examples include:

### Diabetes

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

### Heart Disease

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- ECG Results
- Maximum Heart Rate
- Exercise-Induced Angina

### Kidney Disease

- Blood Pressure
- Specific Gravity
- Albumin
- Sugar
- Blood Glucose
- Blood Urea
- Serum Creatinine
- Sodium
- Potassium
- Hemoglobin

### Breast Cancer

Various cell nucleus characteristics including:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Symmetry

---

# 📸 Application Screenshots

You can add screenshots here.

Example:

```
screenshots/
│
├── home.png
├── diabetes_prediction.png
├── heart_prediction.png
├── kidney_prediction.png
└── breast_prediction.png
```

---

# 📦 Future Improvements

- User Authentication
- Database Integration
- Cloud Deployment
- Docker Support
- CI/CD Pipeline
- Explainable AI (SHAP & LIME)
- Model Monitoring
- Prediction History
- PDF Report Generation
- REST API using FastAPI

---

# 🎯 Learning Outcomes

This project demonstrates:

- End-to-End Machine Learning
- Data Preprocessing
- Feature Engineering
- Model Building
- Hyperparameter Tuning
- Model Deployment
- Streamlit Development
- Software Engineering Best Practices
- Modular Project Architecture

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository

2. Create a feature branch

```bash
git checkout -b feature-name
```

3. Commit changes

```bash
git commit -m "Added new feature"
```

4. Push changes

```bash
git push origin feature-name
```

5. Create a Pull Request

---

# 📝 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Ashish Tiwari**

B.Tech Computer Science Engineering

GitHub:
https://github.com/Ashish07Tiwari

LinkedIn:
(Add your LinkedIn profile here)

---

# ⭐ If you found this project useful

Give this repository a ⭐ on GitHub.

It motivates me to build more Machine Learning projects.

---

## Thank You ❤️
