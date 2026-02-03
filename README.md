# 🎓 Student Performance Prediction API

An end-to-end **Machine Learning + MLOps** project that predicts a student’s final performance percentage using a **production-ready ML pipeline**, **MLflow model registry**, **FastAPI inference service**, and **Dockerized deployment**.

This project demonstrates how to move from model training to a **scalable, reproducible, and deployable ML system**.

---

## 📌 Problem Statement

Educational institutions often want to **predict student performance early** to provide timely support and interventions.

The goal of this project is to:

* Predict a student’s **Final_Percentage**
* Use only **pre-exam, inference-safe features**
* Avoid **data leakage**
* Deploy the model via a **real API**

---

## 🧠 Solution Overview

The solution follows a **real-world ML system design**:

1. Train a model using a **scikit-learn Pipeline**
2. Track experiments and metrics using **MLflow**
3. Register and version the best model in **MLflow Model Registry**
4. Serve predictions using **FastAPI**
5. Package everything using **Docker**

---

## 🏗️ System Architecture

```
Client (Swagger / curl / UI)
        |
        v
FastAPI (student-performance-api)
        |
        v
MLflow Model Registry (Production Model)
        |
        v
sklearn Pipeline
(Preprocessing + RandomForest)
        |
        v
Predicted Final Percentage
```

---

## 📊 Dataset Overview

Source: Student Performance Dataset (CSV)

### Original Columns

```
Student_ID
Age
Gender
Class
Study_Hours_Per_Day
Attendance_Percentage
Parental_Education
Internet_Access
Extracurricular_Activities
Math_Score
Science_Score
English_Score
Previous_Year_Score
Final_Percentage
Performance_Level
Pass_Fail
```

---

## 🚨 Feature Selection & Data Leakage Handling

To avoid **training–serving skew**, the following columns were **explicitly removed** during training:

❌ Dropped Columns:

* `Student_ID` (identifier)
* `Math_Score`, `Science_Score`, `English_Score` (post-exam leakage)
* `Performance_Level`, `Pass_Fail` (derived targets)
* `Final_Percentage` (target)

---

## ✅ Final Inference Features (API Inputs)

The API accepts **exactly 9 parameters**:

| Feature                    | Type   |
| -------------------------- | ------ |
| Age                        | int    |
| Gender                     | string |
| Class                      | string |
| Study_Hours_Per_Day        | float  |
| Attendance_Percentage      | float  |
| Parental_Education         | string |
| Internet_Access            | string |
| Extracurricular_Activities | string |
| Previous_Year_Score        | float  |

---

## 🤖 Model Details

* **Algorithm:** RandomForestRegressor
* **Preprocessing:** OneHotEncoding for categorical features
* **Pipeline:** `ColumnTransformer + RandomForest`
* **Metrics Logged:**

  * MSE
  * RMSE
  * MAE
  * R²

All preprocessing is bundled **inside the pipeline**, ensuring:

* Consistent training and inference
* No feature mismatch
* Production safety

---

## 🔬 Experiment Tracking (MLflow)

MLflow is used to:

* Track experiments
* Log parameters and metrics
* Compare model runs
* Register models with versioning

### Registered Model

```
Name: StudentPerformanceModel
Stage: Production
Version: v3
```

---

## 🚀 API Design (FastAPI)

### Endpoint

```
POST /predict
```

### Example Request

```json
{
  "Age": 16,
  "Gender": "Female",
  "Class": "10",
  "Study_Hours_Per_Day": 3.5,
  "Attendance_Percentage": 88,
  "Parental_Education": "Graduate",
  "Internet_Access": "Yes",
  "Extracurricular_Activities": "Yes",
  "Previous_Year_Score": 72
}
```

### Example Response

```json
{
  "predicted_final_percentage": 75.9
}
```

### API Docs

Swagger UI is available at:

```
http://localhost:8000/docs
```

---

## 🐳 Dockerized Deployment

The API is fully containerized using Docker.

### Build Image

```bash
docker build -t student-performance-api .
```

### Run Container (with MLflow DB mounted)

```bash
docker run -p 8000:8000 \
-v C:/Users/hp/Desktop/mlflow-student-performance/src:/mlflow \
-e MLFLOW_TRACKING_URI=sqlite:///mlflow/mlflow.db \
student-performance-api
```

---

## 📁 Project Structure

```
student-performance-api/
│
├── app/
│   └── main.py          # FastAPI application
│
├── Dockerfile
├── requirements.txt
├── README.md
```

---

## 🛠️ Tech Stack

* **Python 3.10**
* **scikit-learn**
* **MLflow**
* **FastAPI**
* **Pydantic**
* **Docker**
* **Uvicorn**

---

## 🧪 Key ML Engineering Concepts Demonstrated

* Feature selection & data leakage prevention
* Training–serving consistency
* sklearn Pipelines
* Experiment tracking
* Model versioning & promotion
* Production inference APIs
* Docker-based deployment

---

## 📌 Future Improvements

* Input validation (ranges, enums)
* Model monitoring & drift detection
* CI/CD pipeline
* Cloud deployment (AWS / Azure)
* Authentication & rate limiting

---

## 👤 Author

**Ankit Kumar Singh**
Data Engineering & Machine Learning Enthusiast
