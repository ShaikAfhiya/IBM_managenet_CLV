📊 Customer Lifetime Value Prediction (CLV) - FastAPI ML Project
🚀 Project Overview

This project predicts Customer Lifetime Value (CLV) using a Machine Learning model deployed with FastAPI.
The system takes customer details as input and returns the predicted lifetime value.

🎯 Objective

To build a machine learning model that helps businesses:

Identify high-value customers
Improve marketing strategies
Increase customer retention
Optimize business decisions
🧠 Machine Learning Workflow
Data Preprocessing
Feature Engineering
Model Training
Model Evaluation
Model Deployment using FastAPI

📂 Dataset Information

The dataset contains customer-related features such as:

Coverage
Education
Employment Status
Gender
Income
Location Code
Marital Status
Policy Type
Monthly Premium Auto
Months Since Last Claim
Number of Policies
Total Claim Amount
Vehicle Class
Vehicle Size

👉 Target Variable:

Customer Lifetime Value (CLV)

🛠️ Tech Stack

Python 🐍
Pandas
NumPy
Scikit-learn
FastAPI ⚡
Pydantic
Joblib

⚙️ Model Deployment (FastAPI)

📌 API Endpoint

POST /

📥 Input Format (JSON)

  {
  "Response": "No",
  "Coverage": "Basic",
  "Education": "Bachelor",
  "EmploymentStatus": "Employed",
  "Gender": "F",
  "Income": 56274,
  "Location Code": "Suburban",
  "Marital Status": "Married",
  "Monthly Premium Auto": 69,
  "Months Since Last Claim": 32,
  "Months Since Policy Inception": 5,
  "Number of Open Complaints": 0,
  "Number of Policies": 1,
  "Policy Type": "Corporate Auto",
  "Policy": "Corporate L3",
  "Renew Offer Type": "Offer1",
  "Sales Channel": "Agent",
  "Total Claim Amount": 384.811147,
  "Vehicle Class": "Two-Door Car",
  "Vehicle Size": "Medsize"
}


📤 Output


{
  "Customer Lifetime Value Prediction": 7.92
}


▶️ How to Run Locally


1️⃣ Clone the repository

git clone https://github.com/your-username/clv-project.git
cd clv-project

2️⃣ Install dependencies

pip install fastapi uvicorn pandas scikit-learn joblib

3️⃣ Run FastAPI server

uvicorn app:app --reload

4️⃣ Open API docs

http://127.0.0.1:8000/docs

📊 Model Features Handling
Uses Pydantic aliases to handle column names with spaces
Converts request data into DataFrame for prediction
Ensures proper type conversion for ML model

🚀 Key Highlights

✔ End-to-end ML pipeline
✔ FastAPI deployment
✔ Handles real-world dataset formatting issues
✔ Ready for production-level enhancement
