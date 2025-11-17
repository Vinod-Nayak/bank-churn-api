**Bank Customer Churn Prediction (End-to-End ML Project)**

**Project Overview:**
Customer churn is one of the most critical challenges faced by subscription-based and service-oriented businesses, including banks. Retaining existing customers is significantly more profitable than acquiring new ones.
This project builds a production-ready machine learning pipeline to predict whether a bank customer will churn (exit) based on demographic, financial, and account-activity features.

The project includes:

-> Data preprocessing & EDA
-> Feature engineering
-> Multiple ML models with hyperparameter tuning
-> Exporting final model via script
-> FastAPI-based prediction service
-> Docker containerization
-> Cloud deployment

**Business Objective:**
To build an automated churn prediction system capable of identifying customers who are likely to leave the bank. This enables the business team to proactively apply retention campaigns, personalize offers, and prevent revenue loss.

**Dataset:**
The dataset contains 10,000 customer records with the following selected features:

Feature	Description
credit_score	    Customer credit score
gender	          Male/Female
age	              Age of the customer
tenure	          Years with the bank
balance	          Account balance
products_number	  Number of bank products used
credit_card	      Whether the customer has a credit card
active_member	    Account activity flag
estimated_salary	Estimated yearly salary
country	          Country of residence
churn	            Target variable (0 = stay, 1 = leave)

Both raw and cleaned datasets are available in the project folder.

**EDA Summary:**
Key Insights extracted:
-> Churn rate is ~20%
-> Customers with higher age, low activity, and high product count show higher churn probability
-> Country-wise churn distribution shows Germany has the highest churn rate
-> No missing values detected
-> Standardization is required for numerical features
-> One-hot encoding applied for categorical variables

**Visualizations include:**
-> Churn distribution bar chart
-> Correlation heatmap
-> Geography-wise churn bar plot
-> Feature importance plot (XGBoost & Random Forest)

**Model Training & Selection:**
Multiple models were trained and evaluated using Accuracy, F1-Score, and ROC-AUC:
Model	Accuracy	F1 Score	ROC-AUC
Logistic Regression	0.8085	0.2867	0.7745
Random Forest	0.8680	0.5938	0.8531
XGBoost (Final Model)	0.8675	0.5978	0.8618

Final chosen model: XGBoost Classifier
Saved using pickle for production

**Project Structure:**
Bank_Customer_Churn/
│
├── notebook.ipynb
├── data/
│   └── Bank Customer Churn Prediction.csv
|   └── cleaned_bank_churn.csv
│
├── models/
│   └── churn_model.pkl
│   └── scaler.pkl
│
├── src/
|   ├──models/
│   |  └── churn_model.pkl
│   |  └── scaler.pkl
│   ├── train.py
│   └── predict.py
│
├── requirements.txt
├── Dockerfile
└── README.md

**Steps to run the project:**

Run Locally
1. Clone the repository
git clone https://github.com/Vinod-Nayak/bank-churn-api.git
cd bank-churn-api

2. Install dependencies
pip install -r requirements.txt

3. Train the model
python src/train.py

4. Run the API
uvicorn src.predict:app --reload

Visit API docs:
👉 http://127.0.0.1:8000/docs

**🐳 Docker Deployment:**
1. Build Image
docker build -t bank-churn-api .

2. Run container
docker run -p 8000:8000 bank-churn-api

API Available at:
👉 http://localhost:8000/docs

**☁️ Cloud Deployment (Render):**
Application is live and publicly accessible:

🌍 Live App URL:
👉 https://bank-churn-api.onrender.com

🌍 Swagger UI:
👉 https://bank-churn-api.onrender.com/docs

**Example Prediction Request:**

{
  "credit_score": 650,
  "gender": 1,
  "age": 40,
  "tenure": 5,
  "balance": 120000,
  "products_number": 2,
  "credit_card": 1,
  "active_member": 1,
  "estimated_salary": 80000,
  "country_germany": 1,
  "country_spain": 0
}

**Future Improvements:**
-> Improvement	Benefit
-> Feature scaling & SMOTE	Improve minority class performance
-> Model explainability (SHAP)	Business interpretability
-> Integrate DB logging, Track usage & retention results
-> CI/CD pipeline	Fully automated MLOps

👨‍💻 Author

Vinod Nayak Devavath
AI / ML Engineer
d.vinodnayak157@gmail.com
🔗 LinkedIn: https://github.com/Vinod-Nayak
