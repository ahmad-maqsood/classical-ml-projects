# Customer Churn Prediction
 
Predicts whether a telecom customer will churn based on their account details, services, and contract information.
 
### Dataset
**Source:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle) — 7,043 rows, 21 columns
 
### Approach
- Binary mapped all Yes/No columns
- Collapsed "No phone service" and "No internet service" to 0
- Ordinal encoded Contract (Month-to-month=0, One year=1, Two year=2)
- OHE applied on gender, InternetService, PaymentMethod
- Fixed TotalCharges: 11 missing values filled with MonthlyCharges * 24
- Compared Logistic Regression, Decision Tree, and Random Forest on default parameters
- Tuned best model (Logistic Regression) with GridSearchCV
 
**Final model (Logistic Regression) on test set:**
 
| Metric | Not Churned (0) | Churned (1) |
|--------|----------------|-------------|
| Precision | 0.85 | 0.67 |
| Recall | 0.91 | 0.55 |
| F1-Score | 0.88 | 0.61 |
| Accuracy | | 81.3% |
 
### Key Insights
- Month-to-month contract customers churn at higher rate than one-year and two-year contracts
- Fiber optic internet customers churn the most
- Newer customers churn significantly more than long-tenured ones
- Senior citizens churn at a higher rate than other customers
 
### Setup
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```
 
1. Run `1_data_analysis.ipynb` to clean and preprocess the data
2. Run `2_data_visualization.ipynb` for EDA
3. Run `3_model_training.ipynb` to train and save the model
4. Run `4_predict_using_model.py` to make predictions
