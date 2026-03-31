# Hotel Reservation Prediction
 
Predicts whether a customer will honor or cancel their hotel reservation based on booking details, guest demographics, and special requests.
 
### Dataset
**Source:** [Hotel Reservations Classification Dataset](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset) (Kaggle) — 36,275 rows, 19 columns
 
### Approach
- **Binary Mapping:** Converted the target variable `booking_status` (Canceled/Not_Canceled) to numeric format.
- **Feature Engineering:** Processed categorical features like `type_of_meal_plan`, `room_type_reserved`, and `market_segment_type`.
- **Encoding:** Applied Label Encoding or One-Hot Encoding (OHE) where appropriate for categorical variables.
- **Model Comparison:** Evaluated multiple classifiers including Logistic Regression, Decision Tree, and Random Forest.
- **Hyperparameter Tuning:** Optimized the best-performing model using `RandomizedSearchCV` to maximize predictive accuracy.
 
### Key Insights
- **Lead Time:** Bookings made long in advance are significantly more likely to be canceled.
- **Average Price:** Higher room prices correlate with a higher probability of cancellation.
- **Special Requests:** Customers who make one or more special requests are less likely to cancel their stay.
- **Market Segment:** Online bookings show a higher cancellation rate compared to corporate or offline segments.
 
### Setup
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```
 
1. Run `1_data_analysis.ipynb` to clean and preprocess the hotel dataset.
2. Run `2_data_visualization.ipynb` for Exploratory Data Analysis (EDA).
3. Run `3_model_training.ipynb` to train, evaluate, and save the final model.
4. Run `4_predict_using_model.py` (or the `.ipynb` version) to make new predictions.
