import numpy as np
import joblib
import pandas as pd

model = joblib.load("model/logistic_regression_model.pkl")
ohe = joblib.load("model/one_hot_encoder.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

in_senior = int(input("Are you a Senior Citizen? (1 = Yes, 0 = No) : "))
in_partner = int(input("Do you have a Partner? (1 = Yes, 0 = No) : "))
in_dependents = int(input("Do you have Dependents? (1 = Yes, 0 = No) : "))
in_tenure = int(input("Tenure (Months with company) : "))
in_phone = int(input("Phone Service? (1 = Yes, 0 = No) : "))
in_multiple_lines = int(input("Multiple Lines? (1 = Yes, 0 = No) : "))
in_online_security = int(input("Online Security? (1 = Yes, 0 = No) : "))
in_online_backup = int(input("Online Backup? (1 = Yes, 0 = No) : "))
in_device_protection = int(input("Device Protection? (1 = Yes, 0 = No) : "))
in_tech_support = int(input("Tech Support? (1 = Yes, 0 = No) : "))
in_streaming_tv = int(input("Streaming TV? (1 = Yes, 0 = No) : "))
in_streaming_movies = int(input("Streaming Movies? (1 = Yes, 0 = No) : "))
in_paperless = int(input("Paperless Billing? (1 = Yes, 0 = No) : "))
in_monthly = float(input("Monthly Charges : "))
total_charges = in_monthly * in_tenure
print(f"Total Charges : {total_charges}")
in_contract = int(input("Contract Type (0 = Month-to-month, 1 = One Year, 2 = Two Year) : "))

valid_genders = ohe.categories_[0].tolist()
valid_internet = ohe.categories_[1].tolist()
valid_payment = ohe.categories_[2].tolist()

while True:
    in_gender = input(f"Gender {valid_genders} : ")
    in_internet = input(f"Internet Service {valid_internet} : ")
    in_payment = input(f"Payment Method {valid_payment} : ")

    if in_gender not in valid_genders:
        print(f"Invalid gender. Choose from: {valid_genders}")
    elif in_internet not in valid_internet:
        print(f"Invalid internet service. Choose from: {valid_internet}")
    elif in_payment not in valid_payment:
        print(f"Invalid payment method. Choose from: {valid_payment}")
    else:
        break

numerical_input = pd.DataFrame([{
    "SeniorCitizen": in_senior,
    "Partner": in_partner,
    "Dependents": in_dependents,
    "tenure": in_tenure,
    "PhoneService": in_phone,
    "MultipleLines": in_multiple_lines,
    "OnlineSecurity": in_online_security,
    "OnlineBackup": in_online_backup,
    "DeviceProtection": in_device_protection,
    "TechSupport": in_tech_support,
    "StreamingTV": in_streaming_tv,
    "StreamingMovies": in_streaming_movies,
    "Contract": in_contract,
    "PaperlessBilling": in_paperless,
    "MonthlyCharges": in_monthly,
    "TotalCharges": total_charges,
}])

cat_input = pd.DataFrame([{
    "gender": in_gender,
    "InternetService": in_internet,
    "PaymentMethod": in_payment
}])

encoded_cat = ohe.transform(cat_input)
encoded_df = pd.DataFrame(encoded_cat, columns=ohe.get_feature_names_out(["gender", "InternetService", "PaymentMethod"]))

final_input = pd.concat([numerical_input, encoded_df], axis=1)
final_input = final_input[feature_columns]  # there was some issue with the feature columns, so I had to add this

prediction = model.predict(final_input)

if prediction[0] == 1:
    print("\nPrediction : This customer is likely to Churn.")
else:
    print("\nPrediction : This customer is NOT likely to Churn.")