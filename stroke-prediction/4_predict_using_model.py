import joblib
import pandas as pd

model = joblib.load("model/random_forest_classifier.pkl")
ohe = joblib.load("model/one_hot_encoder.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

in_age = float(input("Enter age : "))
in_bmi = float(input("Enter BMI : "))
in_glucose = float(input("Enter average glucose level : "))
in_gender = int(input("Gender (1 : Male, 0 : Female) : "))
in_hypertension = int(input("Has hypertension? (1 : Yes, 0 : No) : "))
in_heart_disease = int(input("Has heart disease? (1 : Yes, 0 : No) : "))
in_ever_married = int(input("Ever married? (1 : Yes, 0 : No) : "))
in_residence = int(input("Residence type (1 : Urban, 0 : Rural) : "))

valid_work_type = ohe.categories_[0].tolist()
valid_smoking_status = ohe.categories_[1].tolist()

while True:
    in_work_type = input(f"Enter work type {valid_work_type} : ")
    if in_work_type not in valid_work_type:
        print(f"Invalid. Choose from : {valid_work_type}")
    else:
        break

while True:
    in_smoking = input(f"Enter smoking status {valid_smoking_status} : ")
    if in_smoking not in valid_smoking_status:
        print(f"Invalid. Choose from : {valid_smoking_status}")
    else:
        break

str_inputs = pd.DataFrame([{
    "work_type": in_work_type,
    "smoking_status": in_smoking
}])

encoded_str_inputs = ohe.transform(str_inputs)
encoded_str_df = pd.DataFrame(encoded_str_inputs, columns=ohe.get_feature_names_out(["work_type", "smoking_status"]))

numeric_inputs = pd.DataFrame([{
    "age" : in_age,
    "bmi" : in_bmi,
    "avg_glucose_level" : in_glucose,

    "gender": in_gender,
    "hypertension": in_hypertension,
    "heart_disease": in_heart_disease,
    "ever_married": in_ever_married,
    "Residence_type": in_residence
}])
numeric_inputs[["age", "bmi", "avg_glucose_level"]] = scaler.transform(numeric_inputs[["age", "bmi", "avg_glucose_level"]])

final_input = pd.concat([numeric_inputs, encoded_str_df], axis=1)
final_input = final_input[feature_columns]

prediction = model.predict(final_input)

if prediction[0] == 0:
    print("The patient is less likely to have a stroke.")
else:
    print("The patient is more likely to have a stroke.")