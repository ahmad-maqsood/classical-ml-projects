import joblib
import pandas as pd

model = joblib.load("model/random_forest_classifier.pkl")
ohe = joblib.load("model/one_hot_encoder.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

in_adults = int(input("Enter the no. of adults : "))
in_children = int(input("Enter the no. of children : "))
in_weekend_nights = int(input("Enter the no. of weekend nights : "))
in_week_nights = int(input("Enter the no. of week nights : "))
in_req_parking = int(input("Does you customer requires parking? (1 : Yes, 0 : No) : "))
in_lead_time = int(input("Number of days between the date of booking and the arrival date : "))
in_repeated_guest = int(input("Is the customer a repeated guest? (0 : No, 1 : Yes) : "))
in_previous_cancellations = int(input("Number of previous bookings that were canceled by the customer prior to the current booking : "))
in_previous_not_cancellations = int(input('Number of previous bookings "not" canceled by the customer prior to the current booking : '))
in_avg_price = float(input("Average price per day of the reservation : "))
in_special_requests = int(input("Total number of special requests made by the customer : "))

valid_meal_plan = ohe.categories_[0].tolist()
valid_room_type = ohe.categories_[1].tolist()
valid_market_segment = ohe.categories_[2].tolist()

while(True):
    in_meal_plan = input("Enter the meal plan(Meal Plan 1, __ 2, __ 3, Not Selected) : ")
    if in_meal_plan not in valid_meal_plan:
        print(f"Invalid Meal Plan. Choose from : {valid_meal_plan}")
    else:
        break

while(True):
    in_room_type = input("Enter the room type(Room_Type 1, Room_Type 2, ______, Room_Type 7) : ")
    if in_room_type not in valid_room_type:
        print(f"Invalid Room Type. Choose from : {valid_room_type}")
    else:
        break

while(True):
    in_market_segment = input("Enter the Market Segment Type('Aviation', 'Complementary', 'Corporate', 'Offline', 'Online') : ")
    if in_market_segment not in valid_market_segment:
        print(f"Invalid Market Segment Type. Choose from : {valid_market_segment}")
    else:
        break

numeric_inputs = pd.DataFrame([{
    "no_of_adults" : in_adults,
    "no_of_children" : in_children,
    "no_of_weekend_nights" : in_weekend_nights,
    "no_of_week_nights" : in_week_nights,
    "required_car_parking_space" : in_req_parking,
    "lead_time" : in_lead_time,
    "repeated_guest" : in_repeated_guest,
    "no_of_previous_cancellations" : in_previous_cancellations,
    "no_of_previous_bookings_not_canceled" : in_previous_not_cancellations,
    "avg_price_per_room" : in_avg_price,
    "no_of_special_requests"  : in_special_requests
}])

str_inputs = pd.DataFrame([{ 
    "type_of_meal_plan" : in_meal_plan,
    "room_type_reserved" : in_room_type,
    "market_segment_type" : in_market_segment
}])

encoded_str_inputs = ohe.transform(str_inputs)
encoded_str_inputs_df = pd.DataFrame(encoded_str_inputs, 
                                     columns=ohe.get_feature_names_out(["type_of_meal_plan", "room_type_reserved", "market_segment_type"]))

final_input = pd.concat([numeric_inputs, encoded_str_inputs_df], axis=1)
final_input = final_input[feature_columns]

prediction = model.predict(final_input)

if(prediction[0] == 0):
    print("The customer is less likely to cancel the reservation.")
else:
    print("The customer is likely to cancel the reservation.")