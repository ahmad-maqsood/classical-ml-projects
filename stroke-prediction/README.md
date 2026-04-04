# Stroke Prediction

A machine learning project that predicts stroke risk based on patient health data. The dataset is highly imbalanced (~5% stroke cases), so SMOTE is used to handle the class imbalance during training.

**Dataset :** [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) on Kaggle.

---

## Project Structure

```
stroke-prediction/
│
├── data/
│   ├── healthcare-dataset-stroke-data.csv   # Raw dataset
│   └── cleaned-stroke-dataset.csv           # Cleaned dataset (output of notebook 1)
│
├── model/
│   ├── random_forest_classifier.pkl
│   ├── one_hot_encoder.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── 1_data_analysis.ipynb
├── 2_data_visualization.ipynb
├── 3_model_training.ipynb
└── 4_predict_using_model.py
```

---

## Workflow

### 1. Data Analysis
- Dropped the `id` column (no predictive value)
- Removed the single "Other" gender row
- Mapped binary categorical columns: `gender`, `ever_married`, `Residence_type`
- Combined `children` with `Never_worked` in `work_type`
- Filled missing `bmi` values with the column mean

### 2. Data Visualization
- Correlation heatmap: age has the highest correlation with stroke, followed by hypertension, heart disease, and glucose level
- Pie chart: confirms the dataset is heavily imbalanced (~95% no stroke, ~5% stroke)
- Stroke rate by smoking status: formerly smoked group has the highest stroke rate (7.9%) compared to never smoked (4.7%) and current smokers (5.3%)
- Stroke rate by age: risk increases sharply after age 50
- Stroke rate by heart disease: patients with heart disease have a notably higher stroke rate

### 3. Model Training
- Applied OHE on `work_type` and `smoking_status`
- Scaled `age`, `bmi`, and `avg_glucose_level` using StandardScaler
- Train/test split: 80/20
- Applied SMOTE only on training data to handle class imbalance
- Compared Logistic Regression, Decision Tree, and Random Forest using cross-validation
- Tuned Random Forest using RandomizedSearchCV

### 4. Prediction Script
- Loads the saved model, scaler, OHE, and feature columns
- Takes patient input from the terminal
- Scales and encodes the input using the same objects used during training
- Outputs stroke risk prediction

---

## Results

| Metric | Class 0 (No Stroke) | Class 1 (Stroke) |
|---|---|---|
| Precision | 0.95 | 0.18 |
| Recall | 0.94 | 0.19 |
| F1-Score | 0.95 | 0.19 |
| Accuracy | | 90% |

The overall accuracy of 90% is misleading due to class imbalance. A model that always predicts "no stroke" would achieve ~94% accuracy. The more important metric here is recall for class 1 (stroke), which is 0.19. This means the model catches only 19% of actual stroke cases.

SMOTE helped the model learn stroke patterns to some extent, but the limited number of real stroke cases in the dataset and the available features are likely not sufficient to reliably identify stroke patients. This model is not suitable for clinical use.

---

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
imbalanced-learn
joblib
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib imbalanced-learn joblib
```

---

## Usage

Run the prediction script from the project root:

```bash
python 4_predict_using_model.py
```

You will be prompted to enter patient details such as age, BMI, glucose level, and health history. The script will output whether the patient is at risk of a stroke.