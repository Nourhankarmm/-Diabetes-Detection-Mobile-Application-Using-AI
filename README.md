# Diabetes Detection Application Using AI

## Project Description

This project implements a **Diabetes Detection Application** that uses **Artificial Intelligence (AI)** and machine learning models to predict whether a person is likely to have diabetes based on medical data. The application uses a dataset with various health-related features to train and evaluate a classification model that can predict the presence or absence of diabetes.

The main goal of the project is to provide an automated and reliable system that can assist in the early detection of diabetes by analyzing medical data, enabling timely intervention and improved health outcomes.

## Dataset

The dataset used for training and evaluation is the **Diabetes Detection Dataset**, which contains medical data for patients, and the target variable indicates whether the person has diabetes.

- **Dataset URL**: [Diabetes Detection Dataset](https://www.kaggle.com/datasets/)

The dataset contains the following columns:
- `gender`: Gender of the patient
- `age`: Age of the patient
- `hypertension`: Whether the patient has hypertension (1: Yes, 0: No)
- `heart_disease`: Whether the patient has heart disease (1: Yes, 0: No)
- `smoking_history`: History of smoking (e.g., never smoked, former smoker, etc.)
- `bmi`: Body Mass Index (weight in kg / height in m²)
- `HbA1c_level`: Level of HbA1c (a marker of blood glucose levels over time)
- `blood_glucose_level`: Blood glucose level after a fasting test
- `diabetes`: Target variable (1 if the patient has diabetes, 0 otherwise)

## Installation

To set up the project locally, follow these steps:

### 1. Clone the repository
git clone https://github.com/yourusername/diabetes-detection.git

## 2. Navigate to the project directory
cd diabetes-detection

## 3. Install dependencies
Make sure you have Python 3.x installed. Then, install the required libraries by running:

pip install -r requirements.txt
## 4. Download the dataset
Download the dataset from the link provided above and place the file in the data/ folder of the project.

## Usage
1. Data Preprocessing
The data preprocessing involves cleaning the dataset, handling missing values, and scaling the features to ensure that the model performs efficiently.

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('data/diabetes_prediction_dataset.csv')

# Preprocessing: Handle missing values, scale features, etc.
scaler = StandardScaler()
data[['bmi', 'HbA1c_level', 'blood_glucose_level', 'age']] = scaler.fit_transform(
    data[['bmi', 'HbA1c_level', 'blood_glucose_level', 'age']])
2. Train the Model
We train the model using a Random Forest Classifier, but you can experiment with other classifiers like Logistic Regression, SVM, etc.

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Split data into features (X) and target (y)
X = data.drop(columns=['diabetes'])
y = data['diabetes']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
3. Predicting New Data
Once the model is trained, you can use it to predict whether new data (a new patient) has diabetes or not.

def predict_diabetes(input_data):
    # Preprocess the input data (scaling, etc.)
    input_data = scaler.transform([input_data])
    
    # Make prediction
    prediction = classifier.predict(input_data)
    return "Diabetic" if prediction == 1 else "Non-diabetic"

# Example usage:
new_patient = [1, 45, 1, 0, 'never smoked', 29.0, 5.8, 120]  # Example input data for a new patient
result = predict_diabetes(new_patient)
print(f"The patient is: {result}")
4. Running the Script
To run the entire workflow and train the model, use the following command:

python train_and_predict.py
## Project Structure
diabetes-detection/
│
├── src/               # Source code for diabetes detection
│   ├── train_and_predict.py   # Main script for training and prediction
│   └── preprocess_data.py     # Functions for preprocessing the data
├── data/              # Folder containing the dataset
│   └── diabetes_prediction_dataset.csv   # The diabetes prediction dataset
├── requirements.txt   # Python dependencies
├── README.md          # Project documentation
└── LICENSE            # Project license
## Dependencies
The project requires the following Python libraries:

pandas: For data manipulation
scikit-learn: For machine learning algorithms and tools
matplotlib: For data visualization
seaborn: For statistical data visualization
You can install all required dependencies using the requirements.txt:

pip install -r requirements.txt
## Acknowledgments
Kaggle for the diabetes prediction dataset.
Scikit-learn for machine learning tools.
Matplotlib and Seaborn for data visualization.
vbnet
