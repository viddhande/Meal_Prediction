import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load your dataset
df = pd.read_csv('Nutritious.csv')

# Assuming X and y are defined
X = df[['DietaryPreference', 'Lifestyle', 'HealthCondition', 'ExsitingAllergies']]
y = df['Meal']

# Apply Label Encoding to features
label_encoder = LabelEncoder()
X['DietaryPreference'] = label_encoder.fit_transform(X['DietaryPreference'])
X['Lifestyle'] = label_encoder.fit_transform(X['Lifestyle'])
X['HealthCondition'] = label_encoder.fit_transform(X['HealthCondition'])
X['ExsitingAllergies'] = label_encoder.fit_transform(X['ExsitingAllergies'])

# Examine unique labels in the target variable (y)
unique_labels = y.unique()
print("Unique Labels in y:", unique_labels)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Label Encoding to target variable only on training data
y_train = label_encoder.fit_transform(y_train)

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

# Load model
with open('random_forest_model.pkl', 'rb') as f:
    clf = pickle.load(f)

# Streamlit UI
st.title("Meal Prediction")

# User inputs
dietary_preference = st.slider("Select Dietary Preference (0 for Vegetarian, 1 for Non-Vegetarian)", min_value=0, max_value=1, step=1)
lifestyle = st.slider("Select Lifestyle (0 for Active, 1 for Sedentary)", min_value=0, max_value=1, step=1)
health_condition = st.slider("Select Health Condition (0 for Good, 1 for Poor)", min_value=0, max_value=1, step=1)
existing_allergies = st.slider("Select Existing Allergies (0 for Yes, 1 for No)", min_value=0, max_value=1, step=1)

# Predict and display result
if st.button("Predict"):
    input_data = [[dietary_preference, lifestyle, health_condition, existing_allergies]]
    prediction = clf.predict(input_data)
    predicted_meal_label = prediction[0]
    predicted_meal = label_encoder.inverse_transform([predicted_meal_label])[0]
    st.write("Predicted Meal:", predicted_meal)
