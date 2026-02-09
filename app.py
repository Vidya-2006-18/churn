import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

with open('lm.pkl', 'rb') as file:
    lr_model_data = pickle.load(file)
    lr_model = lr_model_data['model']
    lr_features_name = lr_model_data['features_name']

with open('dt.pkl', 'rb') as file:
    dt_model_data = pickle.load(file)
    dt_model = dt_model_data['model']
    dt_features_name = dt_model_data['features_name']

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('encoders.pkl', 'rb') as file:
    encoders = pickle.load(file)

st.title('Telco Customer Churn Prediction')
st.write('Enter customer details to predict churn status.')

selected_model_name = st.selectbox('Select Model', ['Logistic Regression', 'Decision Tree'])

if selected_model_name == 'Logistic Regression':
    model = lr_model
    features_name = lr_features_name
elif selected_model_name == 'Decision Tree':
    model = dt_model
    features_name = dt_features_name
gender = st.selectbox('Gender', ['Female', 'Male'])
senior_citizen = st.selectbox('Senior Citizen', [0, 1])
partner = st.selectbox('Partner', ['Yes', 'No'])
dependents = st.selectbox('Dependents', ['Yes', 'No'])
tenure = st.slider('Tenure (months)', 0, 72, 1)
phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
multiple_lines = st.selectbox('Multiple Lines', ['No phone service', 'No', 'Yes'])
internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
online_backup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
monthly_charges = st.number_input('Monthly Charges', min_value=0.0, value=20.0)
total_charges = st.number_input('Total Charges', min_value=0.0, value=20.0)

def predict_churn(model, scaler, encoders, features_name, user_input):
    input_df = pd.DataFrame([user_input])

    for column, encoder in encoders.items():
        if column in input_df.columns:
            if input_df[column].iloc[0] in encoder.classes_:
                input_df[column] = encoder.transform(input_df[column])
            else:
                st.warning(f"Category '{input_df[column].iloc[0]}' not seen in training data for feature '{column}'. Assigning a default value (e.g., 0).")
                input_df[column] = 0

    numerical_features_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']

    input_df[numerical_features_to_scale] = scaler.transform(input_df[numerical_features_to_scale])

    input_df['AvgMonthlyCharges'] = input_df.apply(lambda row: row['TotalCharges'] / row['tenure'] if row['tenure'] != 0 else 0, axis=1)

    processed_input = pd.DataFrame(columns=features_name)
    for col in features_name:
        if col in input_df.columns:
            processed_input[col] = input_df[col]
        else:
            processed_input[col] = 0

    processed_input = processed_input[features_name]

    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)[:, 1]

    return prediction[0], prediction_proba[0]

if st.button('Predict Churn'):
    user_input = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
    }

    prediction, probability = predict_churn(model, scaler, encoders, features_name, user_input)

    if prediction == 1:
        st.error(f"The customer is likely to churn with a probability of {probability:.2f}")
    else:
        st.success(f"The customer is not likely to churn with a probability of {1-probability:.2f}")
