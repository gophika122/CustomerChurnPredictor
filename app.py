# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import sqlite3
from datetime import datetime

app = Flask(__name__)

# --- Configuration ---
DATABASE_NAME = 'customer_churn.db'
MODEL_PATH = 'churn_model.joblib'
FEATURES_PATH = 'model_features.joblib'

# --- Load Model and Features ---
try:
    model = joblib.load(MODEL_PATH)
    model_features = joblib.load(FEATURES_PATH)
    print("Model and features loaded successfully for web app.")
except FileNotFoundError:
    print(f"Error: Model or feature file not found. Make sure '{MODEL_PATH}' and '{FEATURES_PATH}' exist.")
    exit() # Exit if model not found, as web app cannot function

# --- Database & Preprocessing Functions (copied from churn_prediction_app.py for self-containment) ---
def fetch_customer_data_by_id(customer_id):
    """Fetches a single customer's data from the database."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        customer_df = pd.read_sql_query(f"SELECT * FROM Customers WHERE customer_id = '{customer_id}'", conn)
        usage_df = pd.read_sql_query(f"SELECT * FROM UsageHistory WHERE customer_id = '{customer_id}'", conn)
        print(f"Data fetched for customer_id: {customer_id}")
        return customer_df, usage_df
    except sqlite3.Error as e:
        print(f"Error fetching data from database for customer {customer_id}: {e}")
        return pd.DataFrame(), pd.DataFrame()
    finally:
        if conn:
            conn.close()

def preprocess_new_data(customer_df, usage_df, model_features, prediction_date=None):
    """
    Preprocesses new customer data for prediction.
    This function should largely mirror the preprocessing in data_preprocessing.py
    to ensure consistency of features.
    """
    if customer_df.empty:
        print("No customer data to preprocess.")
        return None

    # Ensure consistent column types as in training
    customer_df['signup_date'] = pd.to_datetime(customer_df['signup_date'])
    current_date = prediction_date if prediction_date else datetime.now()
    customer_df['tenure_months'] = (current_date - customer_df['signup_date']).dt.days // 30
    customer_df['tenure_months'] = customer_df['tenure_months'].apply(lambda x: max(x, 0))

    # --- CRITICAL FIX FOR ALL POTENTIAL NUMERIC BYTE STRINGS (Mirroring training) ---
    for col in ['monthly_charges', 'total_charges', 'age']:
        customer_df[col] = customer_df[col].apply(
            lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
        )
        problematic_strings = [
            '-\x00\x00\x00\x00\x00\x00\x00',
            'F\x00\x00\x00\x00\x00\x00\x00',
            '\x05\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00'
        ]
        customer_df[col] = customer_df[col].replace(problematic_strings, pd.NA)

        customer_df[col] = pd.to_numeric(customer_df[col], errors='coerce').fillna(0).astype(float)
    # --- END OF CRITICAL FIX ---

    # Special handling for total_charges where it's 0 but monthly_charges is not
    customer_df.loc[(customer_df['total_charges'] == 0) & (customer_df['monthly_charges'] > 0), 'total_charges'] = customer_df['monthly_charges']

    # Aggregate usage data per customer: calculate average usage and total support tickets
    if not usage_df.empty:
        usage_df['usage_date'] = pd.to_datetime(usage_df['usage_date'])
        agg_usage_df = usage_df.groupby('customer_id').agg(
            avg_data_usage_gb=('data_usage_gb', 'mean'),
            avg_call_minutes=('call_minutes', 'mean'),
            total_support_tickets=('num_support_tickets', 'sum')
        ).reset_index()
    else:
        agg_usage_df = pd.DataFrame({'customer_id': [customer_df['customer_id'].iloc[0]],
                                     'avg_data_usage_gb': [0],
                                     'avg_call_minutes': [0],
                                     'total_support_tickets': [0]})

    df = customer_df.copy()
    df = pd.merge(df, agg_usage_df, on='customer_id', how='left')

    df['avg_data_usage_gb'] = df['avg_data_usage_gb'].fillna(0)
    df['avg_call_minutes'] = df['avg_call_minutes'].fillna(0)
    df['total_support_tickets'] = df['total_support_tickets'].fillna(0)

    for col in ['has_internet_service', 'has_phone_service']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # One-hot encode categorical features (ensure all categories are present as in training)
    # This requires special handling to match training set's dummy columns
    gender_dummies = pd.get_dummies(df['gender'], prefix='gender')
    contract_type_dummies = pd.get_dummies(df['contract_type'], prefix='contract_type')

    # Reindex and add missing dummy columns with 0, drop_first=True equivalent
    if 'gender_Male' not in gender_dummies.columns:
        gender_dummies['gender_Male'] = 0
    gender_dummies = gender_dummies[['gender_Male']] # Only keep the 'Male' column for gender (assuming drop_first)

    if 'contract_type_One year' not in contract_type_dummies.columns:
        contract_type_dummies['contract_type_One year'] = 0
    if 'contract_type_Two year' not in contract_type_dummies.columns:
        contract_type_dummies['contract_type_Two year'] = 0
    contract_type_dummies = contract_type_dummies[['contract_type_One year', 'contract_type_Two year']] # Keep both

    df = pd.concat([df.drop(columns=['gender', 'contract_type']), gender_dummies, contract_type_dummies], axis=1)

    df = df.drop(columns=['signup_date'], errors='ignore')

    # Ensure all model_features are present, fill missing with 0
    processed_df = pd.DataFrame(columns=model_features)
    for feature in model_features:
        if feature in df.columns:
            processed_df[feature] = df[feature]
        else:
            processed_df[feature] = 0 # Fill missing feature with 0

    # Ensure correct order of columns for prediction
    processed_df = processed_df[model_features]

    # --- FINAL SAFETY STEP (Mirroring training) ---
    for col in processed_df.columns:
        if processed_df[col].dtype == 'object':
            try:
                processed_df[col] = processed_df[col].replace(problematic_strings, pd.NA)
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0).astype(float)
            except Exception as e:
                print(f"Warning: Could not force convert column '{col}' to numeric in prediction app. Error: {e}")
    # --- END OF FINAL SAFETY STEP ---

    return processed_df

# --- Flask Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    customer_id = request.form['customer_id'].strip().upper()

    customer_df, usage_df = fetch_customer_data_by_id(customer_id)

    if customer_df.empty:
        return render_template('index.html', prediction_result=f"Customer ID '{customer_id}' not found in the database.")

    prepared_data = preprocess_new_data(customer_df, usage_df, model_features)

    if prepared_data is None or prepared_data.empty:
        return render_template('index.html', prediction_result="Failed to prepare data for prediction.")

    # Make prediction
    prediction = model.predict(prepared_data)
    prediction_proba = model.predict_proba(prepared_data)

    churn_probability = prediction_proba[0][1] # Probability of class 1 (churn)
    churn_status = "Will Churn" if prediction[0] == 1 else "Will Not Churn"

    result_message = f"Customer ID: {customer_id}<br>" \
                     f"Predicted Churn Status: <strong>{churn_status}</strong><br>" \
                     f"Probability of Churn: <strong>{churn_probability:.2f}</strong>"

    return render_template('index.html', prediction_result=result_message)

if __name__ == '__main__':
    # You can set debug=True during development to auto-reload
    # For production, set debug=False
    app.run(debug=True)