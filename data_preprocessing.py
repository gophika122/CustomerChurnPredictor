# data_preprocessing.py
import pandas as pd
import sqlite3
from datetime import datetime

DATABASE_NAME = 'customer_churn.db'

def fetch_data_from_db():
    """Fetches raw data from the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        # Fetch Customers data
        customers_df = pd.read_sql_query("SELECT * FROM Customers", conn)
        # Fetch UsageHistory data
        usage_df = pd.read_sql_query("SELECT * FROM UsageHistory", conn)
        # Fetch ChurnLabels data
        churn_df = pd.read_sql_query("SELECT * FROM ChurnLabels", conn)
        print("Raw data fetched from database successfully.")
        return customers_df, usage_df, churn_df
    except sqlite3.Error as e:
        print(f"Error fetching data from database: {e}")
        return None, None, None
    finally:
        if conn:
            conn.close()

def preprocess_data(customers_df, usage_df, churn_df, prediction_date=None):
    """
    Performs feature engineering and merges data into a single DataFrame.
    `prediction_date` is used to calculate tenure relative to a specific point in time.
    If None, it uses the current system date.
    """
    if customers_df is None or usage_df is None or churn_df is None:
        print("Input DataFrames are None. Cannot preprocess.")
        return None

    # --- 1. Process Customers Data ---
    customers_df['signup_date'] = pd.to_datetime(customers_df['signup_date'])
    current_date = prediction_date if prediction_date else datetime.now()

    # Calculate tenure (duration customer has been with the service) in months
    customers_df['tenure_months'] = (current_date - customers_df['signup_date']).dt.days // 30
    customers_df['tenure_months'] = customers_df['tenure_months'].apply(lambda x: max(x, 0)) # Tenure can't be negative

    # --- START OF CRITICAL FIX FOR ALL POTENTIAL NUMERIC BYTE STRINGS (Initial pass) ---
    # Robustly handle monthly_charges, total_charges, and age from Customers table
    for col in ['monthly_charges', 'total_charges', 'age']:
        # Step 1: Decode any byte strings to regular Python strings
        customers_df[col] = customers_df[col].apply(
            lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
        )
        # NEW: Explicitly replace specific problematic string patterns with NaN BEFORE to_numeric
        # This targets the exact patterns seen in errors: b'-\x00...', b'F\x00...', b'\x05\x00...'
        # After decode, they would be regular strings like '-\x00...' etc.
        problematic_strings = [
            '-\x00\x00\x00\x00\x00\x00\x00',
            'F\x00\x00\x00\x00\x00\x00\x00',
            '\x05\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00'
        ]
        customers_df[col] = customers_df[col].replace(problematic_strings, pd.NA) # Use pd.NA for nullable type

        # Step 2: Convert to numeric, coercing any errors to NaN, then fill NaN with 0, and ensure float type
        customers_df[col] = pd.to_numeric(customers_df[col], errors='coerce').fillna(0).astype(float)
    # --- END OF CRITICAL FIX FOR ALL POTENTIAL NUMERIC BYTE STRINGS ---

    # Special handling for total_charges where it's 0 but monthly_charges is not
    customers_df.loc[(customers_df['total_charges'] == 0) & (customers_df['monthly_charges'] > 0), 'total_charges'] = customers_df['monthly_charges']


    # --- 2. Process Usage History Data ---
    usage_df['usage_date'] = pd.to_datetime(usage_df['usage_date'])

    # Aggregate usage data per customer: calculate average usage and total support tickets
    agg_usage_df = usage_df.groupby('customer_id').agg(
        avg_data_usage_gb=('data_usage_gb', 'mean'),
        avg_call_minutes=('call_minutes', 'mean'),
        total_support_tickets=('num_support_tickets', 'sum')
    ).reset_index()

    # --- 3. Process Churn Labels ---
    # Convert 'churned' in churn_df to 0 or 1 integers robustly.
    churn_df['churned'] = pd.to_numeric(churn_df['churned'], errors='coerce').fillna(0).astype(int)

    # --- 4. Merge all DataFrames ---
    # Start with customers_df
    df = customers_df.copy()

    # Merge with aggregated usage data
    df = pd.merge(df, agg_usage_df, on='customer_id', how='left')

    # Merge with churn labels (this is our target variable for training)
    df = pd.merge(df, churn_df[['customer_id', 'churned']], on='customer_id', how='left')

    # Handle potential NaNs created by left merges for customers without usage history
    df['avg_data_usage_gb'] = df['avg_data_usage_gb'].fillna(0)
    df['avg_call_minutes'] = df['avg_call_minutes'].fillna(0)
    df['total_support_tickets'] = df['total_support_tickets'].fillna(0)

    # Convert boolean columns (has_internet_service, has_phone_service) to integer (0 or 1) for ML models
    for col in ['has_internet_service', 'has_phone_service']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['gender', 'contract_type'], drop_first=True)

    # Drop original date columns as tenure is derived
    df = df.drop(columns=['signup_date'], errors='ignore')

    # Ensure 'churned' column is the target variable (0 or 0 for now until loaded)
    df['churned'] = pd.to_numeric(df['churned'], errors='coerce').fillna(-1).astype(int)

    # --- FINAL SAFETY STEP: ENSURE ALL EXPECTED NUMERICAL COLUMNS ARE NUMERIC ---
    # This will catch any remaining non-numeric types that might have slipped through
    # It attempts to convert all columns that are NOT 'customer_id' or string/object dtypes
    # to float, coercing errors to NaN and filling with 0.
    for col in df.columns:
        if df[col].dtype == 'object' and col != 'customer_id': # Check for object dtype (can contain mixed types like strings/bytes)
            try:
                # NEW: Explicitly replace specific problematic string patterns with NaN
                df[col] = df[col].replace(problematic_strings, pd.NA) # Use pd.NA here as well

                # Attempt to convert to numeric, coerce errors, fill NaNs, and convert to float
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
            except Exception as e:
                print(f"Warning: Could not force convert column '{col}' to numeric. Error: {e}")
    # --- END OF FINAL SAFETY STEP ---

    print("Data preprocessing and feature engineering complete.")
    return df

def main():
    customers_df, usage_df, churn_df = fetch_data_from_db()
    if customers_df is not None:
        processed_df = preprocess_data(customers_df, usage_df, churn_df)
        if processed_df is not None:
            print("\nSample of the final processed DataFrame:")
            print(processed_df.head())
            print(f"\nShape of the processed DataFrame: {processed_df.shape}")
            print("\nMissing values in processed DataFrame (should be mostly 0 for training):")
            print(processed_df.isnull().sum())
            return processed_df
    return None

if __name__ == '__main__':
    final_dataset = main()