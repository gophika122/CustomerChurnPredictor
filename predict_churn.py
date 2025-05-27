# predict_churn.py
import pandas as pd
import sqlite3
import joblib
import os
from datetime import datetime
import data_preprocessing # Import our preprocessing script

DATABASE_NAME = 'customer_churn.db'
MODEL_FILENAME = 'churn_model.joblib'
PREPROCESSOR_COLUMNS_FILENAME = 'model_features.joblib'

def load_model():
    """Loads the trained machine learning model."""
    if not os.path.exists(MODEL_FILENAME):
        print(f"Error: Model file '{MODEL_FILENAME}' not found. Please train the model first by running churn_model_training.py")
        return None
    if not os.path.exists(PREPROCESSOR_COLUMNS_FILENAME):
        print(f"Error: Feature columns file '{PREPROCESSOR_COLUMNS_FILENAME}' not found. Please train the model first.")
        return None

    model = joblib.load(MODEL_FILENAME)
    feature_columns = joblib.load(PREPROCESSOR_COLUMNS_FILENAME)
    print(f"Model and feature columns loaded successfully from {MODEL_FILENAME} and {PREPROCESSOR_COLUMNS_FILENAME}.")
    return model, feature_columns

def fetch_customers_for_prediction():
    """Fetches all customer data for prediction."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        customers_df = pd.read_sql_query("SELECT * FROM Customers", conn)
        usage_df = pd.read_sql_query("SELECT * FROM UsageHistory", conn)
        churn_df = pd.read_sql_query("SELECT customer_id, churned FROM ChurnLabels", conn) # Only relevant columns
        print("Customer data fetched for prediction.")
        return customers_df, usage_df, churn_df
    except sqlite3.Error as e:
        print(f"Error fetching data for prediction: {e}")
        return None, None, None
    finally:
        if conn:
            conn.close()

def save_predictions_to_db(predictions_df):
    """Saves the churn predictions to the ChurnPredictions table."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()

        # Clear existing predictions for demo (optional, you might append in real apps)
        # cursor.execute("DELETE FROM ChurnPredictions")
        # conn.commit()
        # print("Cleared existing predictions from ChurnPredictions table.")

        # Insert new predictions
        for index, row in predictions_df.iterrows():
            cursor.execute(
                '''INSERT INTO ChurnPredictions (customer_id, prediction_date, predicted_churn_probability, churn_risk_category)
                   VALUES (?, ?, ?, ?)''',
                (row['customer_id'], row['prediction_date'], row['predicted_churn_probability'], row['churn_risk_category'])
            )
        conn.commit()
        print(f"Successfully saved {len(predictions_df)} predictions to ChurnPredictions table.")
    except sqlite3.Error as e:
        print(f"Error saving predictions to database: {e}")
    finally:
        if conn:
            conn.close()

def main():
    # 1. Load the trained model
    model, feature_columns = load_model()
    if model is None:
        return

    # 2. Fetch current customer data for prediction
    customers_df, usage_df, churn_df = fetch_customers_for_prediction()
    if customers_df is None:
        return

    # 3. Preprocess the data for prediction (using the same logic as training)
    # We want to predict for ALL customers, including those with a known churn status
    # The 'churned' column in the processed_df will be -1 for customers without a label
    # or for customers we want to predict 'future' churn for.
    processed_df = data_preprocessing.preprocess_data(customers_df, usage_df, churn_df, prediction_date=datetime.now())

    if processed_df is None or processed_df.empty:
        print("No data to make predictions.")
        return

    # Drop customers who already have a confirmed churn label for this specific prediction run
    # OR, if you want to predict future churn, you'd filter out customers who churned *before* a certain date
    # For simplicity, let's predict for all customers in the processed_df.
    # We need to make sure the features align with what the model was trained on.
    
    # Filter out customer_id and the actual churned label if present
    X_predict = processed_df.drop(columns=['customer_id', 'churned'], errors='ignore')

    # Ensure all feature columns are present and in the correct order as during training
    for col in feature_columns:
        if col not in X_predict.columns:
            X_predict[col] = 0 # Add missing dummy variables as zeros
    X_predict = X_predict[feature_columns] # Reorder columns to match training data


    # 4. Make predictions
    print("Making churn predictions...")
    predictions_proba = model.predict_proba(X_predict)[:, 1] # Probability of churn (class 1)

    # 5. Prepare predictions DataFrame
    predictions_df = pd.DataFrame({
        'customer_id': processed_df['customer_id'],
        'prediction_date': datetime.now().strftime('%Y-%m-%d'),
        'predicted_churn_probability': predictions_proba
    })

    # Add churn risk category
    predictions_df['churn_risk_category'] = predictions_df['predicted_churn_probability'].apply(
        lambda p: 'High' if p > 0.7 else ('Medium' if p > 0.4 else 'Low')
    )

    print("\nSample of predictions:")
    print(predictions_df.head())

    # 6. Save predictions to the database
    save_predictions_to_db(predictions_df)

    print("\nChurn prediction process complete. Check 'customer_churn.db' for results.")

if __name__ == '__main__':
    main()