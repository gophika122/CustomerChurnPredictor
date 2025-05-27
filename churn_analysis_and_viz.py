# churn_analysis_and_viz.py
import pandas as pd
import sqlite3
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import fetch_data_from_db, preprocess_data # Use existing preprocessing

# --- Configuration ---
DATABASE_NAME = 'customer_churn.db'
MODEL_PATH = 'churn_model.joblib'
FEATURES_PATH = 'model_features.joblib'

def analyze_and_visualize_predictions():
    """
    Fetches processed data, makes predictions, and generates visualizations
    of churn predictions and churn by gender.
    """
    print("Starting analysis and visualization...")

    # Load model and features (make sure they are trained/saved)
    try:
        model = joblib.load(MODEL_PATH)
        model_features = joblib.load(FEATURES_PATH)
        print("Model and features loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model or feature file not found. Please ensure '{MODEL_PATH}' and '{FEATURES_PATH}' exist and model is trained.")
        return

    # Fetch and preprocess all data for prediction
    print("Fetching and preprocessing all data for analysis...")
    customers_df, usage_df, churn_df = fetch_data_from_db()

    if customers_df.empty:
        print("No customer data found in the database. Cannot perform analysis.")
        return

    processed_df = preprocess_data(customers_df, usage_df, churn_df)

    if processed_df is None or processed_df.empty:
        print("Failed to preprocess data for analysis.")
        return

    # Drop 'churned' column from processed_df before making predictions if it exists
    # We want to predict churn, not use the actual churn for prediction features.
    X_predict = processed_df.drop(columns=['customer_id', 'churned'], errors='ignore')

    # Ensure columns match training features
    # Create an empty DataFrame with the exact columns used during training
    # Then fill it with data from X_predict
    X_final = pd.DataFrame(columns=model_features)
    for feature in model_features:
        if feature in X_predict.columns:
            X_final[feature] = X_predict[feature]
        else:
            X_final[feature] = 0 # Fill missing features (e.g., if a dummy variable wasn't present in this batch)

    # Ensure all columns are numeric
    for col in X_final.columns:
        if X_final[col].dtype == 'object':
            X_final[col] = pd.to_numeric(X_final[col], errors='coerce').fillna(0).astype(float)


    print("Making predictions for all customers...")
    # Make predictions
    predictions = model.predict(X_final)
    prediction_probas = model.predict_proba(X_final)[:, 1] # Probability of churn

    # Add predictions back to the original processed_df for analysis
    processed_df['predicted_churn'] = predictions
    processed_df['churn_probability'] = prediction_probas

    print("Predictions complete. Generating visualizations...")

    # --- Visualization 1: Predicted Churn Distribution ---
    plt.figure(figsize=(8, 6))
    sns.countplot(x='predicted_churn', data=processed_df, palette='viridis')
    plt.title('Predicted Churn Distribution')
    plt.xlabel('Predicted Churn (0: No Churn, 1: Will Churn)')
    plt.ylabel('Number of Customers')
    plt.xticks(ticks=[0, 1], labels=['Will Not Churn', 'Will Churn'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    print("Predicted Churn Distribution chart displayed.")

    # --- Visualization 2: Predicted Churn by Gender ---
    # Need to get gender back from the processed data
    # We had gender_Male, so we can infer original gender
    processed_df['gender_original'] = processed_df['gender_Male'].apply(lambda x: 'Male' if x == 1 else 'Female')

    plt.figure(figsize=(10, 7))
    sns.countplot(x='gender_original', hue='predicted_churn', data=processed_df, palette='coolwarm')
    plt.title('Predicted Churn by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Number of Customers')
    plt.legend(title='Predicted Churn', labels=['Will Not Churn', 'Will Churn'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    print("Predicted Churn by Gender chart displayed.")

    print("Analysis and visualization complete.")

if __name__ == '__main__':
    analyze_and_visualize_predictions()