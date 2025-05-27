# churn_model_training.py
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import sqlite3

# Import data preprocessing function
from data_preprocessing import fetch_data_from_db, preprocess_data

# Import SMOTE for handling imbalanced data
from imblearn.over_sampling import SMOTE

# Define paths for saving model and features
MODEL_PATH = 'churn_model.joblib'
FEATURES_PATH = 'model_features.joblib'

def train_model(df):
    """Trains the XGBoost Classifier model with SMOTE for data balancing."""
    # Separate features (X) and target (y)
    X = df.drop(columns=['customer_id', 'churned']) # customer_id is not a feature
    y = df['churned']

    # Store feature names for prediction (important for consistent column order)
    model_features = X.columns.tolist()
    joblib.dump(model_features, FEATURES_PATH)
    print("Feature columns saved to model_features.joblib")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training data shape (before SMOTE): {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print("\nChurn distribution in training set (before SMOTE):")
    print(y_train.value_counts(normalize=True))
    print("\nChurn distribution in testing set:")
    print(y_test.value_counts(normalize=True))

    # --- APPLYING SMOTE (Data Balancing) ---
    print("\nApplying SMOTE to balance the training data...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"Training data shape (after SMOTE): {X_train_resampled.shape}")
    print("\nChurn distribution in training set (after SMOTE):")
    print(y_train_resampled.value_counts(normalize=True))
    # --- END OF SMOTE ---


    # --- MODEL TRAINING: Using XGBoost Classifier with balanced data and TUNED PARAMETERS ---
    print("\nTraining XGBoost Classifier with balanced data (Tuned Parameters)...")
    # Initialize the XGBoost Classifier with NEW parameters
    model = XGBClassifier(objective='binary:logistic',  # For binary classification tasks
                          n_estimators=200,           # Tuned: Increased from 100 to 200
                          learning_rate=0.05,         # Tuned: Decreased from 0.1 to 0.05
                          max_depth=5,                # Tuned: Added max_depth for more control
                          use_label_encoder=False,    # Suppress a future deprecation warning
                          eval_metric='logloss',      # Evaluation metric during training (for internal use)
                          random_state=42)            # For reproducibility of results

    # Train the model using the resampled data
    model.fit(X_train_resampled, y_train_resampled) # Use resampled data here
    print("Model training complete.")
    # --- END OF MODEL TRAINING ---


    # Evaluate the model
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability of the positive class (churn)

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    class_report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.3f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("Classification Report:")
    print(class_report)

    # Save the trained model
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved as {MODEL_PATH}")

def main():
    print("Starting data fetching and preprocessing for model training...")
    customers_df, usage_df, churn_df = fetch_data_from_db()

    if customers_df is not None:
        processed_df = preprocess_data(customers_df, usage_df, churn_df)
        if processed_df is not None:
            # Ensure churned column is present and is the correct type
            if 'churned' not in processed_df.columns:
                print("Error: 'churned' column not found in processed data.")
                return
            if processed_df['churned'].isnull().any():
                print("Warning: 'churned' column contains NaN values. Filling with -1.")
                processed_df['churned'] = processed_df['churned'].fillna(-1).astype(int)
            # Filter out rows where churned is -1 (if any were introduced due to missing labels)
            processed_df = processed_df[processed_df['churned'] != -1]

            # Display basic info of processed data before training
            print("\nSample of the final processed DataFrame:")
            print(processed_df.head())
            print(f"\nShape of the processed DataFrame: {processed_df.shape}")
            print("\nMissing values in processed DataFrame (should be mostly 0 for training):")
            print(processed_df.isnull().sum())

            train_model(processed_df)
        else:
            print("Data preprocessing failed.")
    else:
        print("Data fetching failed.")

if __name__ == '__main__':
    main()