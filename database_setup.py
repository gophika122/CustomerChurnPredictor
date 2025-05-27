# database_setup.py
import sqlite3
import pandas as pd
import random
from datetime import datetime, timedelta

DATABASE_NAME = 'customer_churn.db'

def create_tables(conn):
    """Creates the necessary tables in the SQLite database."""
    cursor = conn.cursor()

    # Customers Table: Stores basic customer demographic and contract info
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Customers (
            customer_id TEXT PRIMARY KEY,
            signup_date DATE,
            gender TEXT,
            age INTEGER,
            contract_type TEXT,
            monthly_charges REAL,
            total_charges REAL,
            has_internet_service BOOLEAN,
            has_phone_service BOOLEAN
        )
    ''')

    # UsageHistory Table: Stores simulated usage data (e.g., data consumption, call minutes)
    # This will be simplified for initial setup; in real-world, it's more granular
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS UsageHistory (
            usage_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            usage_date DATE,
            data_usage_gb REAL,
            call_minutes REAL,
            num_support_tickets INTEGER,
            FOREIGN KEY (customer_id) REFERENCES Customers(customer_id)
        )
    ''')

    # ChurnLabels Table: Stores historical churn status for model training
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ChurnLabels (
            churn_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT UNIQUE,
            churn_date DATE,
            churned BOOLEAN, -- True if churned, False if still active
            FOREIGN KEY (customer_id) REFERENCES Customers(customer_id)
        )
    ''')

    # ChurnPredictions Table: To store AI-generated predictions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ChurnPredictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            prediction_date DATE,
            predicted_churn_probability REAL,
            churn_risk_category TEXT, -- e.g., 'Low', 'Medium', 'High'
            FOREIGN KEY (customer_id) REFERENCES Customers(customer_id)
        )
    ''')
    conn.commit()
    print("Tables created successfully!")

def generate_synthetic_data(num_customers=1000):
    """Generates synthetic customer and usage data."""
    customers_data = []
    usage_data = []
    churn_labels_data = []

    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31) # Simulating data up to end of 2023

    for i in range(num_customers):
        customer_id = f'CUST{i:04d}'
        signup_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
        gender = random.choice(['Male', 'Female'])
        age = random.randint(18, 70)
        contract_type = random.choice(['Month-to-month', 'One year', 'Two year'])
        monthly_charges = round(random.uniform(20, 120), 2)
        total_charges = round(monthly_charges * random.randint(1, 36), 2) # Simulate charges over tenure
        has_internet_service = random.choice([True, False])
        has_phone_service = random.choice([True, False])

        customers_data.append((
            customer_id, signup_date.strftime('%Y-%m-%d'), gender, age, contract_type,
            monthly_charges, total_charges, has_internet_service, has_phone_service
        ))

        # Simulate usage for a few months for each customer
        num_months_usage = random.randint(3, 12)
        for _ in range(num_months_usage):
            # Usage entries should generally be before the current prediction date (which we'll use as current_date)
            # For simplicity, we'll keep usage within the signup period,
            # but in a real scenario, you'd generate recent usage data too.
            usage_date = signup_date + timedelta(days=random.randint(0, (datetime.now() - signup_date).days))
            data_usage = round(random.uniform(5, 100), 2) if has_internet_service else 0.0
            call_minutes = round(random.uniform(50, 500), 2) if has_phone_service else 0.0
            num_tickets = random.randint(0, 5) # Simulate support tickets
            usage_data.append((customer_id, usage_date.strftime('%Y-%m-%d'), data_usage, call_minutes, num_tickets))

        # Simulate churn: ~30% of customers churn, mostly within the first year-two
        if random.random() < 0.3:
            churned = True
            # Churn date should be after signup, but before our current prediction date (e.g., current_date)
            # and within a reasonable timeframe (e.g., 2 years)
            churn_offset_days = random.randint(30, 730) # Churn within 2 years of signup
            churn_date = signup_date + timedelta(days=churn_offset_days)
            if churn_date > datetime.now(): # Ensure churn date is not in the future
                churn_date = datetime.now() - timedelta(days=random.randint(1,30)) # make it recent
        else:
            churned = False
            churn_date = None # No churn date if customer hasn't churned

        churn_labels_data.append((customer_id, churn_date.strftime('%Y-%m-%d') if churn_date else None, churned))

    return pd.DataFrame(customers_data, columns=[
        'customer_id', 'signup_date', 'gender', 'age', 'contract_type',
        'monthly_charges', 'total_charges', 'has_internet_service', 'has_phone_service'
    ]), pd.DataFrame(usage_data, columns=[
        'customer_id', 'usage_date', 'data_usage_gb', 'call_minutes', 'num_support_tickets'
    ]), pd.DataFrame(churn_labels_data, columns=[
        'customer_id', 'churn_date', 'churned'
    ])

def insert_data(conn, customers_df, usage_df, churn_df):
    """Inserts generated data into the database tables."""
    cursor = conn.cursor()

    # Insert Customers
    customer_records = customers_df.to_records(index=False)
    cursor.executemany(
        '''INSERT INTO Customers (customer_id, signup_date, gender, age, contract_type,
                                   monthly_charges, total_charges, has_internet_service, has_phone_service)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        customer_records
    )

    # Insert UsageHistory
    usage_records = usage_df.to_records(index=False)
    cursor.executemany(
        '''INSERT INTO UsageHistory (customer_id, usage_date, data_usage_gb, call_minutes, num_support_tickets)
           VALUES (?, ?, ?, ?, ?)''',
        usage_records
    )

    # Insert ChurnLabels (handling NULL for non-churned customers)
    for index, row in churn_df.iterrows():
        cursor.execute(
            '''INSERT INTO ChurnLabels (customer_id, churn_date, churned)
               VALUES (?, ?, ?)''',
            (row['customer_id'], row['churn_date'], 1 if row['churned'] else 0)
        )

    conn.commit()
    print("Synthetic data inserted successfully!")

def main():
    conn = None
    try:
        # Check if DB already exists, if so, remove it to start fresh for demo
        import os
        if os.path.exists(DATABASE_NAME):
            os.remove(DATABASE_NAME)
            print(f"Existing database '{DATABASE_NAME}' removed.")

        conn = sqlite3.connect(DATABASE_NAME)
        create_tables(conn)

        print("Generating synthetic data...")
        customers_df, usage_df, churn_df = generate_synthetic_data(num_customers=1000)

        print("Inserting data into database...")
        insert_data(conn, customers_df, usage_df, churn_df)

        print(f"Database '{DATABASE_NAME}' setup complete with synthetic data.")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    main()