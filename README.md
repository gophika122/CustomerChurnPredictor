# CustomerChurnPredictor
# Customer Churn Prediction Web Application

This project develops a customer churn prediction web application. It leverages machine learning (XGBoost) to predict if a customer is likely to churn, helping businesses identify at-risk customers.

The application includes data simulation, preprocessing, model training, and a Flask-based web interface for real-time churn prediction.

## Features

* **Data Simulation:** Generates synthetic customer, usage, and churn data to populate an SQLite database.
* **Data Preprocessing:** Cleans, transforms, and engineers features from the raw data, including handling categorical variables.
* **XGBoost Model Training:** Trains a robust churn prediction model using the XGBoost Classifier. It incorporates SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance in the training data, improving the model's ability to identify churners.
* **Model Evaluation:** Provides key performance metrics such as Accuracy, ROC-AUC Score, and a Classification Report to assess the model's effectiveness.
* **Flask Web Application:** A user-friendly web interface that allows users to input a customer ID and get an instant churn prediction (Will Churn/Will Not Churn) along with the probability.
* **Data Visualization:** Generates insightful charts (e.g., Predicted Churn Distribution, Predicted Churn by Gender) to visualize the model's predictions and underlying churn patterns.

## Technologies Used

* **Python 3.x**
* **Flask:** Web framework for the application.
* **Pandas:** Data manipulation and analysis.
* **Scikit-learn:** Machine learning utilities (e.g., `train_test_split`).
* **XGBoost:** Gradient Boosting machine learning algorithm for the prediction model.
* **Imblearn (Scikit-learn-contrib):** Specifically `SMOTE` for handling imbalanced datasets.
* **Matplotlib & Seaborn:** For data visualization.
* **SQLite3:** Lightweight relational database for storing customer data.

## Setup and Installation

Follow these steps to get the project up and running on your local machine.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/gophika122/CustomerChurnPredictor.git](https://github.com/gophika122/CustomerChurnPredictor.git)
    cd CustomerChurnPredictor
    ```

2.  **Create a virtual environment (recommended):**
    A virtual environment helps manage project dependencies.
    ```bash
    python -m venv venv
    ```
    * **Activate the virtual environment:**
        * **For Windows:**
            ```bash
            .\venv\Scripts\activate
            ```
        * **For macOS/Linux:**
            ```bash
            source venv/bin/activate
            ```

3.  **Install dependencies:**
    Once your virtual environment is active, install all required Python packages. First, generate the `requirements.txt` file (if you haven't already and pushed it to Git), then install:
    ```bash
    pip freeze > requirements.txt # Run this if requirements.txt is not in your repo
    pip install -r requirements.txt
    ```

4.  **Set up the database and train the model:**
    This will create the `customer_churn.db` file and train the `churn_model.joblib`.
    ```bash
    python database_setup.py
    python churn_model_training.py
    ```

5.  **Run the Flask web application:**
    ```bash
    python app.py
    ```
    Access the application in your web browser at: `http://127.0.0.1:5000`

6.  **Run visualization script (optional):**
    To generate the churn distribution and gender-based churn charts, run:
    ```bash
    python churn_analysis_and_viz.py
    ```

## Usage

* Navigate to `http://127.0.0.1:5000` in your web browser.
* Enter an existing Customer ID (e.g., `CUST0001`) in the input field.
* Click the "Predict Churn" button to view the prediction status and probability.

## Future Enhancements

* Implement more advanced hyperparameter tuning techniques (e.g., GridSearchCV, RandomizedSearchCV) for the XGBoost model to further optimize performance.
* Integrate additional interactive dashboards and visualizations for deeper insights into churn drivers.
* Consider deploying the application to a cloud platform (e.g., Heroku, AWS, Azure).

## Author

* **Your Name**

---
