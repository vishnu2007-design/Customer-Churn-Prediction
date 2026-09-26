# Import Flask classes needed to create the web application
from flask import Flask, render_template, request

# Import pandas to create a DataFrame from customer input
import pandas as pd

# Import joblib to load the saved ML model and preprocessor
import joblib

# Import os to create reliable file paths
import os


# Get the folder where this app.py file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Create the Flask application
app = Flask(__name__)


# Load the final trained Logistic Regression model
final_model = joblib.load(
    os.path.join(BASE_DIR, 'final_churn_model.pkl')
)


# Load the preprocessing transformer
# This is the same preprocessor used during model training
preprocessor = joblib.load(
    os.path.join(BASE_DIR, 'churn_preprocessor.pkl')
)


# Create the function used during feature engineering
# to convert numerical tenure into tenure groups
def create_tenure_group(tenure):

    # Customers with 12 months or less are considered New
    if tenure <= 12:
        return 'New'

    # Customers from 13 to 24 months are considered Short-term
    elif tenure <= 24:
        return 'Short-term'

    # Customers from 25 to 48 months are considered Medium-term
    elif tenure <= 48:
        return 'Medium-term'

    # Customers with more than 48 months are considered Long-term
    else:
        return 'Long-term'


# Create the home page route
@app.route('/')
def home():

    # Display the HTML form
    return render_template('index.html')


# Create the prediction route
# This route receives the customer information from the form
@app.route('/predict', methods=['POST'])
def predict():

    # Get the customer input from the HTML form
    gender = request.form['gender']
    senior_citizen = int(request.form['SeniorCitizen'])
    partner = request.form['Partner']
    dependents = request.form['Dependents']
    tenure = float(request.form['tenure'])
    phone_service = request.form['PhoneService']
    multiple_lines = request.form['MultipleLines']
    internet_service = request.form['InternetService']
    online_security = request.form['OnlineSecurity']
    online_backup = request.form['OnlineBackup']
    device_protection = request.form['DeviceProtection']
    tech_support = request.form['TechSupport']
    streaming_tv = request.form['StreamingTV']
    streaming_movies = request.form['StreamingMovies']
    contract = request.form['Contract']
    paperless_billing = request.form['PaperlessBilling']
    payment_method = request.form['PaymentMethod']
    monthly_charges = float(request.form['MonthlyCharges'])
    total_charges = float(request.form['TotalCharges'])


    # Store the six service columns
    # These were also used to create TotalServices during training
    service_values = [
        online_security,
        online_backup,
        device_protection,
        tech_support,
        streaming_tv,
        streaming_movies
    ]


    # Count how many services have the value "Yes"
    # This recreates the TotalServices feature used during training
    total_services = service_values.count('Yes')


    # Create the TenureGroup feature
    # using the same rules that were used during training
    tenure_group = create_tenure_group(tenure)


    # Create a dictionary containing all model input features
    customer_data = {
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
        'TotalServices': total_services,
        'TenureGroup': tenure_group
    }


    # Convert the dictionary into a pandas DataFrame
    # The model expects data in DataFrame format
    customer_df = pd.DataFrame([customer_data])


    # Apply the saved preprocessing transformer
    # This converts categorical values into the same encoded format
    # that was used when training the model
    customer_processed = preprocessor.transform(customer_df)


    # Make the churn prediction using the final model
    prediction = final_model.predict(customer_processed)[0]


    # Get the probability of churn
    # Index 1 represents the probability of Churn = 1
    churn_probability = final_model.predict_proba(
        customer_processed
    )[0][1]


    # Convert the numerical prediction into a readable message
    if prediction == 1:

        # Customer is predicted to churn
        result = "Customer is likely to churn."

    else:

        # Customer is predicted to stay
        result = "Customer is likely to stay."


    # Convert the probability into a percentage
    churn_probability_percentage = round(
        churn_probability * 100,
        2
    )


    # Display the prediction result on the webpage
    return render_template(
        'index.html',
        prediction=result,
        probability=churn_probability_percentage
    )


# Run the Flask application
if __name__ == '__main__':

    # Start the Flask development server
    app.run(debug=True)