from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer

# Initializing Flask app
app = Flask(__name__)

# Loading the model and feature names
model = joblib.load('model.pkl')
feature_names = joblib.load('feature_names.pkl')

# Defining the columns as they are required for imputation and predictions
categorical_columns = ['Make', 'Model', 'Vehicle_Class', 'Transmission']
numerical_columns = [
    'Engine_Size',
    'Cylinders',
    'Fuel_Consumption_in_City(L/100 km)',
    'Fuel_Consumption_in_City_Hwy(L/100 km)',
    'Fuel_Consumption_comb(L/100km)',
    'Smog_Level'
]
required_columns = numerical_columns + categorical_columns

# Main page
@app.route('/')
def home():
    return render_template('index.html')

# Page when 'predict' is clicked
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieving form data submitted by the user
        data = request.form.to_dict()

        # Creating a DataFrame with the feature names
        input_data = {feature: data.get(feature, np.nan) for feature in required_columns}
        input_df = pd.DataFrame([input_data])

        # Handling categorical data
        categorical_data = input_df[categorical_columns]
        numerical_data = input_df[numerical_columns]

        # Creating dummy variables for categorical features
        df_dummies = pd.get_dummies(categorical_data, drop_first=True)

        # Combining numerical and dummy features
        df_comb = pd.concat([numerical_data, df_dummies], axis=1)

        # Ensuring all features are present
        for col in feature_names:
            if col not in df_comb.columns:
                df_comb[col] = 0
        df_comb = df_comb[feature_names]

        # Imputing missing values
        imputer = SimpleImputer(strategy='mean')
        df_comb_imputed = imputer.fit_transform(df_comb)
        
        # Making prediction
        prediction = model.predict(df_comb_imputed)
        
        # Rendering the main page template with the prediction result
        prediction_text = f'Predicted CO2 Emission: {prediction[0]:.2f} g/km'
        return render_template('index.html', prediction_text=prediction_text)
    
    # Handling exceptions
    except Exception as e:
        print(f"Error occurred: {e}")  # gets info about the error to debug
        return render_template('index.html', prediction_text='An error occurred. Please try again.')

if __name__ == '__main__':
    app.run(debug=True)
