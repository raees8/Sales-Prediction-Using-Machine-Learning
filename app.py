from flask import Flask, request, render_template
import numpy as np
import pickle
import os

# Load the trained model
model_path = 'F:\\Vision Project\\bigmart pred\\big_model.pkl'
model = pickle.load(open(model_path, 'rb'))

# Create Flask app
app = Flask(__name__)

# Function to preprocess input data
def preprocess_input(form_data):
    try:
        # Define expected numeric fields
        numeric_fields = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year']

        # Define expected categorical fields
        categorical_fields = ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 
                              'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

        # Convert numeric fields
        input_features = []
        for field in numeric_fields:
            input_features.append(float(form_data[field]))  # Convert to float

        # Encode categorical fields (assuming you used label encoding during training)
        for field in categorical_fields:
            input_features.append(hash(form_data[field]) % 1000)  # Simple hash-based encoding

        # Convert to NumPy array
        input_array = np.array(input_features).reshape(1, -1)
        return input_array

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

# Define prediction route
@app.route('/', methods=['GET'])
def home():
    return render_template('practice.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form
        input_data = preprocess_input(form_data)

        if input_data is None:
            return render_template('practice.html', message="Invalid input data")

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Format the prediction output
        formatted_prediction = f"Predicted Sales: ₹{round(prediction, 2)}"

        return render_template('practice.html', message=formatted_prediction)

    except Exception as e:
        print(f"Prediction error: {e}")
        return render_template('practice.html', message="Error in prediction")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
