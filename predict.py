import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler
import json

# Create a flask app
app = Flask(__name__)

# Load the trained model and scaler from the pickle file
with open('model.pkl', 'rb') as file:
    xgb_model, scaler = pickle.load(file)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

@app.route("/predict", methods=["POST"])
def predict():
    # Define the column names of the input features
    feature_names = ['event', 'venue', 'cuisine', 'style', 'guest_number', 'weektype', 'dj_services',
                    'emcee', 'photog', 'videog', 'm_artist', 'bar_area', 'inv_cards']
    
    # Get the input data from the request
    input_data = request.get_json()

    # Convert the input data to a DataFrame
    input_df = pd.DataFrame.from_dict([input_data])

    # Set column names for input DataFrame
    input_df.columns = feature_names

    # Scale the input data using the same scaler as in training
    scaled_input_data = scaler.transform(input_df)

    # Make the prediction using the trained model
    prediction = xgb_model.predict(scaled_input_data)

    # Convert the prediction to a float
    float_prediction = prediction.item()

    # Return JSON response with the prediction
    # return render_template('budget-output.php', jsonify({'prediction': float_prediction}))
    return jsonify({'prediction': float_prediction})

if __name__ == "__main__":
    app.run(debug=True)