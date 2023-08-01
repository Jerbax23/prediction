import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler
import json

# Create a flask app
app = Flask(__name__)

# Load the trained model and scaler from the pickle file
with open('trained_model.pkl', 'rb') as file:
    xgb_model, scaler = pickle.load(file)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

@app.route("/predict", methods=["POST"])
# @app.route('/')


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

    # Convert categorical columns to numerical using the same encoding as in training
    input_df.replace({
        'event': {'Wedding': 1, 'Birthday': 2, 'Christening': 3, 'Anniversary': 4, 'Corporate': 5},
        'venue': {
            'The Emerald Events Place': 1, 'The Mango Farm Events Place': 2, 'Lihim ng Kubli': 3,
            'Versailles Palace': 4, 'The Madisons Events Place': 5, 'Paradisso Terrestre': 6,
            'Glass Garden': 7, 'Fernwood Gardens': 8, 'The Green Lounge': 9, 'Sitio Elena': 10,
            'Patio de Manila': 11, 'Sedretos Royale': 12, 'The Forest Barn': 13,
            'Nuevo Comienzo Resort': 14, 'The Silica Event Place': 15, 'The Circle Events Place': 16,
            'One Grand Pavillion': 17, 'Josephine Events': 18
        },
        'cuisine': {'Normal': 1, 'Deluxe': 2, 'Royal': 3},
        'style': {'Basic': 1, 'Sleek': 2, 'Polished': 3},
        'guest_number': {'1-50': 1, '51-100': 2, '101-200': 3, '201-300': 4},
        'weektype': {'weekday': 1, 'weekend': 2}
    }, inplace=True)

    # Scale the input data using the same scaler as in training
    scaled_input_data = scaler.transform(input_df)

    # Make the prediction using the trained model
    prediction = xgb_model.predict(scaled_input_data)

 # Convert the prediction to a float
    float_prediction = prediction.item()

    # Return JSON response with the prediction
    return jsonify({'prediction': float_prediction})

if __name__ == "__main__":
    app.run(debug=True)
