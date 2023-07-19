from flask import Flask
app = Flask(__name__)
@app.route('/')
def hello_word():
    return 'jeupri'

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