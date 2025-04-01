from flask import Flask, request, jsonify
import pandas as pd
from model_manager import predict_vehicle_features

app = Flask(__name__)

# Define your fields and types.
new_car_fields = [
    ('Make', 'text'),
    ('Model', 'text'),
    ('Year', 'number'),
    ('Engine Fuel Type', 'text'),
    ('Engine HP', 'number'),
    ('Engine Cylinders', 'number'),
    ('Transmission Type', 'text'),
    ('Driven_Wheels', 'text'),
    ('Number of Doors', 'number'),
    ('Market Category', 'text'),
    ('Vehicle Size', 'text'),
    ('Vehicle Style', 'text'),
    ('highway MPG', 'number'),
    ('city mpg', 'number'),
    ('Popularity', 'number'),
    ('MSRP', 'number')
]

# In-memory "database" as a list of dictionaries.
database = []

def load_csv_to_db(csv_file_path):
    """Load data from a CSV file into the in-memory database."""
    df = pd.read_csv(csv_file_path)
    # Convert dataframe rows to dictionary records.
    records = df.to_dict(orient='records')
    for record in records:
        # Convert and clean data based on new_car_fields.
        for field, field_type in new_car_fields:
            if field in record:
                if field_type == 'number':
                    try:
                        record[field] = float(record[field])
                    except ValueError:
                        record[field] = None
                else:
                    record[field] = str(record[field])
        database.append(record)

# Load the CSV file when the server starts.
load_csv_to_db("final_vehicle_data.csv")  # Ensure the CSV file is in your working directory.

@app.route('/cars', methods=['GET'])
def get_cars():
    """Return the entire database as JSON."""
    return jsonify(database)

@app.route('/cars', methods=['POST'])
def add_car():
    """Add a new car entry to the database."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400

    new_car = {}
    # Validate and process each field.
    for field, field_type in new_car_fields:
        if field not in data:
            return jsonify({'error': f'Missing field: {field}'}), 400

        value = data[field]
        # Convert types as needed.
        if field_type == 'number':
            try:
                value = float(value)
            except ValueError:
                return jsonify({'error': f'Field "{field}" must be a number'}), 400
        else:
            value = str(value)
        new_car[field] = value

    # Use models to assign additional fields
    new_car_df = pd.DataFrame([new_car])
    new_car_df = predict_vehicle_features(new_car_df)
    updated_car = new_car_df.iloc[0].to_dict()

    # Calculate value_score and meta_value_score for new vehicles
    cluster = updated_car['cluster']
    meta_cluster = updated_car['meta_cluster']
    msrp = updated_car['MSRP']

    cluster_msrps = [car['MSRP'] for car in database if car.get('cluster') == cluster and car.get('MSRP') is not None]
    cluster_msrps.append(msrp)
    cluster_avg = sum(cluster_msrps) / len(cluster_msrps)

    value_score = msrp / cluster_avg if cluster_avg != 0 else None

    meta_value_scores = [
        car['value_score'] for car in database
        if car.get('meta_cluster') == meta_cluster and car.get('value_score') is not None
    ]
    meta_value_scores.append(value_score)
    mean_meta_value_score = sum(meta_value_scores) / len(meta_value_scores) if meta_value_scores else None
    meta_value_score = value_score / mean_meta_value_score if mean_meta_value_score != 0 else None

    # Add calculated fields to new car
    updated_car['cluster_avg'] = cluster_avg
    updated_car['value_score'] = value_score
    updated_car['meta_value_score'] = meta_value_score

    print(updated_car)

    # Add the new car to the in-memory database.
    database.append(updated_car)
    return jsonify(database), 201

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1')
