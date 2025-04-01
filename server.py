from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
import pandas as pd
from model_manager import predict_vehicle_features

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cars.db'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cars.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define the Car model.
class Car(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    Make = db.Column(db.String(80))
    Model = db.Column(db.String(80))
    Year = db.Column(db.Float)
    Engine_Fuel_Type = db.Column(db.String(80))
    Engine_HP = db.Column(db.Float)
    Engine_Cylinders = db.Column(db.Float)
    Transmission_Type = db.Column(db.String(80))
    Driven_Wheels = db.Column(db.String(80))
    Number_of_Doors = db.Column(db.Float)
    Market_Category = db.Column(db.String(80))
    Vehicle_Size = db.Column(db.String(80))
    Vehicle_Style = db.Column(db.String(80))
    highway_MPG = db.Column(db.Float)
    city_mpg = db.Column(db.Float)
    Popularity = db.Column(db.Float)
    MSRP = db.Column(db.Float)
    # Fields expected to be set by the ML model.
    cluster = db.Column(db.String(80))
    P1 = db.Column(db.Float)
    P2 = db.Column(db.Float)
    P3 = db.Column(db.Float)
    meta_cluster = db.Column(db.String(80))
    # Computed fields.
    cluster_avg = db.Column(db.Float)
    value_score = db.Column(db.Float)
    meta_value_score = db.Column(db.Float)

    def to_dict(self):
        return {
            'id': self.id,
            'Make': self.Make,
            'Model': self.Model,
            'Year': self.Year,
            'Engine Fuel Type': self.Engine_Fuel_Type,
            'Engine HP': self.Engine_HP,
            'Engine Cylinders': self.Engine_Cylinders,
            'Transmission Type': self.Transmission_Type,
            'Driven_Wheels': self.Driven_Wheels,
            'Number of Doors': self.Number_of_Doors,
            'Market Category': self.Market_Category,
            'Vehicle Size': self.Vehicle_Size,
            'Vehicle Style': self.Vehicle_Style,
            'highway MPG': self.highway_MPG,
            'city mpg': self.city_mpg,
            'Popularity': self.Popularity,
            'MSRP': self.MSRP,
            'cluster': self.cluster,
            'P1': self.P1,
            'P2': self.P2,
            'P3': self.P3,
            'meta_cluster': self.meta_cluster,
            'cluster_avg': self.cluster_avg,
            'value_score': self.value_score,
            'meta_value_score': self.meta_value_score
        }

# Create the database tables.
with app.app_context():
    db.create_all()

def load_csv_to_db(csv_file_path):
    df = pd.read_csv(csv_file_path)
    for index, row in df.iterrows():
        car = Car(
            Make=str(row['Make']),
            Model=str(row['Model']),
            Year=float(row['Year']),
            Engine_Fuel_Type=str(row['Engine Fuel Type']),
            Engine_HP=float(row['Engine HP']),
            Engine_Cylinders=float(row['Engine Cylinders']),
            Transmission_Type=str(row['Transmission Type']),
            Driven_Wheels=str(row['Driven_Wheels']),
            Number_of_Doors=float(row['Number of Doors']),
            Market_Category=str(row['Market Category']),
            Vehicle_Size=str(row['Vehicle Size']),
            Vehicle_Style=str(row['Vehicle Style']),
            highway_MPG=float(row['highway MPG']),
            city_mpg=float(row['city mpg']),
            Popularity=float(row['Popularity']),
            MSRP=float(row['MSRP']),
            cluster=float(row['cluster']),
            P1=float(row['P1']),
            P2=float(row['P2']),
            P3=float(row['P3']),
            meta_cluster=float(row['meta_cluster']),
            cluster_avg=float(row['cluster_avg']),
            value_score=float(row['value_score']),
            meta_value_score=float(row['meta_value_score'])
        )
        db.session.add(car)
    db.session.commit()


@app.route('/cars', methods=['GET'])
def get_cars():
    """Return the entire database as JSON."""
    cars = Car.query.all()
    return jsonify([car.to_dict() for car in cars])

@app.route('/cars', methods=['POST'])
def add_car():
    """Add a new car entry to the database."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400

    try:
        new_car = {
            'Make': str(data['Make']),
            'Model': str(data['Model']),
            'Year': float(data['Year']),
            'Engine Fuel Type': str(data['Engine Fuel Type']),
            'Engine HP': float(data['Engine HP']),
            'Engine Cylinders': float(data['Engine Cylinders']),
            'Transmission Type': str(data['Transmission Type']),
            'Driven_Wheels': str(data['Driven_Wheels']),
            'Number of Doors': float(data['Number of Doors']),
            'Market Category': str(data['Market Category']),
            'Vehicle Size': str(data['Vehicle Size']),
            'Vehicle Style': str(data['Vehicle Style']),
            'highway MPG': float(data['highway MPG']),
            'city mpg': float(data['city mpg']),
            'Popularity': float(data['Popularity']),
            'MSRP': float(data['MSRP'])
        }
    except (KeyError, ValueError) as e:
        return jsonify({'error': f'Invalid or missing field: {str(e)}'}), 400

    # Use models to assign additional fields
    new_car_df = pd.DataFrame([new_car])
    new_car_df = predict_vehicle_features(new_car_df)
    updated_car = new_car_df.iloc[0].to_dict()

    # Calculate value_score and meta_value_score for new vehicles
    cluster = updated_car['cluster']
    meta_cluster = updated_car['meta_cluster']
    msrp = updated_car['MSRP']

    if cluster is None or meta_cluster is None or msrp is None:
        return jsonify({'error': 'Missing cluster, meta_cluster, or MSRP from ML prediction'}), 400
    
    # --- Compute Additional Fields ---
    # Compute cluster_avg: the average MSRP for vehicles in the same cluster,
    # including the new vehicle.
    cluster_avg_result = db.session.query(func.avg(Car.MSRP)).filter(Car.cluster == cluster).scalar()
    count_cluster = db.session.query(func.count(Car.id)).filter(Car.cluster == cluster).scalar() or 0
    if count_cluster == 0 or cluster_avg_result is None:
        cluster_avg = msrp
    else:
        cluster_avg = (cluster_avg_result * count_cluster + msrp) / (count_cluster + 1)

    # Compute value_score: the ratio of MSRP to the cluster average.
    value_score = msrp / cluster_avg if cluster_avg != 0 else None

    # Compute meta_value_score: new value_score divided by the mean value_score of vehicles in the same meta_cluster.
    meta_value_avg_result = db.session.query(func.avg(Car.value_score)).filter(Car.meta_cluster == meta_cluster).scalar()
    count_meta = db.session.query(func.count(Car.id)).filter(Car.meta_cluster == meta_cluster).scalar() or 0
    if count_meta == 0 or meta_value_avg_result is None:
        meta_value_score = value_score
    else:
        meta_value_avg = (meta_value_avg_result * count_meta + value_score) / (count_meta + 1)
        meta_value_score = value_score / meta_value_avg if meta_value_avg != 0 else None

    # Create a new Car record with base fields, predicted fields, and computed fields.
    car = Car(
        Make=updated_car['Make'],
        Model=updated_car['Model'],
        Year=updated_car['Year'],
        Engine_Fuel_Type=updated_car['Engine Fuel Type'],
        Engine_HP=updated_car['Engine HP'],
        Engine_Cylinders=updated_car['Engine Cylinders'],
        Transmission_Type=updated_car['Transmission Type'],
        Driven_Wheels=updated_car['Driven_Wheels'],
        Number_of_Doors=updated_car['Number of Doors'],
        Market_Category=updated_car['Market Category'],
        Vehicle_Size=updated_car['Vehicle Size'],
        Vehicle_Style=updated_car['Vehicle Style'],
        highway_MPG=updated_car['highway MPG'],
        city_mpg=updated_car['city mpg'],
        Popularity=updated_car['Popularity'],
        MSRP=updated_car['MSRP'],
        # Fields predicted by the ML model.
        cluster=updated_car['cluster'],
        P1=updated_car['P1'],
        P2=updated_car['P2'],
        P3=updated_car['P3'],
        meta_cluster=updated_car['meta_cluster'],
        # Computed fields.
        cluster_avg=cluster_avg,
        value_score=value_score,
        meta_value_score=meta_value_score
    )
    db.session.add(car)
    db.session.commit()
    return jsonify(car.to_dict()), 201

def print_db_stats():
    """Prints full stats of the current database:
       - Total number of rows.
       - Any rows with missing or NaN values.
    """
    with app.app_context():
        all_cars = Car.query.all()
        total_rows = len(all_cars)
        print(f"Total number of rows in the database: {total_rows}")

        missing_rows = []
        for car in all_cars:
            car_dict = car.to_dict()
            # Check for missing values (None or NaN).
            if any(pd.isna(value) or value is None for value in car_dict.values()):
                missing_rows.append(car_dict)
        if missing_rows:
            print("Rows with missing or NaN values:")
            for row in missing_rows:
                print(row)
        else:
            print("No rows with missing or NaN values.")

if __name__ == '__main__':
    with app.app_context():
        if Car.query.first() is None:
            print("Database is empty, initializing from CSV...")
            load_csv_to_db('final_vehicle_data.csv')
        else:
            print('Using existing persistent database')
    print_db_stats()
    app.run(debug=True, host='127.0.0.1')
