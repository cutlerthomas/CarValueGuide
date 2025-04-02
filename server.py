from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
from model_manager import predict_vehicle_features
from flask_cors import CORS
import os
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging
from typing import Dict, Any
from werkzeug.security import generate_password_hash, check_password_hash
import re
from datetime import datetime, timedelta
import secrets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Security configuration
class SecurityConfig:
    # CORS settings
    CORS_ORIGINS = ['http://localhost:8050']  # Add your frontend URL
    CORS_METHODS = ['GET', 'POST']
    CORS_HEADERS = ['Content-Type', 'Authorization']
    
    # Rate limiting
    RATE_LIMIT = os.getenv('RATE_LIMIT', '100 per minute')
    RATE_LIMIT_POST = os.getenv('RATE_LIMIT_POST', '10 per minute')
    
    # Session settings
    SESSION_LIFETIME = timedelta(hours=1)
    
    # Input validation
    MAX_STRING_LENGTH = 80
    MAX_NUMERIC_VALUE = 1000000
    ALLOWED_MAKES = set()  # Will be populated from data
    ALLOWED_MODELS = set()  # Will be populated from data
    ALLOWED_FUEL_TYPES = {'regular unleaded', 'premium unleaded', 'diesel', 'electric'}
    ALLOWED_TRANSMISSION_TYPES = {'AUTOMATIC', 'MANUAL'}
    ALLOWED_DRIVEN_WHEELS = {'front wheel drive', 'rear wheel drive', 'all wheel drive', 'four wheel drive'}
    ALLOWED_VEHICLE_SIZES = {'Compact', 'Midsize', 'Large'}
    ALLOWED_VEHICLE_STYLES = {'Sedan', 'SUV', 'Truck', 'Van', 'Wagon', 'Coupe', 'Convertible'}

# Configuration
class Config:
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///cars.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    RATE_LIMIT = SecurityConfig.RATE_LIMIT
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    SECRET_KEY = os.getenv('SECRET_KEY', secrets.token_hex(32))  # Generate a secure secret key
    SESSION_COOKIE_SECURE = True  # Only send cookies over HTTPS
    SESSION_COOKIE_HTTPONLY = True  # Prevent JavaScript access to session cookie
    SESSION_COOKIE_SAMESITE = 'Lax'  # Protect against CSRF
    PERMANENT_SESSION_LIFETIME = SecurityConfig.SESSION_LIFETIME

app.config.from_object(Config)

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Configure CORS with security settings
CORS(app, 
     origins=SecurityConfig.CORS_ORIGINS,
     methods=SecurityConfig.CORS_METHODS,
     allow_headers=SecurityConfig.CORS_HEADERS,
     supports_credentials=True)

# Add security headers middleware
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';"
    return response

# Add rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[app.config['RATE_LIMIT']]
)

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
        try:
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
        except Exception as e:
            logger.error(f"Error converting car to dict: {str(e)}")
            raise

def sanitize_input(value: str) -> str:
    """Sanitize string input to prevent XSS and injection attacks."""
    if not isinstance(value, str):
        return str(value)
    # Remove potentially dangerous characters
    value = re.sub(r'[<>]', '', value)
    # Truncate to max length
    return value[:SecurityConfig.MAX_STRING_LENGTH]

def validate_numeric_input(value: float, field_name: str) -> tuple[bool, str]:
    """Validate numeric input values."""
    if not isinstance(value, (int, float)):
        return False, f"{field_name} must be a number"
    if value <= 0:
        return False, f"{field_name} must be positive"
    if value > SecurityConfig.MAX_NUMERIC_VALUE:
        return False, f"{field_name} exceeds maximum allowed value"
    return True, ""

def validate_car_data(data: Dict[str, Any]) -> tuple[bool, str]:
    """Validate car data before processing with enhanced security checks."""
    required_fields = [
        'Make', 'Model', 'Year', 'Engine Fuel Type', 'Engine HP', 'Engine Cylinders',
        'Transmission Type', 'Driven_Wheels', 'Number of Doors', 'Market Category',
        'Vehicle Size', 'Vehicle Style', 'highway MPG', 'city mpg', 'Popularity', 'MSRP'
    ]
    
    # Check for missing required fields
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    # Sanitize string inputs
    for field in ['Make', 'Model', 'Engine Fuel Type', 'Transmission Type', 
                 'Driven_Wheels', 'Market Category', 'Vehicle Size', 'Vehicle Style']:
        data[field] = sanitize_input(data[field])
    
    # Validate categorical fields
    if data['Engine Fuel Type'] not in SecurityConfig.ALLOWED_FUEL_TYPES:
        return False, "Invalid engine fuel type"
    if data['Transmission Type'] not in SecurityConfig.ALLOWED_TRANSMISSION_TYPES:
        return False, "Invalid transmission type"
    if data['Driven_Wheels'] not in SecurityConfig.ALLOWED_DRIVEN_WHEELS:
        return False, "Invalid driven wheels type"
    if data['Vehicle Size'] not in SecurityConfig.ALLOWED_VEHICLE_SIZES:
        return False, "Invalid vehicle size"
    if data['Vehicle Style'] not in SecurityConfig.ALLOWED_VEHICLE_STYLES:
        return False, "Invalid vehicle style"
    
    # Validate numeric fields
    numeric_fields = ['Year', 'Engine HP', 'Engine Cylinders', 'Number of Doors', 
                     'highway MPG', 'city mpg', 'Popularity', 'MSRP']
    for field in numeric_fields:
        is_valid, error_msg = validate_numeric_input(float(data[field]), field)
        if not is_valid:
            return False, error_msg
    
    # Validate ranges
    if not (1900 <= float(data['Year']) <= datetime.now().year):
        return False, f"Year must be between 1900 and {datetime.now().year}"
    
    return True, ""

@app.route('/cars', methods=['GET'])
@limiter.limit(SecurityConfig.RATE_LIMIT)
def get_cars():
    """Return the entire database as JSON with security headers."""
    try:
        cars = Car.query.all()
        return jsonify([car.to_dict() for car in cars])
    except SQLAlchemyError as e:
        logger.error(f"Database error while fetching cars: {str(e)}")
        return jsonify({'error': 'Database error occurred'}), 500
    except Exception as e:
        logger.error(f"Unexpected error while fetching cars: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/cars', methods=['POST'])
@limiter.limit(SecurityConfig.RATE_LIMIT_POST)
def add_car():
    """Add a new car entry to the database with enhanced security."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Validate input data
        is_valid, error_message = validate_car_data(data)
        if not is_valid:
            return jsonify({'error': error_message}), 400

        # Convert and sanitize data to proper types
        new_car = {
            'Make': sanitize_input(str(data['Make'])),
            'Model': sanitize_input(str(data['Model'])),
            'Year': float(data['Year']),
            'Engine Fuel Type': sanitize_input(str(data['Engine Fuel Type'])),
            'Engine HP': float(data['Engine HP']),
            'Engine Cylinders': float(data['Engine Cylinders']),
            'Transmission Type': sanitize_input(str(data['Transmission Type'])),
            'Driven_Wheels': sanitize_input(str(data['Driven_Wheels'])),
            'Number of Doors': float(data['Number of Doors']),
            'Market Category': sanitize_input(str(data['Market Category'])),
            'Vehicle Size': sanitize_input(str(data['Vehicle Size'])),
            'Vehicle Style': sanitize_input(str(data['Vehicle Style'])),
            'highway MPG': float(data['highway MPG']),
            'city mpg': float(data['city mpg']),
            'Popularity': float(data['Popularity']),
            'MSRP': float(data['MSRP'])
        }

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
        
        # Compute Additional Fields
        try:
            cluster_avg_result = db.session.query(func.avg(Car.MSRP)).filter(Car.cluster == cluster).scalar()
            count_cluster = db.session.query(func.count(Car.id)).filter(Car.cluster == cluster).scalar() or 0
            if count_cluster == 0 or cluster_avg_result is None:
                cluster_avg = msrp
            else:
                cluster_avg = (cluster_avg_result * count_cluster + msrp) / (count_cluster + 1)

            value_score = msrp / cluster_avg if cluster_avg != 0 else None

            meta_value_avg_result = db.session.query(func.avg(Car.value_score)).filter(Car.meta_cluster == meta_cluster).scalar()
            count_meta = db.session.query(func.count(Car.id)).filter(Car.meta_cluster == meta_cluster).scalar() or 0
            if count_meta == 0 or meta_value_avg_result is None:
                meta_value_score = value_score
            else:
                meta_value_avg = (meta_value_avg_result * count_meta + value_score) / (count_meta + 1)
                meta_value_score = value_score / meta_value_avg if meta_value_avg != 0 else None
        except SQLAlchemyError as e:
            logger.error(f"Database error while computing scores: {str(e)}")
            return jsonify({'error': 'Database error occurred while computing scores'}), 500

        # Create a new Car record
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
            cluster=updated_car['cluster'],
            P1=updated_car['P1'],
            P2=updated_car['P2'],
            P3=updated_car['P3'],
            meta_cluster=updated_car['meta_cluster'],
            cluster_avg=cluster_avg,
            value_score=value_score,
            meta_value_score=meta_value_score
        )

        try:
            db.session.add(car)
            db.session.commit()
            return jsonify(car.to_dict()), 201
        except SQLAlchemyError as e:
            db.session.rollback()
            logger.error(f"Database error while adding car: {str(e)}")
            return jsonify({'error': 'Database error occurred while adding car'}), 500

    except Exception as e:
        logger.error(f"Unexpected error while adding car: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

def load_csv_to_db(csv_file_path):
    """Load data from CSV to database with enhanced security."""
    try:
        df = pd.read_csv(csv_file_path)
        # Populate allowed values from data
        SecurityConfig.ALLOWED_MAKES = set(df['Make'].unique())
        SecurityConfig.ALLOWED_MODELS = set(df['Model'].unique())
        
        for index, row in df.iterrows():
            try:
                # Sanitize all string inputs
                car = Car(
                    Make=sanitize_input(str(row['Make'])),
                    Model=sanitize_input(str(row['Model'])),
                    Year=float(row['Year']),
                    Engine_Fuel_Type=sanitize_input(str(row['Engine Fuel Type'])),
                    Engine_HP=float(row['Engine HP']),
                    Engine_Cylinders=float(row['Engine Cylinders']),
                    Transmission_Type=sanitize_input(str(row['Transmission Type'])),
                    Driven_Wheels=sanitize_input(str(row['Driven_Wheels'])),
                    Number_of_Doors=float(row['Number of Doors']),
                    Market_Category=sanitize_input(str(row['Market Category'])),
                    Vehicle_Size=sanitize_input(str(row['Vehicle Size'])),
                    Vehicle_Style=sanitize_input(str(row['Vehicle Style'])),
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
            except Exception as e:
                logger.error(f"Error processing row {index}: {str(e)}")
                continue
        db.session.commit()
    except Exception as e:
        logger.error(f"Error loading CSV to database: {str(e)}")
        db.session.rollback()
        raise

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
        try:
            if Car.query.first() is None:
                print("Database is empty, initializing from CSV...")
                load_csv_to_db('final_vehicle_data.csv')
            else:
                print('Using existing persistent database')
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            print(f"Error initializing database: {str(e)}")
    print_db_stats()
    app.run(debug=True, host='127.0.0.1')
