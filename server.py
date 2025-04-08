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

# Configure application logging for monitoring and debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Flask application
app = Flask(__name__)

# Security configuration class defining application security parameters
class SecurityConfig:
    # CORS configuration for cross-origin resource sharing
    CORS_ORIGINS = ['http://localhost:8050']  # Frontend URL
    CORS_METHODS = ['GET', 'POST']
    CORS_HEADERS = ['Content-Type', 'Authorization']
    
    # Rate limiting configuration to prevent abuse
    RATE_LIMIT = os.getenv('RATE_LIMIT', '100 per minute')  # Default rate limit
    RATE_LIMIT_POST = os.getenv('RATE_LIMIT_POST', '10 per minute')  # Stricter limit for POST requests
    
    # Session configuration for user authentication
    SESSION_LIFETIME = timedelta(hours=1)
    
    # Input validation parameters
    MAX_STRING_LENGTH = 80
    MAX_NUMERIC_VALUE = 1000000
    ALLOWED_MAKES = set()  # Populated dynamically from database
    ALLOWED_MODELS = set()  # Populated dynamically from database
    ALLOWED_FUEL_TYPES = {
        'diesel',
        'electric',
        'flex-fuel (premium unleaded recommended/E85)',
        'flex-fuel (premium unleaded required/E85)',
        'flex-fuel (unleaded/E85)',
        'flex-fuel (unleaded/natural gas)',
        'natural gas',
        'premium unleaded (recommended)',
        'premium unleaded (required)',
        'regular unleaded'
    }
    ALLOWED_TRANSMISSION_TYPES = {
        'AUTOMATED_MANUAL',
        'AUTOMATIC',
        'DIRECT_DRIVE',
        'MANUAL',
        'UNKNOWN'
    }
    ALLOWED_DRIVEN_WHEELS = {
        'all wheel drive',
        'four wheel drive',
        'front wheel drive',
        'rear wheel drive'
    }
    ALLOWED_VEHICLE_SIZES = {
        'Compact',
        'Large',
        'Midsize'
    }
    ALLOWED_VEHICLE_STYLES = {
        '2dr Hatchback',
        '2dr SUV',
        '4dr Hatchback',
        '4dr SUV',
        'Cargo Minivan',
        'Cargo Van',
        'Convertible',
        'Convertible SUV',
        'Coupe',
        'Crew Cab Pickup',
        'Extended Cab Pickup',
        'Passenger Minivan',
        'Passenger Van',
        'Regular Cab Pickup',
        'Sedan',
        'Wagon'
    }
    # Market Category validation handled separately due to large number of possible values

# Application configuration class defining core settings
class Config:
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///cars.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    RATE_LIMIT = SecurityConfig.RATE_LIMIT
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB maximum file size
    SECRET_KEY = os.getenv('SECRET_KEY', secrets.token_hex(32))  # Secure random key generation
    SESSION_COOKIE_SECURE = True  # Enforce HTTPS for cookies
    SESSION_COOKIE_HTTPONLY = True  # Prevent JavaScript access to cookies
    SESSION_COOKIE_SAMESITE = 'Lax'  # CSRF protection
    PERMANENT_SESSION_LIFETIME = SecurityConfig.SESSION_LIFETIME

# Apply configuration to the Flask application
app.config.from_object(Config)

# Initialize SQLAlchemy ORM for database operations
db = SQLAlchemy(app)

# Configure CORS with security settings to control cross-origin requests
CORS(app, 
     origins=SecurityConfig.CORS_ORIGINS,
     methods=SecurityConfig.CORS_METHODS,
     allow_headers=SecurityConfig.CORS_HEADERS,
     supports_credentials=True,
     expose_headers=['Content-Type', 'X-Content-Type-Options', 'X-Frame-Options', 
                    'X-XSS-Protection', 'Strict-Transport-Security', 'Content-Security-Policy'])

# Middleware to add CORS headers to all responses
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Methods'] = ', '.join(SecurityConfig.CORS_METHODS)
    response.headers['Access-Control-Allow-Headers'] = ', '.join(SecurityConfig.CORS_HEADERS)
    return response

# Middleware to add security headers to all responses
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'  # Allow embedding in same origin
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';"
    return response

# Configure rate limiting to prevent API abuse
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[app.config['RATE_LIMIT']]
)

# Database model representing vehicle data
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
    # Machine learning model outputs
    cluster = db.Column(db.Float)
    P1 = db.Column(db.Float)
    P2 = db.Column(db.Float)
    P3 = db.Column(db.Float)
    meta_cluster = db.Column(db.Float)
    # Computed metrics for value analysis
    cluster_avg = db.Column(db.Float)
    value_score = db.Column(db.Float)
    meta_value_score = db.Column(db.Float)

    # Convert database record to dictionary for API responses
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

# Input sanitization function to prevent XSS and injection attacks
def sanitize_input(value: str) -> str:
    """Sanitize string input to prevent XSS and injection attacks."""
    if not isinstance(value, str):
        return str(value)
    # Remove potentially dangerous characters
    value = re.sub(r'[<>]', '', value)
    # Truncate to maximum allowed length
    return value[:SecurityConfig.MAX_STRING_LENGTH]

# Numeric input validation function
def validate_numeric_input(value: float, field_name: str) -> tuple[bool, str]:
    """Validate numeric input values against defined constraints."""
    if not isinstance(value, (int, float)):
        return False, f"{field_name} must be a number"
    if value < 0:  # Allow 0 for electric vehicles with no cylinders
        return False, f"{field_name} must be non-negative"
    if value > SecurityConfig.MAX_NUMERIC_VALUE:
        return False, f"{field_name} exceeds maximum allowed value"
    return True, ""

# Comprehensive vehicle data validation function
def validate_car_data(data: Dict[str, Any]) -> tuple[bool, str]:
    """Validate vehicle data against defined constraints and security requirements."""
    required_fields = [
        'Make', 'Model', 'Year', 'Engine Fuel Type', 'Engine HP', 'Engine Cylinders',
        'Transmission Type', 'Driven_Wheels', 'Number of Doors', 'Market Category',
        'Vehicle Size', 'Vehicle Style', 'highway MPG', 'city mpg', 'Popularity', 'MSRP'
    ]
    
    # Verify all required fields are present
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    # Sanitize all string inputs
    for field in ['Make', 'Model', 'Engine Fuel Type', 'Transmission Type', 
                 'Driven_Wheels', 'Market Category', 'Vehicle Size', 'Vehicle Style']:
        data[field] = sanitize_input(data[field])
    
    # Validate categorical fields against allowed values
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
    
    # Validate year is within reasonable range
    if not (1900 <= float(data['Year']) <= datetime.now().year):
        return False, f"Year must be between 1900 and {datetime.now().year}"
    
    return True, ""

# API endpoint to retrieve all vehicles
@app.route('/cars', methods=['GET'])
@limiter.limit(SecurityConfig.RATE_LIMIT)
def get_cars():
    """Retrieve all vehicles from the database with security headers."""
    try:
        cars = Car.query.all()
        return jsonify([car.to_dict() for car in cars])
    except SQLAlchemyError as e:
        logger.error(f"Database error while fetching cars: {str(e)}")
        return jsonify({'error': 'Database error occurred'}), 500
    except Exception as e:
        logger.error(f"Unexpected error while fetching cars: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

# API endpoint to add a new vehicle
@app.route('/cars', methods=['POST'])
@limiter.limit(SecurityConfig.RATE_LIMIT_POST)
def add_car():
    """Process and store a new vehicle with validation and security measures."""
    try:
        logger.info("Received POST request to /cars endpoint")
        data = request.get_json()
        if not data:
            logger.error("No JSON data provided in request")
            return jsonify({'error': 'No JSON data provided'}), 400

        logger.info(f"Received car data: {data}")
        
        # Validate input data against security requirements
        is_valid, error_message = validate_car_data(data)
        if not is_valid:
            logger.error(f"Invalid car data: {error_message}")
            return jsonify({'error': error_message}), 400

        logger.info("Car data validation successful")

        # Convert and sanitize data to appropriate types
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

        logger.info("Data sanitization complete")

        # Apply machine learning models to generate additional features
        logger.info("Starting prediction with ML models")
        new_car_df = pd.DataFrame([new_car])
        new_car_df = predict_vehicle_features(new_car_df)
        updated_car = new_car_df.iloc[0].to_dict()
        logger.info(f"ML prediction complete. Results: {updated_car}")

        # Extract key values for value score calculation
        cluster = updated_car['cluster']
        meta_cluster = updated_car['meta_cluster']
        msrp = updated_car['MSRP']

        if cluster is None or meta_cluster is None or msrp is None:
            logger.error(f"Missing required prediction values: cluster={cluster}, meta_cluster={meta_cluster}, msrp={msrp}")
            return jsonify({'error': 'Missing cluster, meta_cluster, or MSRP from ML prediction'}), 400
        
        # Calculate value scores based on cluster and meta-cluster averages
        try:
            logger.info("Computing cluster average")
            cluster_avg_result = db.session.query(func.avg(Car.MSRP)).filter(Car.cluster == cluster).scalar()
            count_cluster = db.session.query(func.count(Car.id)).filter(Car.cluster == cluster).scalar() or 0
            if count_cluster == 0 or cluster_avg_result is None:
                cluster_avg = msrp
            else:
                cluster_avg = (cluster_avg_result * count_cluster + msrp) / (count_cluster + 1)
            logger.info(f"Cluster average computed: {cluster_avg}")

            value_score = msrp / cluster_avg if cluster_avg != 0 else None
            logger.info(f"Value score computed: {value_score}")

            logger.info("Computing meta cluster average")
            meta_value_avg_result = db.session.query(func.avg(Car.value_score)).filter(Car.meta_cluster == meta_cluster).scalar()
            count_meta = db.session.query(func.count(Car.id)).filter(Car.meta_cluster == meta_cluster).scalar() or 0
            if count_meta == 0 or meta_value_avg_result is None:
                meta_value_score = value_score
            else:
                meta_value_avg = (meta_value_avg_result * count_meta + value_score) / (count_meta + 1)
                meta_value_score = value_score / meta_value_avg if meta_value_avg != 0 else None
            logger.info(f"Meta value score computed: {meta_value_score}")
        except SQLAlchemyError as e:
            logger.error(f"Database error while computing scores: {str(e)}")
            return jsonify({'error': 'Database error occurred while computing scores'}), 500

        # Create and store the new vehicle record
        logger.info("Creating new Car record")
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
            cluster=float(updated_car['cluster']),
            P1=float(updated_car['P1']),
            P2=float(updated_car['P2']),
            P3=float(updated_car['P3']),
            meta_cluster=float(updated_car['meta_cluster']),
            cluster_avg=cluster_avg,
            value_score=value_score,
            meta_value_score=meta_value_score
        )

        try:
            logger.info("Adding car to database")
            db.session.add(car)
            db.session.commit()
            logger.info(f"Car added successfully with ID: {car.id}")
            return jsonify(car.to_dict()), 201
        except SQLAlchemyError as e:
            db.session.rollback()
            logger.error(f"Database error while adding car: {str(e)}")
            return jsonify({'error': 'Database error occurred while adding car'}), 500

    except Exception as e:
        logger.error(f"Unexpected error while adding car: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

# Function to initialize database from CSV file
def load_csv_to_db(csv_file_path):
    """Load vehicle data from CSV file into the database with security validation."""
    try:
        df = pd.read_csv(csv_file_path)
        # Populate allowed values from data for validation
        SecurityConfig.ALLOWED_MAKES = set(df['Make'].unique())
        SecurityConfig.ALLOWED_MODELS = set(df['Model'].unique())
        
        for index, row in df.iterrows():
            try:
                # Sanitize all string inputs before database insertion
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

# Function to display database statistics
def print_db_stats():
    """Display database statistics including row count and data quality metrics."""
    with app.app_context():
        all_cars = Car.query.all()
        total_rows = len(all_cars)
        print(f"Total number of rows in the database: {total_rows}")

        missing_rows = []
        for car in all_cars:
            car_dict = car.to_dict()
            # Identify rows with missing or invalid values
            if any(pd.isna(value) or value is None for value in car_dict.values()):
                missing_rows.append(car_dict)
        if missing_rows:
            print("Rows with missing or NaN values:")
            for row in missing_rows:
                print(row)
        else:
            print("No rows with missing or NaN values.")

# Application entry point
if __name__ == '__main__':
    with app.app_context():
        try:
            # Initialize database schema
            db.create_all()
            
            # Load initial data if database is empty
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
