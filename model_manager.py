# model_manager.py

import pandas as pd
from joblib import load
import os
import psutil
import logging
from typing import Optional
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_memory_available(required_memory_gb: float) -> bool:
    """Check if enough memory is available for model loading."""
    available_memory = psutil.virtual_memory().available / (1024**3)  # Convert to GB
    return available_memory >= required_memory_gb

def safe_load_model(model_path: str, required_memory_gb: float) -> Optional[object]:
    """Safely load a model with memory checks."""
    try:
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None
            
        if not check_memory_available(required_memory_gb):
            logger.error(f"Insufficient memory to load model: {model_path}")
            return None
            
        # Load the model
        model = load(model_path)
        
        # If it's a pipeline with ColumnTransformer, fix the version mismatch
        if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
            preprocessor = model.named_steps['preprocessor']
            if isinstance(preprocessor, ColumnTransformer):
                # Recreate the ColumnTransformer with the same configuration
                transformers = preprocessor.transformers
                new_preprocessor = ColumnTransformer(
                    transformers=transformers,
                    remainder=preprocessor.remainder,
                    sparse_threshold=preprocessor.sparse_threshold,
                    n_jobs=preprocessor.n_jobs
                )
                # Replace the old preprocessor with the new one
                model.named_steps['preprocessor'] = new_preprocessor
        
        return model
    except Exception as e:
        logger.error(f"Error loading model {model_path}: {str(e)}")
        return None

# Load the saved models with memory requirements
MODEL_MEMORY_REQUIREMENTS = {
    "vehicle_cluster_model.joblib": 5.0,  # GB
    "vehicle_meta_cluster_model.joblib": 0.1,
    "vehicle_p1_model.joblib": 0.1,
    "vehicle_p2_model.joblib": 0.1,
    "vehicle_p3_model.joblib": 0.1
}

# Load models with safety checks
cluster_model = safe_load_model("models/vehicle_cluster_model.joblib", 
                              MODEL_MEMORY_REQUIREMENTS["vehicle_cluster_model.joblib"])
p1_model = safe_load_model("models/vehicle_p1_model.joblib", 
                          MODEL_MEMORY_REQUIREMENTS["vehicle_p1_model.joblib"])
p2_model = safe_load_model("models/vehicle_p2_model.joblib", 
                          MODEL_MEMORY_REQUIREMENTS["vehicle_p2_model.joblib"])
p3_model = safe_load_model("models/vehicle_p3_model.joblib", 
                          MODEL_MEMORY_REQUIREMENTS["vehicle_p3_model.joblib"])
meta_cluster_model = safe_load_model("models/vehicle_meta_cluster_model.joblib", 
                                   MODEL_MEMORY_REQUIREMENTS["vehicle_meta_cluster_model.joblib"])

def predict_vehicle_features(vehicle_df):
    """
    Given a DataFrame with raw vehicle features, this function predicts:
      - cluster (using the cluster_model)
      - P1, P2, P3 (using their respective regression models)
      - meta_cluster (using the meta_cluster_model)
    It also calculates derived values such as:
      - cluster_avg: average MSRP of cluster
      - value_score: actualMSRP/clusterAvgMSRP
      - meta_value_score: an example combination of meta_cluster and value_score

    Parameters:
        vehicle_df (pd.DataFrame): DataFrame with raw vehicle data.
          Expected columns include:
            'Year', 'Engine Fuel Type', 'Engine HP', 'Engine Cylinders', 'Transmission Type',
            'Driven_Wheels', 'Number of Doors', 'Market Category', 'Vehicle Size',
            'Vehicle Style', 'highway MPG', 'city mpg', 'Popularity'
          (Note: Make, Model, MSRP are excluded based on your earlier decision.)

    Returns:
        dict: Predicted values and calculated metrics.
    """
    if not all([cluster_model, p1_model, p2_model, p3_model, meta_cluster_model]):
        logger.error("One or more models failed to load")
        raise RuntimeError("Required models are not available")

    results_df = vehicle_df.copy()

    # Define the features used for the cluster prediction
    cluster_features = ['Year', 'Engine Fuel Type', 'Engine HP', 'Engine Cylinders', 'Transmission Type', 
                        'Driven_Wheels', 'Number of Doors', 'Market Category', 
                        'Vehicle Size', 'Vehicle Style', 'highway MPG', 'city mpg', 'Popularity']
    
    try:
        # Predict the cluster
        predicted_cluster = cluster_model.predict(vehicle_df[cluster_features])
        
        # Add the predicted cluster into the DataFrame for the subsequent models
        vehicle_df_p1 = vehicle_df.copy()
        vehicle_df_p1['cluster'] = predicted_cluster
        p1_pred = p1_model.predict(vehicle_df_p1)

        vehicle_df_p2 = vehicle_df_p1.copy()
        vehicle_df_p2['P1'] = p1_pred
        p2_pred = p2_model.predict(vehicle_df_p2)

        vehicle_df_p3 = vehicle_df_p2.copy()
        vehicle_df_p3['P2'] = p2_pred
        p3_pred = p3_model.predict(vehicle_df_p3)

        vehicle_df_meta = vehicle_df_p3.copy()
        vehicle_df_meta['P3'] = p3_pred
        meta_cluster_pred = meta_cluster_model.predict(vehicle_df_meta)

        # Add results to complete DataFrame
        results_df['cluster'] = predicted_cluster[0] if len(predicted_cluster) == 1 else predicted_cluster
        results_df['P1'] = p1_pred[0] if len(p1_pred) == 1 else p1_pred
        results_df['P2'] = p2_pred[0] if len(p2_pred) == 1 else p2_pred
        results_df['P3'] = p3_pred[0] if len(p3_pred) == 1 else p3_pred
        results_df['meta_cluster'] = meta_cluster_pred[0] if len(meta_cluster_pred) == 1 else meta_cluster_pred
        
        return results_df
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

# For testing the integration pipeline
'''
if __name__ == "__main__":
    # Create a sample input DataFrame for a single vehicle
    sample_data = {
        'Make': ['Toyota'],               # Not used in predictions
        'Model': ['Camry'],               # Not used in predictions
        'Year': [2020],
        'Engine Fuel Type': ['regular unleaded'],
        'Engine HP': [203],
        'Engine Cylinders': [4],
        'Transmission Type': ['AUTOMATIC'],
        'Driven_Wheels': ['front wheel drive'],
        'Number of Doors': [4],
        'Market Category': ['Hybrid'],
        'Vehicle Size': ['Midsize'],
        'Vehicle Style': ['Sedan'],
        'highway MPG': [34],
        'city mpg': [28],
        'Popularity': [100],
        'MSRP': [25000]                   # Not used in predictions
    }
    
    sample_df = pd.DataFrame(sample_data)
    new_df = predict_vehicle_features(sample_df)
    
    print("Predictions for sample vehicle:")
    print(new_df)'
'''