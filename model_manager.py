# model_manager.py

import pandas as pd
from joblib import load

# Load the saved models
cluster_model = load("models/vehicle_cluster_model.joblib")
p1_model = load("models/vehicle_p1_model.joblib")
p2_model = load("models/vehicle_p2_model.joblib")
p3_model = load("models/vehicle_p3_model.joblib")
meta_cluster_model = load("models/vehicle_meta_cluster_model.joblib")

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
    results_df = vehicle_df.copy()

    # Define the features used for the cluster prediction
    cluster_features = ['Year', 'Engine Fuel Type', 'Engine HP', 'Engine Cylinders', 'Transmission Type', 
                        'Driven_Wheels', 'Number of Doors', 'Market Category', 
                        'Vehicle Size', 'Vehicle Style', 'highway MPG', 'city mpg', 'Popularity']
    
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