import pandas as pd
from joblib import load
import logging
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.models = {}
        self.model_paths = {
            'cluster': 'models/vehicle_cluster_model.joblib',
            'p1': 'models/vehicle_p1_model.joblib',
            'p2': 'models/vehicle_p2_model.joblib',
            'p3': 'models/vehicle_p3_model.joblib',
            'meta_cluster': 'models/vehicle_meta_cluster_model.joblib'
        }
        self._load_models()

    def _load_models(self):
        try:
            # Load cluster model first as it's needed for initial prediction
            self.models['cluster'] = load(self.model_paths['cluster'])
            logger.info("Successfully loaded cluster model")
            # Force garbage collection after loading
            gc.collect()
        except Exception as e:
            logger.error(f"Error loading cluster model: {str(e)}")
            raise

    def _ensure_model_loaded(self, model_name: str):
        if model_name not in self.models:
            try:
                # Force garbage collection before loading new model
                gc.collect()
                self.models[model_name] = load(self.model_paths[model_name])
                logger.info(f"Successfully loaded {model_name} model")
            except Exception as e:
                logger.error(f"Error loading {model_name} model: {str(e)}")
                raise

    def predict_vehicle_features(self, vehicle_df):
        try:
            results_df = vehicle_df.copy()
            
            # Define features for cluster prediction
            cluster_features = ['Year', 'Engine Fuel Type', 'Engine HP', 'Engine Cylinders', 
                              'Transmission Type', 'Driven_Wheels', 'Number of Doors', 
                              'Market Category', 'Vehicle Size', 'Vehicle Style', 
                              'highway MPG', 'city mpg', 'Popularity']
            
            # Predict cluster
            predicted_cluster = self.models['cluster'].predict(vehicle_df[cluster_features])
            results_df['cluster'] = predicted_cluster[0] if len(predicted_cluster) == 1 else predicted_cluster
            
            # Load and use P1 model
            self._ensure_model_loaded('p1')
            vehicle_df_p1 = vehicle_df.copy()
            vehicle_df_p1['cluster'] = predicted_cluster
            p1_pred = self.models['p1'].predict(vehicle_df_p1)
            results_df['P1'] = p1_pred[0] if len(p1_pred) == 1 else p1_pred
            del vehicle_df_p1
            gc.collect()
            
            # Load and use P2 model
            self._ensure_model_loaded('p2')
            vehicle_df_p2 = vehicle_df.copy()
            vehicle_df_p2['cluster'] = predicted_cluster
            vehicle_df_p2['P1'] = p1_pred
            p2_pred = self.models['p2'].predict(vehicle_df_p2)
            results_df['P2'] = p2_pred[0] if len(p2_pred) == 1 else p2_pred
            del vehicle_df_p2
            gc.collect()
            
            # Load and use P3 model
            self._ensure_model_loaded('p3')
            vehicle_df_p3 = vehicle_df.copy()
            vehicle_df_p3['cluster'] = predicted_cluster
            vehicle_df_p3['P1'] = p1_pred
            vehicle_df_p3['P2'] = p2_pred
            p3_pred = self.models['p3'].predict(vehicle_df_p3)
            results_df['P3'] = p3_pred[0] if len(p3_pred) == 1 else p3_pred
            del vehicle_df_p3
            gc.collect()
            
            # Load and use meta_cluster model
            self._ensure_model_loaded('meta_cluster')
            vehicle_df_meta = vehicle_df.copy()
            vehicle_df_meta['cluster'] = predicted_cluster
            vehicle_df_meta['P1'] = p1_pred
            vehicle_df_meta['P2'] = p2_pred
            vehicle_df_meta['P3'] = p3_pred
            meta_cluster_pred = self.models['meta_cluster'].predict(vehicle_df_meta)
            results_df['meta_cluster'] = meta_cluster_pred[0] if len(meta_cluster_pred) == 1 else meta_cluster_pred
            del vehicle_df_meta
            gc.collect()
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

# Create a singleton instance
model_manager = ModelManager()

# Export the predict_vehicle_features function
def predict_vehicle_features(vehicle_df):
    return model_manager.predict_vehicle_features(vehicle_df)

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
    print(new_df)
'''