# Vehicle Value Analysis System

A full-stack web application that analyzes vehicle value based on mechanical components using machine learning clustering and dimensionality reduction techniques.

## Features

- **Interactive 3D/2D Visualization**: 
  - Explore vehicle clusters in 3D space using PCA-reduced dimensions
  - Switch between 3D and 2D views
  - Customize axis selection for both views
  - Color points by different attributes (meta_cluster, Make, Vehicle Size, Engine Fuel Type)
- **Real-time Predictions**: Add new vehicles and get instant cluster assignments and value scores
- **Advanced Filtering**: Filter vehicles by various attributes (make, model, year, etc.)
- **Value Scoring**: Compare vehicle values within clusters and meta-clusters

## Technical Stack

- **Frontend**: Dash (Python), Plotly, Dash Bootstrap Components
- **Backend**: Flask, SQLAlchemy
- **Machine Learning**: scikit-learn, pandas
- **Database**: SQLite (with SQLAlchemy ORM)
- **Security**: Flask-Limiter, Werkzeug Security

## Project Structure

```
CarValueGuide/
├── models/                      # ML model files
│   ├── vehicle_cluster_model.joblib
│   ├── vehicle_meta_cluster_model.joblib
│   ├── vehicle_p[1-3]_model.joblib
│   └── [cluster|meta_cluster|p1|p2|p3]_predict_model.py
├── front_end.py                 # Dash frontend application
├── server.py                    # Flask backend server
├── model_manager.py            # ML model management and predictions
└── final_vehicle_data.csv      # Initial dataset
```

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Initialize Database**:
   - Run `cluster.py` on `cars.csv` which will build the
   `final_vehicle_data.csv` 
   - The server will automatically initialize the database from `final_vehicle_data.csv` on first run
   - No manual database setup required

3. **Run the Application**:
   ```bash
   # Terminal 1: Start the Flask server
   python server.py

   # Terminal 2: Start the Dash frontend
   python front_end.py
   ```

4. **Access the Application**:
   - Frontend: http://localhost:8050
   - Backend API: http://localhost:5000

## ML Pipeline

1. **Data Processing**:
   - Initial clustering of vehicles based on mechanical features
   (strip away Make, Model, and MSRP)
   Silhouette Score (k=1200): 0.5484
   - PCA reduction to 3 dimensions (P1, P2, P3)
      - === Top features in PC1 ===
      - Engine Cylinders: -0.5026
      - highway MPG: 0.4986
      - city mpg: 0.4784
      - Engine HP: -0.4105
      - Driven_Wheels=front wheel drive: 0.1615
      - Number of Doors: 0.1239
      - Driven_Wheels=rear wheel drive: -0.1020
      - Vehicle Size=Large: -0.0961
      - Vehicle Size=Compact: 0.0956
      - Engine Fuel Type=premium unleaded (required): -0.0747
      - === Top features in PC2 ===
      - Year: 0.6608
      - Engine HP: 0.3968
      - Number of Doors: 0.3919
      - highway MPG: 0.2016
      - city mpg: 0.1804
      - Transmission Type=MANUAL: -0.1751
      - Engine Fuel Type=regular unleaded: -0.1575
      - Transmission Type=AUTOMATIC: 0.1483
      - Market Category=N/A: -0.1448
      - Vehicle Size=Compact: -0.1364
      - === Top features in PC3 ===
      - Number of Doors: -0.6864
      - Engine HP: 0.2749
      - city mpg: 0.2600
      - highway MPG: 0.2566
      - Popularity: 0.2492
      - Transmission Type=AUTOMATIC: -0.1917
      - Engine Fuel Type=regular unleaded: -0.1789
      - Year: 0.1498
      - Vehicle Size=Compact: 0.1418
      - Vehicle Style=Coupe: 0.1361
   - Meta-clustering for higher-level vehicle categorization
   Silhouette Score *Meta-clusters* (k=20): 0.2714

2. **Model Architecture**:
   - Primary clustering model
   - Meta-clustering model
   - Three PCA component models

3. **Value Scoring**:
   - Cluster-based value scoring
   - Meta-cluster value comparison
   - Real-time score updates for new vehicles

## Visualization Features

1. **Graph Types**:
   - 3D view: Default view using PCA-reduced dimensions (P1, P2, P3)
   - 2D view: Customizable view using any numeric feature or PCA dimension

2. **Customization Options**:
   - Switch between 2D and 3D visualizations
   - Select features for x, y, and z axes (in 3D mode)
   - Select features for x and y axes (in 2D mode)
   - Color points by different attributes:
     - Meta-cluster (default)
     - Make
     - Vehicle Size
     - Engine Fuel Type

3. **Interactive Features**:
   - Hover over points to see vehicle details
   - Click on points to view detailed vehicle information
   - Apply filters to focus on specific vehicle attributes

## Current Status

- ✅ ML pipeline implementation complete
- ✅ Frontend visualization and interaction
- ✅ Graph customization features
- ✅ Backend API and database
- ✅ Security measures implemented
- ✅ Real-time predictions working

## Future Improvements

- [ ] Add comprehensive test suite
- [ ] Implement model caching
- [ ] Add user authentication
- [ ] Deploy to cloud platform
- [ ] Add API documentation
- [ ] Implement CI/CD pipeline

## Notes

- Database is automatically initialized on first run
- Security measures are in place for production use


