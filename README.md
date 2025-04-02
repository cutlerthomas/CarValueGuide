# Vehicle Value Analysis System

A full-stack web application that analyzes vehicle value based on mechanical components using machine learning clustering and dimensionality reduction techniques.

## Features

- **Interactive 3D Visualization**: Explore vehicle clusters in 3D space using PCA-reduced dimensions
- **Real-time Predictions**: Add new vehicles and get instant cluster assignments and value scores
- **Advanced Filtering**: Filter vehicles by various attributes (make, model, year, etc.)
- **Value Scoring**: Compare vehicle values within clusters and meta-clusters
- **Security Features**: 
  - Input validation and sanitization
  - Rate limiting
  - CORS protection
  - XSS prevention
  - Secure headers

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
   - PCA reduction to 3 dimensions (P1, P2, P3)
   - Meta-clustering for higher-level vehicle categorization

2. **Model Architecture**:
   - Primary clustering model (4.2GB)
   - Meta-clustering model (38MB)
   - Three PCA component models (68MB each)

3. **Value Scoring**:
   - Cluster-based value scoring
   - Meta-cluster value comparison
   - Real-time score updates for new vehicles

## Current Status

- ✅ ML pipeline implementation complete
- ✅ Frontend visualization and interaction
- ✅ Backend API and database
- ✅ Security measures implemented
- ✅ Real-time predictions working
- 🔄 Hosting setup in progress

## Future Improvements

- [ ] Add comprehensive test suite
- [ ] Implement model caching
- [ ] Add user authentication
- [ ] Deploy to cloud platform
- [ ] Add API documentation
- [ ] Implement CI/CD pipeline

## Notes

- The application requires significant memory for model loading
- Large model files are stored separately
- Database is automatically initialized on first run
- Security measures are in place for production use


