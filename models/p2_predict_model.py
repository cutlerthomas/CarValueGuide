import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump, load

df = pd.read_csv('../final_vehicle_data.csv')

# Assume df is your dataframe with the raw features and the 'cluster' column as the target.
features = ['Year', 'Engine Fuel Type', 'Engine HP', 'Engine Cylinders',
            'Transmission Type', 'Driven_Wheels', 'Number of Doors', 'Market Category',
            'Vehicle Size', 'Vehicle Style', 'highway MPG', 'city mpg', 'Popularity', 'cluster', 'P1']
target = 'P2'

X = df[features]
y = df[target]

# Define categorical and numerical features
categorical_features = ['Engine Fuel Type', 'Transmission Type', 'Driven_Wheels', 'Market Category', 'Vehicle Size', 'Vehicle Style']
numerical_features = ['Year', 'Engine HP', 'Engine Cylinders', 'Number of Doors', 'highway MPG', 'city mpg', 'Popularity', 'cluster', 'P1']

# Preprocessor for scaling numerical features and encoding categorical features
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Create a pipeline with preprocessing and regressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Define a 5-fold cross-validation strategy with shuffling
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate the model using cross-validation and print the results
cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='r2')

pipeline.fit(X, y)

dump(pipeline, 'vehicle_p2_model.joblib')
print("Cross Validation R2 Scores(P2):", cv_scores)
print("Mean CV R2 Score(P2):", cv_scores.mean())