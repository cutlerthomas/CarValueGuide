import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump
from tqdm import tqdm
import numpy as np

df = pd.read_csv('../final_vehicle_data.csv')

# Assume df is your dataframe with the raw features and the 'cluster' column as the target.
features = ['Year', 'Engine Fuel Type', 'Engine HP', 'Engine Cylinders',
            'Transmission Type', 'Driven_Wheels', 'Number of Doors', 'Market Category',
            'Vehicle Size', 'Vehicle Style', 'highway MPG', 'city mpg', 'Popularity', 'cluster']
target = 'P1'

X = df[features]
y = df[target]

# Define categorical and numerical features
categorical_features = ['Engine Fuel Type', 'Transmission Type', 'Driven_Wheels', 'Market Category', 'Vehicle Size', 'Vehicle Style']
numerical_features = ['Year', 'Engine HP', 'Engine Cylinders', 'Number of Doors', 'highway MPG', 'city mpg', 'Popularity', 'cluster']

# Preprocessor for scaling numerical features and encoding categorical features
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Create a pipeline with preprocessing and regressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

print("Starting cross-validation...")
# Define a 5-fold cross-validation strategy with shuffling
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate the model using cross-validation with tqdm progress bar
cv_scores = []
for train_idx, val_idx in tqdm(cv.split(X), total=5, desc="Cross-validation folds"):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)
    
    # Evaluate on validation data
    score = pipeline.score(X_val, y_val)
    cv_scores.append(score)

print("Training final model...")
# Train the final model
pipeline.fit(X, y)

print("Saving model...")
# Save the model
dump(pipeline, 'vehicle_p1_model.joblib')
print("Cross Validation R2 Scores(P1):", cv_scores)
print("Mean CV R2 Score(P1):", np.mean(cv_scores))