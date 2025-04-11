import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load
import numpy as np
from tqdm import tqdm

print("Loading data...")
df = pd.read_csv('../final_vehicle_data.csv')

features = ['Year', 'Engine Fuel Type', 'Engine HP', 'Engine Cylinders',
            'Transmission Type', 'Driven_Wheels', 'Number of Doors', 'Market Category',
            'Vehicle Size', 'Vehicle Style', 'highway MPG', 'city mpg', 'Popularity']
target = 'cluster'

X = df[features]
y = df[target]

# Define categorical and numerical features
categorical_features = ['Engine Fuel Type', 'Transmission Type', 'Driven_Wheels', 'Market Category', 'Vehicle Size', 'Vehicle Style']
numerical_features = ['Year', 'Engine HP', 'Engine Cylinders', 'Number of Doors', 'highway MPG', 'city mpg', 'Popularity']

print("Preprocessing data...")
# Preprocessor for scaling numerical features and encoding categorical features
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Create a pipeline with preprocessing and DecisionTree classifier
clf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(
        random_state=42,
        criterion='entropy',    # Information gain for better splits
        min_samples_leaf=1,    # Allow single sample leaves
        min_samples_split=2,   # Minimum splits for granularity
        max_depth=45,          # Even deeper tree for complex patterns
        max_features=0.9,      # Use 90% of features at each split
        class_weight='balanced',# Handle class imbalance
        splitter='best',       # Use best splits
        min_weight_fraction_leaf=0.0,  # Allow very small leaf nodes
        min_impurity_decrease=0.0001   # Require minimum improvement for splits
    ))
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
    clf_pipeline.fit(X_train, y_train)
    
    # Evaluate on validation data
    score = clf_pipeline.score(X_val, y_val)
    cv_scores.append(score)

print("Training final model...")
# Train the final model
clf_pipeline.fit(X, y)

print("Saving model...")
# Save the model
dump(clf_pipeline, 'vehicle_cluster_model.joblib')
print("Cross Validation Accuracy Scores(cluster):", cv_scores)
print("Mean CV Accuracy(cluster):", np.mean(cv_scores))

