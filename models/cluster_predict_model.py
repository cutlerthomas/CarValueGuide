import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump, load

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

# Preprocessor for scaling numerical features and encoding categorical features
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Create a pipeline with preprocessing and classifier
clf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define a 5-fold cross-validation strategy with shuffling
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate the model using cross-validation and print the results
cv_scores = cross_val_score(clf_pipeline, X, y, cv=cv, scoring='accuracy')

clf_pipeline.fit(X, y)

dump(clf_pipeline, 'vehicle_cluster_model.joblib')
print("Cross Validation Accuracy Scores(cluster):", cv_scores)
print("Mean CV Accuracy(cluster):", cv_scores.mean())

