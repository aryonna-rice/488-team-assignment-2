from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from joblib import dump, load
import os
from sklearn.feature_selection import SelectFromModel
import numpy as np
import json

# Define paths
X_train_path = '/nas/longleaf/home/aryonna/488-team-assignment-2/data/train_data/features.joblib'
y_train_path = '/nas/longleaf/home/aryonna/488-team-assignment-2/data/train_data/target.joblib'
checkpoint_path = '/nas/longleaf/home/aryonna/488-team-assignment-2/models/grid_search_checkpoint.joblib'
final_model_path = '/nas/longleaf/home/aryonna/488-team-assignment-2/models/rfc_reduced_data.joblib'

# Load data
X_train = load(X_train_path)
y_train = load(y_train_path)

print("Loaded data")

# Step 1: Initial RandomForest to determine feature importance

print("Starting to fit")
initial_model = RandomForestClassifier(n_estimators=100, random_state=42)
initial_model.fit(X_train, y_train)

print("Selecting most important features")
# Step 2: Feature selection based on importance
selector = SelectFromModel(initial_model, prefit=True, threshold=-np.inf, max_features=10)
X_important_train = selector.transform(X_train)

# Define reduced model with selected features for hyperparameter tuning
model = RandomForestClassifier()

param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt']
}

# Step 3: Hyperparameter tuning on reduced feature set
if os.path.exists(checkpoint_path):
    grid_search = load(checkpoint_path)
    print("Resuming from checkpoint...")
else:
    grid_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=3, n_jobs=-1, verbose=2)
    print("Starting to find best params...")

try:
    grid_search.fit(X_important_train, y_train.ravel())
except Exception as e:
    print("Interrupted. Saving checkpoint...", str(e))
    dump(grid_search, checkpoint_path)
    raise

# Step 4: Best model after hyperparameter tuning
best_model = grid_search.best_estimator_

# Save the best model
dump(best_model, final_model_path)
print("Best parameters found:", grid_search.best_params_)

# Identifying the most important features
selected_features_indices = selector.get_support(indices=True)
selected_features_names = [X_train.columns[i] for i in selected_features_indices]  # Assumes X_train is a DataFrame

# Save the selected feature names or indices
# Adjust this path and file name as necessary
features_path = '../data/selected_features.json'
with open(features_path, 'w') as f:
    json.dump(selected_features_names, f)  # Save names if X_train is a DataFrame, else save selected_features_indices

print("Selected features saved to:", features_path)