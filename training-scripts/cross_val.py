import numpy as np
from joblib import load
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.ensemble import RandomForestClassifier  # Changed from RandomForestRegressor

# Paths remain unchanged
X_path = '/nas/longleaf/home/aryonna/488-team-assignment-2/data/raw_data/features.joblib'
y_path = '/nas/longleaf/home/aryonna/488-team-assignment-2/data/raw_data/target.joblib'
model_path = '/nas/longleaf/home/aryonna/488-team-assignment-2/models/rfc_reduced_data_1.joblib'  # Ensure this is a classifier model
results_path = '/nas/longleaf/home/aryonna/488-team-assignment-2/cross-val-results/rfc_reduced_results_1.txt'

# Load your dataset
print("Begin load data")
X = load(X_path)
y = load(y_path)

print("Begin load model")
# Ensure the loaded model is a RandomForestClassifier
model = load(model_path)

# Specify the cross-validation strategy
print("Starting cross-validation")
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the scorer for classification task (change this according to your specific needs, e.g., accuracy, f1)
scorer = make_scorer(accuracy_score)  # Changed from mean_squared_error to accuracy_score

# Compute the cross-validation scores in parallel
scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)

# Prepare the string to save to a file
output_string = f'Average Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}'

# Open the file in write mode ('w') to save the output
with open(results_path, 'w') as file:
    # Write the Accuracy for each fold
    for i, score in enumerate(scores, start=1):
        file.write(f'Fold {i} Accuracy: {score:.4f}\n')
    
    # Write the average Accuracy and standard deviation
    file.write(f'\nAverage Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}')
