import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

print("--- TRAINING MODEL WITH COMPREHENSIVE DATASET ---")

# 1. LOAD THE DATASET
# We load the file that contains ALL plastics + "no_microplastic"
dataset_file = "comprehensive_microplastics_dataset.csv"

try:
    print(f"Loading data from {dataset_file}...")
    df = pd.read_csv(dataset_file)
except FileNotFoundError:
    print(f"❌ Error: Could not find '{dataset_file}'.")
    print("Please run the data generation script from the previous step first!")
    exit()

# 2. PREPARE TRAINING DATA
# The dataset already has 'Label' and 'Peak_1'...'Peak_10'
# We just separate the Questions (Peaks) from the Answers (Label)
X = df.drop(columns=['Label'])  # The Features (Peaks)
y = df['Label']                 # The Target (Plastic Name)

print(f"Training on {len(df)} samples.")
print(f"Classes found: {y.unique()}")

# 3. TRAIN THE MODEL
# n_estimators=200 makes it more accurate
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X, y)

# 4. SAVE THE BRAIN
model_filename = "microplastics_brain.pkl"
joblib.dump(clf, model_filename)

print("\n" + "="*50)
print(f"✅ MODEL Brain successfully saved to '{model_filename}'")
print("="*50)
print("You can now run 'predict_future.py' to test new samples.")