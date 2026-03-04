import pandas as pd
import joblib
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
# Change this line:
INPUT_FILE = "sample.csv"
MODEL_FILE = "microplastics_brain.pkl"
OUTPUT_FILE = "final_results.csv"

# ==========================================
# 1. LOAD RESOURCES
# ==========================================
try:
    print(f"Loading Model...")
    clf = joblib.load(MODEL_FILE)
    
    print(f"Reading Data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print("❌ Error: Missing files. Did you run the Training script?")
    exit()

# ==========================================
# 2. FORMAT DATA (Standardize to 10 Peaks)
# ==========================================
expected_cols = [f"Peak_{i+1}" for i in range(10)]
X_input = pd.DataFrame()

for col in expected_cols:
    if col in df.columns:
        X_input[col] = df[col]
    else:
        X_input[col] = 0

X_input = X_input.fillna(0)

# ==========================================
# 3. PREDICT
# ==========================================
print("Analyzing samples...")

# The model now has "no_microplastic" as a built-in option!
predictions = clf.predict(X_input)
probs = clf.predict_proba(X_input)
confidence = np.max(probs, axis=1) * 100

# ==========================================
# 4. SAVE RESULTS
# ==========================================
df['Predicted_Result'] = predictions
df['Confidence (%)'] = confidence.round(2)

print("\n" + "="*40)
print("       DETECTION RESULTS")
print("="*40)

for index, row in df.iterrows():
    print(f"Sample: {row.get('sample_name', 'Unknown')}")
    print(f"Result: {row['Predicted_Result']}")
    print(f"Score:  {row['Confidence (%)']}%")
    print("-" * 30)

df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Full results saved to '{OUTPUT_FILE}'")