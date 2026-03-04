import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. GENERATE / LOAD DATA (The "Current Changes" Version)
# ==========================================
# We now include: Standard 5, PA (Nylon), and no_microplastic
print("Generating comprehensive dataset for visualization...")

fingerprints = {
    'PE (Polyethylene)': [2915, 2850, 1470, 1463, 720, 730],
    'PP (Polypropylene)': [2950, 2870, 1460, 1377, 1167, 998, 973, 841],
    'PS (Polystyrene)': [3026, 2921, 1601, 1493, 1452, 1028, 698],
    'PET (Polyester)': [2960, 1715, 1410, 1240, 1090, 1018, 725],
    'PVC (Polyvinyl Chloride)': [2910, 1425, 1330, 1250, 960, 690, 615],
    'PA (Polyamide)': [3300, 2930, 2860, 1640, 1545, 1465, 1265, 935, 690, 580] # Added PA
}

data = []
max_peaks = 10
np.random.seed(42)

# Generate Plastics (50 each)
for label, peaks in fingerprints.items():
    for _ in range(50):
        sim_peaks = [round(p + np.random.normal(0, 4), 2) for p in peaks]
        while len(sim_peaks) < max_peaks: sim_peaks.append(0)
        data.append([label] + sim_peaks[:max_peaks])

# Generate "no_microplastic" (50 samples)
for _ in range(50):
    num_peaks = np.random.randint(4, 9)
    # Random peaks across the spectrum
    rand_peaks = sorted(np.random.uniform(400, 3800, num_peaks), reverse=True)
    p_vals = [round(p, 2) for p in rand_peaks]
    while len(p_vals) < max_peaks: p_vals.append(0)
    data.append(["no_microplastic"] + p_vals[:max_peaks])

# Create DataFrame
cols = ["Label"] + [f"Peak_{i+1}" for i in range(max_peaks)]
full_dataset = pd.DataFrame(data, columns=cols)

# ==========================================
# 2. TRAIN MODEL
# ==========================================
X = full_dataset.drop(columns=['Label'])
y = full_dataset['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Model training complete. Generating charts...")

# ==========================================
# CHART 1: CONFUSION MATRIX
# ==========================================
plt.figure(figsize=(10, 8)) # Made figure slightly taller for more labels
cm = confusion_matrix(y_test, y_pred)
labels = sorted(y.unique())

sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix\n(Includes PA & no_microplastic)')
plt.ylabel('Actual Material')
plt.xlabel('Predicted Material')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ==========================================
# CHART 2: FEATURE IMPORTANCE
# ==========================================
plt.figure(figsize=(10, 6))
importances = clf.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.title('Feature Importance: Which Peaks Identify Microplastics?')
plt.bar(range(X.shape[1]), importances[indices], align='center', color='teal')
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=45)
plt.ylabel('Importance Score')
plt.tight_layout()
plt.show()