import pandas as pd
import numpy as np

np.random.seed(42)

# ==========================================
# PART 1: Generate Standard Plastics Data
# ==========================================
print("Generating standard plastics dataset...")

polymer_fingerprints = {
    'PE (Polyethylene)': [2915, 2850, 1470, 1463, 720, 730],
    'PP (Polypropylene)': [2950, 2870, 1460, 1377, 1167, 998, 973, 841],
    'PS (Polystyrene)': [3026, 2921, 1601, 1493, 1452, 1028, 698],
    'PET (Polyester)': [2960, 1715, 1410, 1240, 1090, 1018, 725],
    'PVC (Polyvinyl Chloride)': [2910, 1425, 1330, 1250, 960, 690, 615]
}

# Determine max peaks for column structure (10 to match SPRITE data)
max_peaks = 10 
num_samples = 50
data_plastics = []

for plastic_type, peaks in polymer_fingerprints.items():
    for _ in range(num_samples):
        sample_peaks = []
        for peak in peaks:
            shift = np.random.normal(0, 3)
            sample_peaks.append(round(peak + shift, 2))
        
        # Fill remaining spots with 0
        while len(sample_peaks) < max_peaks:
            sample_peaks.append(0)
            
        data_plastics.append([plastic_type] + sample_peaks)

cols = ["plastic_type"] + [f"Peak_{i+1}" for i in range(max_peaks)]
df_plastics = pd.DataFrame(data_plastics, columns=cols)
df_plastics.to_csv("microplastics_wide_format.csv", index=False)
print("✅ Created 'microplastics_wide_format.csv'")

# ==========================================
# PART 2: Generate SPRITE (PDF) Data
# ==========================================
print("Generating SPRITE simulated dataset...")

base_peaks_sprite = [3272.00, 2342.76, 2103.90, 1634.10, 1140.03, 
                     1057.06, 1000.34, 928.05, 576.04, 464.35]

data_sprite = []

for i in range(num_samples):
    row = ["SPRITE_Simulated"]
    for peak in base_peaks_sprite:
        shift = np.random.normal(0, 4)
        row.append(round(peak + shift, 2))
    
    # Ensure it also has 10 columns (it naturally does, but good to be safe)
    while len(row) < (max_peaks + 1): # +1 for the Label column
        row.append(0)
        
    data_sprite.append(row)

cols_sprite = ["sample_type"] + [f"Peak_{i+1}" for i in range(max_peaks)]
df_sprite = pd.DataFrame(data_sprite, columns=cols_sprite)
df_sprite.to_csv("sprite_simulated_data.csv", index=False)
print("✅ Created 'sprite_simulated_data.csv'")

print("\n🎉 All files are ready! Now run your ML code again.")