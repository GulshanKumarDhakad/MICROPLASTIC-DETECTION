import pandas as pd
import numpy as np

# 1. Define the Plastic Fingerprints
polymer_fingerprints = {
    'PE (Polyethylene)': [2915, 2850, 1470, 1463, 720, 730],
    'PP (Polypropylene)': [2950, 2870, 1460, 1377, 1167, 998, 973, 841],
    'PS (Polystyrene)': [3026, 2921, 1601, 1493, 1452, 1028, 698],
    'PET (Polyester)': [2960, 1715, 1410, 1240, 1090, 1018, 725],
    'PVC (Polyvinyl Chloride)': [2910, 1425, 1330, 1250, 960, 690, 615]
}

# 2. Configuration
num_samples_per_type = 50  # 50 samples for each type
max_peaks = 10             # Columns Peak_1 to Peak_10
data = []

np.random.seed(42)

# 3. Generate "Plastic" Data
for label, peaks in polymer_fingerprints.items():
    for i in range(num_samples_per_type):
        # Add slight random noise to simulate real readings
        simulated_peaks = [round(p + np.random.normal(0, 4), 2) for p in peaks]
        
        # Pad with 0s if fewer than 10 peaks
        while len(simulated_peaks) < max_peaks:
            simulated_peaks.append(0)
            
        # Add to list
        data.append([label] + simulated_peaks[:max_peaks])

# 4. Generate "no_microplastic" Data
# These represent random noise, sand, or dirt
for i in range(num_samples_per_type):
    label = "no_microplastic"
    
    # Strategy: Generate 4-8 random peaks between 400 and 3800 cm^-1
    # These will likely NOT match the plastic fingerprints
    num_random_peaks = np.random.randint(4, 9)
    random_peaks = sorted(np.random.uniform(400, 3800, num_random_peaks), reverse=True)
    
    simulated_peaks = [round(p, 2) for p in random_peaks]
    
    # Pad with 0s
    while len(simulated_peaks) < max_peaks:
        simulated_peaks.append(0)
        
    data.append([label] + simulated_peaks[:max_peaks])

# 5. Create DataFrame and Save
cols = ["Label"] + [f"Peak_{i+1}" for i in range(max_peaks)]
df_final = pd.DataFrame(data, columns=cols)

# Shuffle the rows so they are mixed up
df_final = df_final.sample(frac=1).reset_index(drop=True)

csv_filename = "comprehensive_microplastics_dataset.csv"
df_final.to_csv(csv_filename, index=False)

print(f"✅ Generated file '{csv_filename}' with {len(df_final)} samples.")
print("Includes: PE, PP, PS, PET, PVC, and no_microplastic.")