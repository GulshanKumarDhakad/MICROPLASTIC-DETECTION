import pandas as pd
import numpy as np

# 1. Setup Fingerprints
polymer_fingerprints = {
    'PE (Polyethylene)': [2915, 2850, 1470, 1463, 720, 730],
    'PP (Polypropylene)': [2950, 2870, 1460, 1377, 1167, 998, 973, 841],
    'PS (Polystyrene)': [3026, 2921, 1601, 1493, 1452, 1028, 698],
    'PET (Polyester)': [2960, 1715, 1410, 1240, 1090, 1018, 725],
    'PVC (Polyvinyl Chloride)': [2910, 1425, 1330, 1250, 960, 690, 615]
}

data = []
max_peaks = 10
np.random.seed(999) # Using a new random seed for "Blind" testing

# 2. Generate Plastic Samples
for label, peaks in polymer_fingerprints.items():
    # Create 2 samples for each plastic type
    for i in range(1, 3):
        sample_name = f"Mystery_Sample_{label.split()[0]}_{i}"
        
        # Add random noise (+/- 5 cm-1)
        simulated_peaks = [round(p + np.random.normal(0, 5), 2) for p in peaks]
        
        # Pad with 0s
        while len(simulated_peaks) < max_peaks:
            simulated_peaks.append(0)
            
        data.append([sample_name] + simulated_peaks[:max_peaks])

# 3. Generate "no_microplastic" Samples
for i in range(1, 3):
    sample_name = f"Mystery_Sample_Unknown_{i}"
    
    # Generate random garbage peaks
    num_peaks = np.random.randint(3, 8)
    random_peaks = sorted(np.random.uniform(400, 3800, num_peaks), reverse=True)
    p_vals = [round(p, 2) for p in random_peaks]
    
    while len(p_vals) < max_peaks:
        p_vals.append(0)
        
    data.append([sample_name] + p_vals[:max_peaks])

# 4. Save to CSV
cols = ["sample_name"] + [f"Peak_{i+1}" for i in range(max_peaks)]
df = pd.DataFrame(data, columns=cols)
df.to_csv("blind_test_dataset.csv", index=False)

print(f"✅ Created 'blind_test_dataset.csv' with {len(df)} samples.")