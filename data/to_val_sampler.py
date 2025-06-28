import pandas as pd
import os

# Load full dataset
df = pd.read_csv("data/csv/movements_to_validate.csv")

# Shuffle the dataset once for reproducibility
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Define sample size and output folder
sample_size = 500
output_dir = "data/csv/samples"
os.makedirs(output_dir, exist_ok=True)

# Calculate number of samples to create
n_samples = len(df) // sample_size

# Create and save samples
for i in range(n_samples):
    start = i * sample_size
    end = start + sample_size
    sample_df = df.iloc[start:end]
    sample_df.to_csv(f"{output_dir}/sample_{i+1}.csv", index=False)
    print(f"[OK] Saved sample_{i+1}.csv with {len(sample_df)} rows")

# Handle remaining rows
remainder = len(df) % sample_size
if remainder > 0:
    leftover_df = df.iloc[-remainder:]
    leftover_df.to_csv(f"{output_dir}/sample_{n_samples+1}.csv", index=False)
    print(f"[OK] Saved final sample with {len(leftover_df)} rows")
