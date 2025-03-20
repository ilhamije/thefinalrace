import matplotlib.pyplot as plt
import json
import os
import numpy as np
from glob import glob

# Find all scalars.json files
base_dir = "."  # Current directory, change if needed
json_files = glob(os.path.join(base_dir, "*/vis_data/scalars.json"))

# Dictionary to store data from each run
runs_data = {}

# Parse all JSON files
for json_file in json_files:
    run_name = os.path.basename(os.path.dirname(os.path.dirname(json_file)))
    data = []

    with open(json_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                data.append(entry)
            except:
                continue

    if data:  # Only add if we successfully parsed data
        runs_data[run_name] = data

# Create plots
plt.figure(figsize=(12, 10))

# Plot Loss
plt.subplot(2, 1, 1)
for run_name, data in runs_data.items():
    iterations = [entry['iter'] for entry in data]
    loss = [entry['loss'] for entry in data]
    plt.plot(iterations, loss, label=f"{run_name}")

plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.legend()

# Plot Accuracy
plt.subplot(2, 1, 2)
for run_name, data in runs_data.items():
    iterations = [entry['iter'] for entry in data]
    accuracy = [entry['decode.acc_seg'] for entry in data]
    plt.plot(iterations, accuracy, label=f"{run_name}")

plt.xlabel('Iteration')
plt.ylabel('Segmentation Accuracy (%)')
plt.title('Segmentation Accuracy')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('combined_training_curves.png', dpi=300)
print("Combined plot saved as 'combined_training_curves.png'")
