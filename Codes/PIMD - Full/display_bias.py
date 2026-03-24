import json
import os
import numpy as np

record_file = os.path.join('data', 'bias_record.json')

if not os.path.exists(record_file):
    print(f"Error: Could not find {record_file}")
    exit(1)

with open(record_file, 'r') as f:
    data = json.load(f)

print(f"{'N':<5} | {'L2 Bias':<20}")
print("-" * 30)

for N in [8, 12, 16, 20, 24, 28, 32]:
    key = str(N)
    if key in data:
        biases = np.array(data[key])
        # Calculate the L2 norm of the bias vector
        l2_bias = np.linalg.norm(biases)
        print(f"{N:<5} | {l2_bias:<20.8e}")
    else:
        print(f"{N:<5} | {'N/A':<20}")