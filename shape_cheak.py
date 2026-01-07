import numpy as np

data = np.load("peg_in_hole_iql_dataset.npz")

print("Keys in npz:", data.files)

for k in data.files:
    print(f"{k}: shape={data[k].shape}, dtype={data[k].dtype}")
