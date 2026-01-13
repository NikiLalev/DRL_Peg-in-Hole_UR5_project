import numpy as np

data = np.load("./checkpoints/sac/circle/peg_in_hole_iql_dataset.npz", allow_pickle=True)

print("Keys in npz:", data.files)

for k in data.files:
    print(f"{k}: shape={data[k].shape}, dtype={data[k].dtype}")
