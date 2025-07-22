import pickle
import os

# Check the actual structure of DemBones results
results_path = r"D:\Code\VVEditor\Rafa_Approves_hd_4k\generated_skeletons\dembone_results.pkl"

with open(results_path, 'rb') as f:
    results = pickle.load(f)

print("DemBones results keys:", list(results.keys()))
for key, value in results.items():
    if hasattr(value, 'shape'):
        print(f"  {key}: {value.shape} ({type(value)})")
    elif isinstance(value, (list, tuple)):
        print(f"  {key}: length {len(value)} ({type(value)})")
    else:
        print(f"  {key}: {type(value)}")
