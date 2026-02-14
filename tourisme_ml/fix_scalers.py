"""
Script to regenerate corrupted scaler files

This script recreates the scaler_y.pkl and scaler_x.pkl files that were corrupted.
Run this script from the tourisme_ml directory after ensuring the training notebook
has been executed and the scalers are available in the notebook's scope.
"""
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

print("🔧 Fixing corrupted scaler files...")

# Check if we can load the existing model to understand the expected format
try:
    with open('../model.pkl', 'rb') as f:
        model = pickle.load(f)
    print(f"✓ Model loaded successfully: {type(model).__name__}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Create fresh scaler instances
# These will need to be fitted with the actual training data from the notebook
scaler_y = StandardScaler()
scaler_x = StandardScaler()

print("\n⚠️  WARNING: The scalers created here are NOT fitted!")
print("   You need to:")
print("   1. Open the training notebook (tourisme.ipynb)")
print("   2. Find the cells where scaler_x and scaler_y are fitted")
print("   3. Re-run those cells")
print("   4. Re-run the cell that saves the scalers (lines with pickle.dump)")
print("\nAlternatively, if you have the fitted scalers in memory in the notebook:")
print("   Run this code in a notebook cell:")
print("""
import pickle

# Save scaler_y
with open('tourisme_ml/scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)
print("✅ scaler_y.pkl saved")

# Save scaler_x  
with open('tourisme_ml/scaler_x.pkl', 'wb') as f:
    pickle.dump(scaler_x, f)
print("✅ scaler_x.pkl saved")
""")
