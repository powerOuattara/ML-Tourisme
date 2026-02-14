import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from xgboost import XGBRegressor
import sys

# Import feature engineering
try:
    from feature_engineering import create_features
except ImportError:
    print("❌ feature_engineering.py not found. Make sure it is in the same directory.")
    sys.exit(1)

# Paths
BASE_DIR = Path("d:/Zindi/Zindi_Prévisions touristiques en Tanzanie")
ML_DIR = BASE_DIR / "tourisme_ml"
TRAIN_PATH = BASE_DIR / "Train.csv"

# Artifact Paths
MODEL_PATH = ML_DIR / "model.pkl"
SCALER_X_PATH = ML_DIR / "scaler_x.pkl"
SCALER_Y_PATH = ML_DIR / "scaler_y.pkl"
ENCODER_PATH = ML_DIR / "encoder.pkl"

def train_pipeline():
    print("🚀 STARTING TRAINING PIPELINE...")
    
    # 1. Load Data
    if not TRAIN_PATH.exists():
        print(f"❌ Train.csv not found at {TRAIN_PATH}")
        return
    
    df = pd.read_csv(TRAIN_PATH)
    print(f"   Data loaded: {df.shape}")
    
    # 2. Feature Engineering
    print("   Applying feature engineering...")
    df = create_features(df)
    
    # 3. Define Columns (Same as main.py)
    LOW_CARD_COLS = [
        'age_group', 'travel_with', 'purpose', 'main_activity', 'info_source',
        'tour_arrangement', 'package_transport_int', 'package_accomodation',
        'package_food', 'package_transport_tz', 'package_sightseeing',
        'package_guided_tour', 'package_insurance', 'payment_mode',
        'first_trip_tz', 'most_impressing'
    ]
    HIGH_CARD_COLS = ['country']
    
    # Filter columns that actually exist in df
    low_card = [c for c in LOW_CARD_COLS if c in df.columns]
    high_card = [c for c in HIGH_CARD_COLS if c in df.columns]
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove target from num_cols if present
    if 'total_cost' in num_cols:
        num_cols.remove('total_cost')
        
    print(f"   Numeric features: {len(num_cols)}")
    print(f"   Categorical features: {len(low_card) + len(high_card)}")
    
    # 4. Prepare Target (Log Transform + RobustScaler)
    print("   Preparing target...")
    y = df['total_cost']
    y_log = np.log1p(y)
    
    scaler_y = RobustScaler()
    y_scaled = scaler_y.fit_transform(y_log.values.reshape(-1, 1)).flatten()
    
    # 5. Encoding & Scaling Inputs
    print("   Fitting encoders and scalers...")
    
    # One-Hot Encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_encoded_low = encoder.fit_transform(df[low_card])
    cat_encoded_low_df = pd.DataFrame(cat_encoded_low, columns=encoder.get_feature_names_out(low_card))
    
    # Frequency Encoding (Simple version matching main.py logic roughly)
    # For training, we use REAL frequencies.
    cat_encoded_high_df = pd.DataFrame(index=df.index)
    for col in high_card:
        freq = df[col].value_counts(normalize=True)
        cat_encoded_high_df[col] = df[col].map(freq)
        
    # Log transform numeric
    num_log = np.log1p(df[num_cols])
    
    # Concat
    X_transformed = pd.concat([
        num_log.reset_index(drop=True),
        cat_encoded_low_df.reset_index(drop=True),
        cat_encoded_high_df.reset_index(drop=True)
    ], axis=1)
    
    print(f"   X_transformed shape: {X_transformed.shape}")
    
    # Scale X
    scaler_x = RobustScaler()
    X_scaled = scaler_x.fit_transform(X_transformed)
    
    # 6. Train Model (XGBoost)
    print("   Training XGBoost model...")
    # Hyperparameters from optimization (Best Params)
    model = XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.017,
        subsample=0.84,
        colsample_bytree=0.64,
        gamma=0.25,
        random_state=42
    )
    model.fit(X_scaled, y_scaled)
    print("   Model trained.")
    
    # 7. Save Artifacts
    print("   Saving artifacts...")
    
    # Ensure directory exists
    ML_DIR.mkdir(exist_ok=True)
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_X_PATH, 'wb') as f:
        pickle.dump(scaler_x, f)
    with open(SCALER_Y_PATH, 'wb') as f:
        pickle.dump(scaler_y, f)
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(encoder, f)
        
    print(f"✅ Training Complete! Artifacts saved to {ML_DIR}")
    print("   Now RESTART your API to load these new files.")

if __name__ == "__main__":
    train_pipeline()
