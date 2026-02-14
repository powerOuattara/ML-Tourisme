
import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Adjust path to import from app/main.py if needed, or just copy the logic
# Copying logic to avoid import issues and side effects
LOW_CARD_COLS = [
    'age_group', 'travel_with', 'purpose', 'main_activity', 'info_source',
    'tour_arrangement', 'package_transport_int', 'package_accomodation',
    'package_food', 'package_transport_tz', 'package_sightseeing',
    'package_guided_tour', 'package_insurance', 'payment_mode',
    'first_trip_tz', 'most_impressing'
]
HIGH_CARD_COLS = ['country']

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # ===== FEATURES NUMÉRIQUES =====
    # Durées de séjour
    df['total_nights'] = df['night_mainland'] + df['night_zanzibar']
    
    # Taille du groupe
    df['group_size'] = df['total_female'] + df['total_male']
    
    # Ratios
    df['ratio_zanzibar'] = df['night_zanzibar'] / (df['total_nights'] + 0.01)
    df['ratio_mainland'] = df['night_mainland'] / (df['total_nights'] + 0.01)
    df['gender_ratio'] = df['total_female'] / (df['total_male'] + 0.01)
    df['pct_female'] = df['total_female'] / (df['group_size'] + 0.01)
    df['pct_male'] = df['total_male'] / (df['group_size'] + 0.01)
    
    # Interactions numériques
    df['nights_x_group'] = df['total_nights'] * df['group_size']
    df['zanzibar_x_group'] = df['night_zanzibar'] * df['group_size']
    df['mainland_x_group'] = df['night_mainland'] * df['group_size']
    
    # Features quadratiques
    df['group_size_squared'] = df['group_size'] ** 2
    df['total_nights_squared'] = df['total_nights'] ** 2
    
    # ===== FEATURES CATÉGORIELLES =====
    
    # Package count
    package_cols = [col for col in df.columns if col.startswith('package_')]
    if package_cols:
        df['package_count'] = 0
        for col in package_cols:
            df['package_count'] += (df[col] == 'Yes').astype(int)
        
        df['full_package'] = (df['package_count'] >= 6).astype(int)
        df['partial_package'] = ((df['package_count'] > 0) & (df['package_count'] < 6)).astype(int)
        df['no_package'] = (df['package_count'] == 0).astype(int)
    
    # Destination
    df['main_destination'] = (df['night_zanzibar'] > df['night_mainland']).astype(int)
    df['only_zanzibar'] = ((df['night_zanzibar'] > 0) & (df['night_mainland'] == 0)).astype(int)
    df['only_mainland'] = ((df['night_mainland'] > 0) & (df['night_zanzibar'] == 0)).astype(int)
    df['both_destinations'] = ((df['night_zanzibar'] > 0) & (df['night_mainland'] > 0)).astype(int)
    
    # Famille
    if 'travel_with' in df.columns:
        df['is_family'] = df['travel_with'].str.contains('Family|Children', case=False, na=False).astype(int)
    
    # Premier voyage
    if 'first_trip_tz' in df.columns:
        df['first_trip_binary'] = (df['first_trip_tz'] == 'Yes').astype(int)
    
    return df

def main():
    base_dir = Path('.')
    model_path = base_dir / "model.pkl"
    scaler_x_path = base_dir / "scaler_x.pkl"
    encoder_path = base_dir / "encoder.pkl"
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_x_path, 'rb') as f:
            scaler_x = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
            
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # Simulate input
    data = {
        "country": "FRANCE",
        "age_group": "25-44",
        "travel_with": "Spouse",
        "total_female": 1,
        "total_male": 1,
        "purpose": "Leisure and Holidays",
        "main_activity": "Wildlife tourism",
        "info_source": "Travel, agent, tour operator",
        "tour_arrangement": "Package Tour",
        "package_transport_int": "Yes",
        "package_accomodation": "Yes",
        "package_food": "Yes",
        "package_transport_tz": "Yes",
        "package_sightseeing": "Yes",
        "package_guided_tour": "Yes",
        "package_insurance": "No",
        "night_mainland": 5,
        "night_zanzibar": 3,
        "payment_mode": "Cash",
        "first_trip_tz": "Yes",
        "most_impressing": "Culture and Heritage"
    }
    
    df = pd.DataFrame([data])
    df_features = create_features(df)
    
    # Preprocess sim
    num_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_features.select_dtypes(include=['object']).columns.tolist()
    
    low_card = [col for col in cat_cols if col in LOW_CARD_COLS]
    high_card = [col for col in cat_cols if col in HIGH_CARD_COLS]
    
    if low_card:
        cat_encoded_low = encoder.transform(df_features[low_card])
        cat_encoded_low_df = pd.DataFrame(
            cat_encoded_low,
            columns=encoder.get_feature_names_out(),
            index=df_features.index
        )
    else:
        cat_encoded_low_df = pd.DataFrame(index=df_features.index)
        
    cat_encoded_high_df = pd.DataFrame(index=df_features.index)
    for col in high_card:
        cat_encoded_high_df[col] = 0.01
        
    num_log = np.log1p(df_features[num_cols])
    
    X_transformed = pd.concat([
        num_log.reset_index(drop=True),
        cat_encoded_low_df.reset_index(drop=True),
        cat_encoded_high_df.reset_index(drop=True)
    ], axis=1)
    
    print("\nXXX Scaler X Expected Features (if available):")
    if hasattr(scaler_x, 'feature_names_in_'):
        print(list(scaler_x.feature_names_in_))
        print(f"Count: {len(scaler_x.feature_names_in_)}")
    else:
        print("scaler_x does not have feature_names_in_")
        
    print("\nXXX Generated Features in API:")
    print(list(X_transformed.columns))
    print(f"Count: {len(X_transformed.columns)}")
    
    if hasattr(scaler_x, 'feature_names_in_'):
         missing_in_scaler = set(X_transformed.columns) - set(scaler_x.feature_names_in_)
         missing_in_api = set(scaler_x.feature_names_in_) - set(X_transformed.columns)
         
         print(f"\nFeatures in API but NOT in Scaler (Unseen): {missing_in_scaler}")
         print(f"Features in Scaler but NOT in API (Missing): {missing_in_api}")

    print("\nXXX Model Expected Features:")
    if hasattr(model, 'feature_names_in_'):
        print(list(model.feature_names_in_))
    else:
        print("Model does not have feature_names_in_")

    try:
        scaler_x.transform(X_transformed)
        print("\nScaler transform SUCCESS.")
    except Exception as e:
        print(f"\nScaler transform FAILED: {e}")

if __name__ == "__main__":
    main()
