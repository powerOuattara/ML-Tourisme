import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def create_features(df):
    """
    Feature Engineering SANS data leakage
    """
    df = df.copy()
    
    print("🔧 FEATURE ENGINEERING EN COURS...")
    
    # ===== NUMÉRIQUES (SANS UTILISER total_cost) =====
    print("1️⃣ Features numériques...")
    
    # Durées de séjour
    df['total_nights'] = df['night_mainland'] + df['night_zanzibar']
    
    # Taille du groupe
    df['group_size'] = df['total_female'] + df['total_male']
    
    # Ratios (SANS total_cost)
    df['ratio_zanzibar'] = df['night_zanzibar'] / (df['total_nights'] + 0.01)
    df['ratio_mainland'] = df['night_mainland'] / (df['total_nights'] + 0.01)
    df['gender_ratio'] = df['total_female'] / (df['total_male'] + 0.01)
    df['pct_female'] = df['total_female'] / (df['group_size'] + 0.01)
    df['pct_male'] = df['total_male'] / (df['group_size'] + 0.01)
    
    # Interactions numériques
    df['nights_x_group'] = df['total_nights'] * df['group_size']  # Capacité totale nuitées
    df['zanzibar_x_group'] = df['night_zanzibar'] * df['group_size']
    df['mainland_x_group'] = df['night_mainland'] * df['group_size']
    
    # Features quadratiques (capturer non-linéarité)
    df['group_size_squared'] = df['group_size'] ** 2
    df['total_nights_squared'] = df['total_nights'] ** 2
    
    # ===== CATÉGORIELLES =====
    print("2️⃣ Features catégorielles combinées...")
    
    # Package count (nombre de services inclus)
    package_cols = [col for col in df.columns if col.startswith('package_')]
    if package_cols:
        # Compter combien de packages = 'Yes'
        df['package_count'] = 0
        for col in package_cols:
            df['package_count'] += (df[col] == 'Yes').astype(int)
        
        df['full_package'] = (df['package_count'] >= 6).astype(int)  # Tous services inclus
        df['partial_package'] = ((df['package_count'] > 0) & (df['package_count'] < 6)).astype(int)
        df['no_package'] = (df['package_count'] == 0).astype(int)
    
    # Destination principale
    df['main_destination'] = (df['night_zanzibar'] > df['night_mainland']).astype(int)
    df['only_zanzibar'] = (df['night_zanzibar'] > 0) & (df['night_mainland'] == 0)
    df['only_mainland'] = (df['night_mainland'] > 0) & (df['night_zanzibar'] == 0)
    df['both_destinations'] = (df['night_zanzibar'] > 0) & (df['night_mainland'] > 0)
    
    # Profil voyageur (interactions catégorielles)
    if 'age_group' in df.columns and 'travel_with' in df.columns:
        df['traveler_profile'] = df['age_group'].astype(str) + '_' + df['travel_with'].astype(str)
    
    if 'purpose' in df.columns and 'age_group' in df.columns:
        df['purpose_age'] = df['purpose'].astype(str) + '_' + df['age_group'].astype(str)
    
    # Groupe familial ?
    if 'travel_with' in df.columns:
        df['is_family'] = df['travel_with'].str.contains('Family|Children', case=False, na=False).astype(int)
    
    # Premier voyage ?
    if 'first_trip_tz' in df.columns:
        df['first_trip_binary'] = (df['first_trip_tz'] == 'Yes').astype(int)
    
    print(f"✅ Features créées : {df.shape[1]} colonnes")
    
    return df

