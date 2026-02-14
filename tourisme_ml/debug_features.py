"""
Script de débogage pour comprendre le problème de shape mismatch
"""
import pandas as pd
import numpy as np
import pickle

# Charger un exemple de données
test_data = {
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

df = pd.DataFrame([test_data])
print("📊 Données initiales:")
print(f"   Colonnes: {len(df.columns)}")
print(f"   {list(df.columns)}\n")

# 1. Feature Engineering
def create_features(df):
    df = df.copy()
    
    # Numériques
    df['total_nights'] = df['night_mainland'] + df['night_zanzibar']
    df['group_size'] = df['total_female'] + df['total_male']
    df['ratio_zanzibar'] = df['night_zanzibar'] / (df['total_nights'] + 0.01)
    df['ratio_mainland'] = df['night_mainland'] / (df['total_nights'] + 0.01)
    df['gender_ratio'] = df['total_female'] / (df['total_male'] + 0.01)
    df['pct_female'] = df['total_female'] / (df['group_size'] + 0.01)
    df['pct_male'] = df['total_male'] / (df['group_size'] + 0.01)
    df['nights_x_group'] = df['total_nights'] * df['group_size']
    df['zanzibar_x_group'] = df['night_zanzibar'] * df['group_size']
    df['mainland_x_group'] = df['night_mainland'] * df['group_size']
    df['group_size_squared'] = df['group_size'] ** 2
    df['total_nights_squared'] = df['total_nights'] ** 2
    
    # Package features
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
    
    # Profil (FEATURES CATÉGORIELLES!)
    if 'age_group' in df.columns and 'travel_with' in df.columns:
        df['traveler_profile'] = df['age_group'].astype(str) + '_' + df['travel_with'].astype(str)
    
    if 'purpose' in df.columns and 'age_group' in df.columns:
        df['purpose_age'] = df['purpose'].astype(str) + '_' + df['age_group'].astype(str)
    
    # Famille
    if 'travel_with' in df.columns:
        df['is_family'] = df['travel_with'].str.contains('Family|Children', case=False, na=False).astype(int)
    
    # Premier voyage
    if 'first_trip_tz' in df.columns:
        df['first_trip_binary'] = (df['first_trip_tz'] == 'Yes').astype(int)
    
    return df

df_features = create_features(df)
print("🔧 Après create_features:")
print(f"   Colonnes: {len(df_features.columns)}")
num_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_features.select_dtypes(include=['object']).columns.tolist()
print(f"   Numériques: {len(num_cols)}")
print(f"   Catégorielles: {len(cat_cols)}")
print(f"   Cat cols: {cat_cols}\n")

# 2. Charger l'encoder
with open('../encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

print("📦 Encoder chargé:")
print(f"   Catégories apprises: {encoder.categories_}")
print(f"   Features input: {encoder.feature_names_in_}")
print(f"   Features output: {len(encoder.get_feature_names_out())}\n")

# 3. Séparer numériques / catégorielles
print("🔍 Analyse des colonnes catégorielles:")
LOW_CARD_COLS = [
    'age_group', 'travel_with', 'purpose', 'main_activity', 'info_source',
    'tour_arrangement', 'package_transport_int', 'package_accomodation',
    'package_food', 'package_transport_tz', 'package_sightseeing',
    'package_guided_tour', 'package_insurance', 'payment_mode',
    'first_trip_tz', 'most_impressing'
]
HIGH_CARD_COLS = ['country']

low_card_in_df = [col for col in cat_cols if col in LOW_CARD_COLS]
high_card_in_df = [col for col in cat_cols if col in HIGH_CARD_COLS]
other_cat = [col for col in cat_cols if col not in LOW_CARD_COLS and col not in HIGH_CARD_COLS]

print(f"   Low cardinality ({len(low_card_in_df)}): {low_card_in_df}")
print(f"   High cardinality ({len(high_card_in_df)}): {high_card_in_df}")
print(f"   ⚠️ AUTRES ({len(other_cat)}): {other_cat}\n")

# 4. Encoder les low cardinality
try:
    cat_encoded_low = encoder.transform(df_features[low_card_in_df])
    print(f"✅ One-Hot Encoding: {cat_encoded_low.shape[1]} colonnes")
except Exception as e:
    print(f"❌ Erreur encoding: {e}\n")

# 5. Frequency encoding (simulé)
cat_encoded_high_df = pd.DataFrame(index=df_features.index)
for col in high_card_in_df:
    cat_encoded_high_df[col] = 0.01  # Valeur par défaut
print(f"✅ Frequency Encoding: {cat_encoded_high_df.shape[1]} colonnes")

# 6. Log transform
num_log = np.log1p(df_features[num_cols])
print(f"✅ Log transform: {num_log.shape[1]} colonnes numériques")

# 7. Concat
X_transformed = pd.concat([
    num_log.reset_index(drop=True),
    pd.DataFrame(cat_encoded_low, columns=encoder.get_feature_names_out()).reset_index(drop=True),
    cat_encoded_high_df.reset_index(drop=True)
], axis=1)

print(f"\n🎯 RÉSULTAT FINAL:")
print(f"   Shape: {X_transformed.shape}")
print(f"   Attendu par le modèle: (1, 91)")
print(f"   Différence: {91 - X_transformed.shape[1]} colonnes manquantes")

if X_transformed.shape[1] != 91:
    print(f"\n❌ PROBLÈME: Il manque {91 - X_transformed.shape[1]} colonnes!")
    print(f"\n💡 Hypothèses:")
    print(f"   1. Les colonnes 'traveler_profile' et 'purpose_age' ne sont pas encodées")
    print(f"   2. L'encoder a été entraîné avec d'autres colonnes")
    print(f"   3. Il manque des features numériques créées")
