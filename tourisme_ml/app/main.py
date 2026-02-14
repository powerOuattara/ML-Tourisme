"""
API FastAPI pour les prédictions de coûts touristiques en Tanzanie

Cette API charge les modèles et transformateurs pré-entraînés et expose un endpoint
de prédiction qui effectue le preprocessing complet des données avant l'inférence.
"""
import os
import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Import du feature engineering partagé
try:
    from feature_engineering import create_features
except ImportError:
    # Fallback si le fichier n'est pas dans le path (ex: exécution directe)
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from feature_engineering import create_features

# ============================================================================
# CONFIGURATION
# ============================================================================

# Chemin vers les fichiers pkl (dans le dossier parent)
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "model.pkl"
# ATTENTION: scaler.pkl contient scaler_y (pas scaler_x !)
# scaler_x n'a jamais été sauvegardé dans le notebook
SCALER_Y_PATH = BASE_DIR / "scaler_y.pkl"  # Ce fichier contient scaler_y
SCALER_X_PATH = BASE_DIR / "scaler_x.pkl"  # Ce fichier contient scaler_x
ENCODER_PATH = BASE_DIR / "encoder.pkl"

# ============================================================================
# INITIALISATION DE L'APPLICATION
# ============================================================================
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(
    title="Tourism Cost Prediction API",
    description="API de prédiction des coûts touristiques en Tanzanie",
    version="1.0.0"
)

# Configuration des CORS (obligatoire pour React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Autorise ton port 5173
    allow_credentials=True,
    allow_methods=["*"], # Autorise OPTIONS, POST, etc.
    allow_headers=["*"],
)

# Variables globales pour stocker les modèles (chargés au démarrage)
model = None
scaler_y = None
encoder = None
scaler_x = None

# Colonnes catégorielles avec faible cardinalité (pour le One-Hot Encoding)
LOW_CARD_COLS = [
    'age_group', 'travel_with', 'purpose', 'main_activity', 'info_source',
    'tour_arrangement', 'package_transport_int', 'package_accomodation',
    'package_food', 'package_transport_tz', 'package_sightseeing',
    'package_guided_tour', 'package_insurance', 'payment_mode',
    'first_trip_tz', 'most_impressing'
]

# Colonnes catégorielles avec haute cardinalité (pour le Frequency Encoding)
HIGH_CARD_COLS = ['country']

# ============================================================================
# MODÈLE PYDANTIC POUR LA VALIDATION DES DONNÉES D'ENTRÉE
# ============================================================================

class TourismInput(BaseModel):
    """Schéma de validation pour les données d'entrée"""
    country: str = Field(..., description="Pays d'origine du touriste")
    age_group: str = Field(..., description="Tranche d'âge")
    travel_with: str = Field(..., description="Avec qui voyage le touriste")
    total_female: int = Field(..., ge=0, description="Nombre de femmes")
    total_male: int = Field(..., ge=0, description="Nombre d'hommes")
    purpose: str = Field(..., description="But du voyage")
    main_activity: str = Field(..., description="Activité principale")
    info_source: str = Field(..., description="Source d'information")
    tour_arrangement: str = Field(..., description="Type d'arrangement")
    package_transport_int: str = Field(..., description="Transport international inclus")
    package_accomodation: str = Field(..., description="Hébergement inclus")
    package_food: str = Field(..., description="Nourriture incluse")
    package_transport_tz: str = Field(..., description="Transport local inclus")
    package_sightseeing: str = Field(..., description="Visites incluses")
    package_guided_tour: str = Field(..., description="Visite guidée incluse")
    package_insurance: str = Field(..., description="Assurance incluse")
    night_mainland: int = Field(..., ge=0, description="Nuits sur le continent")
    night_zanzibar: int = Field(..., ge=0, description="Nuits à Zanzibar")
    payment_mode: str = Field(..., description="Mode de paiement")
    first_trip_tz: str = Field(..., description="Premier voyage en Tanzanie")
    most_impressing: str = Field(..., description="Aspect le plus impressionnant")

    class Config:
        json_schema_extra = {
            "example": {
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
        }

# ============================================================================
# CHARGEMENT DES MODÈLES (EXÉCUTÉ AU DÉMARRAGE)
# ============================================================================

@app.on_event("startup")
async def load_models():
    """Charge tous les modèles et transformateurs au démarrage de l'application"""
    global model, scaler_y, encoder, scaler_x
    
    try:
        print("🔄 Chargement des modèles...")
        
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modèle introuvable: {MODEL_PATH}")
        if not SCALER_Y_PATH.exists():
            raise FileNotFoundError(f"Scaler Y introuvable: {SCALER_Y_PATH}")
        if not SCALER_X_PATH.exists():
            raise FileNotFoundError(f"Scaler X introuvable: {SCALER_X_PATH}")
        if not ENCODER_PATH.exists():
            raise FileNotFoundError(f"Encoder introuvable: {ENCODER_PATH}")
        
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
            print(f"  ✓ Modèle chargé: {type(model).__name__}")
        
        with open(SCALER_Y_PATH, 'rb') as f:
            scaler_y = pickle.load(f)
            print(f"  ✓ Scaler Y chargé: {type(scaler_y).__name__}")
        
        with open(SCALER_X_PATH, 'rb') as f:
            scaler_x = pickle.load(f)
            print(f"  ✓ Scaler X chargé: {type(scaler_x).__name__}")
        
        with open(ENCODER_PATH, 'rb') as f:
            encoder = pickle.load(f)
            print(f"  ✓ Encoder chargé: {type(encoder).__name__}")
        
        print("\n✅ Tous les modèles et transformateurs chargés avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement des modèles: {e}")
        raise

# ============================================================================
# FONCTIONS DE PREPROCESSING
# ============================================================================

# Fonction create_features importée depuis feature_engineering.py
# Suppression de la version locale pour éviter les duplications et incohérences


def preprocess_data(df: pd.DataFrame) -> np.ndarray:
    """
    Applique tout le preprocessing nécessaire pour l'inférence
    
    Args:
        df: DataFrame avec les features engineerées
        
    Returns:
        Array numpy scalé prêt pour la prédiction
    """
    # 1. Séparer numériques et catégoriels
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # 2. Identifier les colonnes low/high cardinality
    low_card = [col for col in cat_cols if col in LOW_CARD_COLS]
    high_card = [col for col in cat_cols if col in HIGH_CARD_COLS]
    
    # 3. One-Hot Encoding sur low cardinality (TRANSFORM uniquement)
    if low_card:
        cat_encoded_low = encoder.transform(df[low_card])
        cat_encoded_low_df = pd.DataFrame(
            cat_encoded_low,
            columns=encoder.get_feature_names_out(),
            index=df.index
        )
    else:
        cat_encoded_low_df = pd.DataFrame(index=df.index)
    
    # 4. Frequency Encoding sur high cardinality
    # Note: En production, il faudrait sauvegarder les fréquences du training
    # Pour l'instant, on utilise une valeur par défaut pour les pays inconnus
    cat_encoded_high_df = pd.DataFrame(index=df.index)
    for col in high_card:
        # Valeur par défaut pour les pays inconnus
        cat_encoded_high_df[col] = 0.01
    
    # 5. Log transformation sur les numériques
    num_log = np.log1p(df[num_cols])
    
    # 6. Concaténer toutes les features
    X_transformed = pd.concat([
        num_log.reset_index(drop=True),
        cat_encoded_low_df.reset_index(drop=True),
        cat_encoded_high_df.reset_index(drop=True)
    ], axis=1)
    
    # 7. ALIGNEMENT DES COLONNES (CRITIQUE pour éviter "Feature names unseen at fit time")
    if hasattr(scaler_x, 'feature_names_in_'):
        # On garde uniquement les colonnes attendues par scaler_x, dans le bon ordre
        # Si des colonnes manquent (ex: une catégorie absente du one-hot), on remplit par 0
        X_transformed = X_transformed.reindex(columns=scaler_x.feature_names_in_, fill_value=0)
    
    # 8. Transform
    X_scaled = scaler_x.transform(X_transformed)
    return X_scaled


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Endpoint racine pour vérifier que l'API fonctionne"""
    return {
        "message": "API de prédiction des coûts touristiques en Tanzanie",
        "status": "online",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Vérifie que tous les modèles sont chargés"""
    models_loaded = all([
        model is not None,
        scaler_y is not None,
        scaler_x is not None,
        encoder is not None
    ])
    
    warnings = []
    if model is None:
        warnings.append("model manquant")
    if scaler_y is None:
        warnings.append("scaler_y manquant")
    if scaler_x is None:
        warnings.append("scaler_x manquant")
    if encoder is None:
        warnings.append("encoder manquant")
    
    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded,
        "warnings": warnings if warnings else None
    }


@app.post("/predict")
async def predict(data: TourismInput):
    """
    Endpoint de prédiction
    
    Args:
        data: Données d'entrée validées par Pydantic
        
    Returns:
        Dictionnaire avec la prédiction du coût en FCFA
    """
    try:
        # Vérifier que les modèles sont chargés
        if model is None or scaler_y is None or scaler_x is None or encoder is None:
            raise HTTPException(
                status_code=503,
                detail="Les modèles ne sont pas encore chargés. Veuillez réessayer."
            )
        
        # 1. Convertir en DataFrame
        df = pd.DataFrame([data.dict()])
        
        # 2. Feature Engineering
        df_features = create_features(df)
        
        # 3. Preprocessing complet
        X_scaled = preprocess_data(df_features)
        
        # 4. Prédiction (en échelle log scalée)
        y_pred_scaled = model.predict(X_scaled)
        
        # 5. Inverse transform du scaler Y
        y_pred_log = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # 6. Inverse transform du log
        y_pred_fcfa = np.expm1(y_pred_log)
        
        return {
            "prediction_fcfa": float(y_pred_fcfa[0]),
            "prediction_formatted": f"{y_pred_fcfa[0]:,.0f} FCFA"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)