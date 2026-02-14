# API de Prédiction des Coûts Touristiques en Tanzanie

## 📋 Description

Cette API FastAPI permet de prédire le coût total d'un séjour touristique en Tanzanie basé sur diverses caractéristiques du voyage.

## 🚀 Installation

```bash
cd tourisme_ml
pip install -r requirements.txt
```

## ▶️ Lancement de l'API

```bash
# Depuis le dossier tourisme_ml
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

L'API sera accessible sur: `http://localhost:8000`

## 📚 Documentation

- **Docs interactives (Swagger)**: http://localhost:8000/docs
- **Docs alternatives (ReDoc)**: http://localhost:8000/redoc

## 🔍 Endpoints

### `GET /`
Point d'entrée racine pour vérifier que l'API fonctionne.

**Réponse:**
```json
{
  "message": "API de prédiction des coûts touristiques en Tanzanie",
  "status": "online",
  "version": "1.0.0"
}
```

### `GET /health`
Vérifie que tous les modèles sont chargés correctement.

**Réponse:**
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

### `POST /predict`
Effectue une prédiction du coût total du séjour.

**Corps de la requête (JSON):**
```json
{
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
```

**Réponse:**
```json
{
  "prediction_fcfa": 12500000.5,
  "prediction_formatted": "12,500,001 FCFA"
}
```

## 🧪 Test de l'API

Utilisez le script `test_api.py`:

```bash
python test_api.py
```

Ou avec curl:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @example_request.json
```

## 📦 Fichiers Requis

L'API nécessite les fichiers suivants dans le dossier `tourisme_ml/`:
- `model.pkl` - Modèle XGBoost entraîné
- `scaler.pkl` - RobustScaler pour les features (scaler_x)
- `scaler_y.pkl` - RobustScaler pour la target
- `encoder.pkl` - OneHotEncoder pour les variables catégorielles

## 🔧 Architecture

```
tourisme_ml/
├── app/
│   ├── __init__.py
│   └── main.py          # Code principal de l'API
├── model.pkl            # Modèle
├── scaler.pkl           # Scaler features
├── scaler_y.pkl         # Scaler target
├── encoder.pkl          # Encoder catégoriel
├── requirements.txt     # Dépendances
├── README.md           # Documentation
└── test_api.py         # Script de test
```

## 🛠️ Améliorations Apportées

### 1. **Chargement Correct des Artifacts**
- Tous les fichiers `.pkl` sont chargés au démarrage
- Inclusion de `scaler_y.pkl` (manquant dans la version originale)
- Gestion des erreurs de chargement

### 2. **Pipeline de Preprocessing Complet**
- Feature engineering identique au training
- Séparation correcte num/cat
- One-Hot Encoding avec `transform()` (pas `fit_transform()`)
- Log transformation sur numériques
- Scaling avec `scaler_x.transform()` (pas `fit_transform()`)

### 3. **Inférence Correcte**
- Prédiction sur données scalées
- Inverse transform du scaler Y: `scaler_y.inverse_transform()`
- Inverse transform du log: `np.expm1()`
- Retour du coût en FCFA

### 4. **Validation et Sécurité**
- Schéma Pydantic pour valider les inputs
- Gestion des erreurs HTTP
- Health checks

### 5. **Code Propre**
- Suppression du code au niveau module
- Fonctions bien séparées
- Documentation complète
- Type hints

## ⚠️ Points d'Attention

1. **Frequency Encoding**: Dans cette version, les pays inconnus reçoivent une fréquence par défaut (0.01). Pour une meilleure robustesse, il faudrait sauvegarder un `freq_maps.pkl` lors du training.

2. **Colonnes**: L'ordre et le nombre de colonnes après preprocessing doivent correspondre exactement au training. Si vous modifiez le preprocessing dans le notebook, mettez à jour l'API.

3. **Versions**: Assurez-vous d'utiliser les mêmes versions de scikit-learn et XGBoost que lors du training.

## 📞 Support

Pour toute question ou problème, vérifiez:
1. Que tous les fichiers `.pkl` sont présents
2. Les logs de l'API au démarrage
3. L'endpoint `/health` retourne `healthy`
