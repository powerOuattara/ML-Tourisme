"""
Script de test pour l'API de prédiction touristique
"""
import requests
import json

# URL de base de l'API
BASE_URL = "http://localhost:8000"

def test_root():
    """Test de l'endpoint racine"""
    print("🧪 Test de l'endpoint racine...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Réponse: {json.dumps(response.json(), indent=2)}\n")
    assert response.status_code == 200

def test_health():
    """Test du health check"""
    print("🧪 Test du health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Réponse: {json.dumps(response.json(), indent=2)}\n")
    assert response.status_code == 200
    assert response.json()["models_loaded"] == True

def test_prediction():
    """Test de prédiction avec des données d'exemple"""
    print("🧪 Test de prédiction...")
    
    # Données d'exemple
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
    
    print("📤 Envoi des données:")
    print(json.dumps(data, indent=2))
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    
    print(f"\n📥 Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Prédiction réussie!")
        print(f"   Coût prédit: {result['prediction_formatted']}")
        print(f"   Valeur brute: {result['prediction_fcfa']:.2f} FCFA")
    else:
        print(f"❌ Erreur: {response.json()}")
    
    assert response.status_code == 200
    print()

def test_multiple_scenarios():
    """Test avec plusieurs scénarios"""
    print("🧪 Test de plusieurs scénarios...")
    
    scenarios = [
        {
            "name": "Budget Backpacker",
            "data": {
                "country": "KENYA",
                "age_group": "18-24",
                "travel_with": "Friends",
                "total_female": 2,
                "total_male": 2,
                "purpose": "Leisure and Holidays",
                "main_activity": "Beach tourism",
                "info_source": "Social Media",
                "tour_arrangement": "Independent",
                "package_transport_int": "No",
                "package_accomodation": "No",
                "package_food": "No",
                "package_transport_tz": "No",
                "package_sightseeing": "No",
                "package_guided_tour": "No",
                "package_insurance": "No",
                "night_mainland": 2,
                "night_zanzibar": 3,
                "payment_mode": "Cash",
                "first_trip_tz": "Yes",
                "most_impressing": "Beach and Watersport"
            }
        },
        {
            "name": "Luxury Safari",
            "data": {
                "country": "UNITED STATES",
                "age_group": "45-64",
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
                "package_insurance": "Yes",
                "night_mainland": 10,
                "night_zanzibar": 5,
                "payment_mode": "Credit Card",
                "first_trip_tz": "No",
                "most_impressing": "Wildlife"
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📌 Scénario: {scenario['name']}")
        response = requests.post(f"{BASE_URL}/predict", json=scenario['data'])
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Coût prédit: {result['prediction_formatted']}")
        else:
            print(f"   ❌ Erreur: {response.status_code}")
    
    print()

def main():
    """Fonction principale"""
    print("=" * 60)
    print("🚀 TESTS DE L'API DE PRÉDICTION TOURISTIQUE")
    print("=" * 60)
    print()
    
    try:
        # Vérifier que l'API est accessible
        test_root()
        test_health()
        test_prediction()
        test_multiple_scenarios()
        
        print("=" * 60)
        print("✅ TOUS LES TESTS SONT PASSÉS!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("❌ Impossible de se connecter à l'API")
        print("   Assurez-vous que l'API est lancée avec:")
        print("   uvicorn app.main:app --reload")
    except AssertionError as e:
        print(f"❌ Test échoué: {e}")
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")

if __name__ == "__main__":
    main()
