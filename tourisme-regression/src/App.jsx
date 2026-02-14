import { useState } from 'react'
import { TextField, Button, Container, Typography, Box, Paper, Grid } from '@mui/material'
import './App.css'

function App() {
  // ============================================
  // 1️⃣ État initial avec toutes les variables
  // ============================================
  const [formData, setFormData] = useState({
    country: '',
    age_group: '',
    travel_with: '',
    total_female: 0,
    total_male: 0,
    purpose: '',
    main_activity: '',
    info_source: '',
    tour_arrangement: '',
    package_transport_int: '',
    package_accomodation: '',
    package_food: '',
    package_transport_tz: '',
    package_sightseeing: '',
    package_guided_tour: '',
    package_insurance: '',
    night_mainland: 0,
    night_zanzibar: 0,
    payment_mode: '',
    first_trip_tz: '',
    most_impressing: ''
  });

  const [prix, setPrix] = useState(null);
  const [loading, setLoading] = useState(false);

  // ============================================
  // 2️⃣ Gestion des changements d'inputs
  // ============================================
  const handleChange = (e) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      // Conversion automatique si c'est un champ nombre
      [name]: type === 'number' ? (value === '' ? '' : parseInt(value)) : value
    }));
  };

  // ============================================
  // 3️⃣ Configuration des champs pour le rendu
  // ============================================
  const formFields = [
    { name: 'country', label: 'Pays', type: 'text' },
    { name: 'age_group', label: 'Tranche d\'âge (ex: 25-44)', type: 'text' },
    { name: 'travel_with', label: 'Voyage avec (ex: Spouse)', type: 'text' },
    { name: 'total_female', label: 'Nombre de femmes', type: 'number' },
    { name: 'total_male', label: 'Nombre d\'hommes', type: 'number' },
    { name: 'purpose', label: 'Objectif du voyage', type: 'text' },
    { name: 'main_activity', label: 'Activité principale', type: 'text' },
    { name: 'info_source', label: 'Source d\'information', type: 'text' },
    { name: 'tour_arrangement', label: 'Organisation du tour', type: 'text' },
    { name: 'package_transport_int', label: 'Transport international (Yes/No)', type: 'text' },
    { name: 'package_accomodation', label: 'Hébergement (Yes/No)', type: 'text' },
    { name: 'package_food', label: 'Nourriture (Yes/No)', type: 'text' },
    { name: 'package_transport_tz', label: 'Transport TZ (Yes/No)', type: 'text' },
    { name: 'package_sightseeing', label: 'Visites touristiques (Yes/No)', type: 'text' },
    { name: 'package_guided_tour', label: 'Tour guidé (Yes/No)', type: 'text' },
    { name: 'package_insurance', label: 'Assurance (Yes/No)', type: 'text' },
    { name: 'night_mainland', label: 'Nuits continent', type: 'number' },
    { name: 'night_zanzibar', label: 'Nuits Zanzibar', type: 'number' },
    { name: 'payment_mode', label: 'Mode de paiement', type: 'text' },
    { name: 'first_trip_tz', label: 'Premier voyage TZ (Yes/No)', type: 'text' },
    { name: 'most_impressing', label: 'Plus impressionnant', type: 'text' }
  ];

  // ============================================
  // 4️⃣ Appel à l'API FastAPI
  // ============================================
  const devinerPrix = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });

      if (!response.ok) {
        const errorData = await response.json();
        console.error("Détails erreur API:", errorData);
        throw new Error("Erreur lors de la prédiction");
      }

      const data = await response.json();
      // Utilisation de la clé exacte de ton code Python
      setPrix(data.prediction_formatted);
    } catch (error) {
      console.error("Erreur lors de l'appel API:", error);
      alert("Erreur: Vérifiez que votre API Python tourne sur le port 8000");
    } finally {
      setLoading(false);
    }
  };

  // ============================================
  // 5️⃣ Rendu
  // ============================================
  return (
    <Container maxWidth="md" sx={{ py: 5 }}>
      <Paper elevation={3} sx={{ p: 4, borderRadius: 4 }}>
        <Typography variant="h4" gutterBottom align="center" sx={{ fontWeight: 'bold', color: '#1976d2' }}>
          🌍 Tanzanie Travel Predictor
        </Typography>
        <Typography variant="body1" align="center" color="textSecondary" sx={{ mb: 4 }}>
          Remplissez les détails du voyage pour estimer le coût total.
        </Typography>

        <Grid container spacing={2}>
          {formFields.map(field => (
            <Grid item xs={12} sm={6} key={field.name}>
              <TextField
                name={field.name}
                label={field.label}
                type={field.type}
                value={formData[field.name]}
                onChange={handleChange}
                variant="outlined"
                fullWidth
                size="small"
                // Pour éviter que les nombres soient négatifs visuellement
                InputProps={field.type === 'number' ? { inputProps: { min: 0 } } : {}}
              />
            </Grid>
          ))}
        </Grid>

        <Box sx={{ mt: 4, textAlign: 'center' }}>
          <Button
            variant="contained"
            size="large"
            onClick={devinerPrix}
            disabled={loading}
            sx={{ px: 8, py: 1.5, borderRadius: 2, fontSize: '1.1rem' }}
          >
            {loading ? "Calcul en cours..." : "Estimer le prix"}
          </Button>

          {prix && (
            <Box sx={{ mt: 4, p: 3, bgcolor: '#e3f2fd', borderRadius: 2, border: '1px solid #90caf9' }}>
              <Typography variant="h6" color="textSecondary">
                Estimation du coût total :
              </Typography>
              <Typography variant="h3" color="primary" sx={{ fontWeight: 'bold' }}>
                {prix}
              </Typography>
            </Box>
          )}
        </Box>
      </Paper>
    </Container>
  )
}

export default App