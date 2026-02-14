import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from statsmodels.stats.diagnostic import linear_harvey_collier, linear_rainbow, linear_reset
import warnings
warnings.filterwarnings('ignore')

class TestLineariteOptimal:
    """
    Test de linéarité complet : Analyse visuelle + Tests statistiques
    Version corrigée pour données transformées (log1p + RobustScaler)
    """
    
    def __init__(self, X, y, scaler=None, log_transformed=True):
        """
        Initialisation avec préparation des modèles
        
        Parameters:
        -----------
        X : DataFrame/array
            Features
        y : Series/array
            Target variable (dans l'espace transformé/normalisé)
        scaler : RobustScaler or None
            Le scaler utilisé pour normaliser y (si None, pas de scaling)
        log_transformed : bool
            Si True, y a été transformé avec log1p avant scaling
        """
        # Conversion en DataFrame/Series
        self.X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        self.y = pd.Series(y) if not isinstance(y, pd.Series) else y.copy()
        
        # ⚠️ CORRECTION : Reset des indices pour éviter désalignement
        # (important si y a été transformé avec .values puis .flatten())
        self.X = self.X.reset_index(drop=True)
        self.y = self.y.reset_index(drop=True)
        
        # Sauvegarde des infos de transformation
        self.scaler = scaler
        self.log_transformed = log_transformed
        
        # Statistiques de base
        self.n_samples = len(self.y)
        self.n_features = self.X.shape[1]
        
        # Modèle scikit-learn
        self.model_sk = LinearRegression()
        self.model_sk.fit(self.X, self.y)
        self.y_pred = self.model_sk.predict(self.X)
        
        # ============================================
        # RÉSIDUS DANS L'ESPACE TRANSFORMÉ (pour tests)
        # ============================================
        self.residuals_transformed = self.y - self.y_pred
        
        # ============================================
        # INVERSE TRANSFORMATION pour analyse réelle
        # ============================================
        self.y_pred_original = self._inverse_transform(self.y_pred)
        self.y_original = self._inverse_transform(self.y.values)
        self.residuals_original = self.y_original - self.y_pred_original
        
        # Modèle Statsmodels (dans l'espace transformé pour tests)
        X_with_const = sm.add_constant(self.X)
        self.model_sm = sm.OLS(self.y, X_with_const).fit()
        
        # Métriques dans l'espace TRANSFORMÉ (pour tests statistiques)
        self.r2 = self.model_sm.rsquared
        self.adj_r2 = self.model_sm.rsquared_adj
        
        # Métriques dans l'espace ORIGINAL (pour interprétation)
        self.rmse_original = np.sqrt(np.mean(self.residuals_original**2))
        self.mae_original = np.mean(np.abs(self.residuals_original))
        
        # MAE transformé pour compatibilité
        self.rmse_transformed = np.sqrt(np.mean(self.residuals_transformed**2))
        self.mae_transformed = np.mean(np.abs(self.residuals_transformed))

    def _inverse_transform(self, y_transformed):
        """
        Inverse la transformation : RobustScaler puis expm1
        
        Parameters:
        -----------
        y_transformed : array
            Valeurs dans l'espace transformé
        
        Returns:
        --------
        array : Valeurs dans l'espace original
        """
        y = y_transformed.copy()
        
        # Étape 1 : Inverse RobustScaler si applicable
        if self.scaler is not None:
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            y = self.scaler.inverse_transform(y).flatten()
        
        # Étape 2 : Inverse log1p (expm1) si applicable
        if self.log_transformed:
            y = np.expm1(y)
        
        return y

    def etape_1_analyse_visuelle(self):
        """Analyse visuelle complète avec diagnostics"""
        print("="*80)
        print("👁️  ÉTAPE 1 : ANALYSE VISUELLE")
        print("="*80)
        print(f"📊 Dataset : {self.n_samples} observations, {self.n_features} features")
        print(f"📈 Métriques (espace transformé) : R² = {self.r2:.4f} | R² ajusté = {self.adj_r2:.4f}")
        print(f"📏 Erreurs (échelle ORIGINALE)   : RMSE = {self.rmse_original:,.2f} FCFA | MAE = {self.mae_original:,.2f} FCFA")
        print(f"📏 Erreurs (espace transformé)   : RMSE = {self.rmse_transformed:.4f} | MAE = {self.mae_transformed:.4f}")
        print()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # ========================================
        # GRAPHIQUE 1 : Résidus vs Prédictions (ÉCHELLE ORIGINALE)
        # ========================================
        ax1 = axes[0, 0]
        ax1.scatter(self.y_pred_original, self.residuals_original, alpha=0.5, s=30, 
                   edgecolors='black', linewidth=0.5, c='steelblue')
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
        
        # Ligne de tendance avec gestion d'erreur
        try:
            idx_sorted = np.argsort(self.y_pred_original)
            y_sorted = self.y_pred_original[idx_sorted]
            r_sorted = self.residuals_original[idx_sorted]
            
            window = min(51, max(11, len(y_sorted) // 5))
            if window % 2 == 0:
                window += 1
            
            if len(y_sorted) >= window:
                smooth = savgol_filter(r_sorted, window, 3)
                ax1.plot(y_sorted, smooth, 'lime', linewidth=3, label='Tendance', alpha=0.8)
                
                # Calcul de courbure
                courbure = np.std(smooth)
                dispersion_totale = np.std(self.residuals_original)
                ratio_courbure = courbure / dispersion_totale
                
                # Seuil adaptatif selon R²
                seuil = 0.15 if self.r2 > 0.7 else 0.10
                is_linear_visuel = ratio_courbure < seuil
            else:
                is_linear_visuel = True
                ratio_courbure = 0
                
        except Exception as e:
            print(f"⚠️ Erreur calcul tendance : {e}")
            is_linear_visuel = True
            ratio_courbure = 0
        
        # Annotation
        status = "✅ ALÉATOIRE" if is_linear_visuel else "⚠️ PATTERN DÉTECTÉ"
        color = "lightgreen" if is_linear_visuel else "lightcoral"
        
        ax1.text(0.05, 0.95, f"{status}\nCourbure: {ratio_courbure:.2%}", 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(facecolor=color, alpha=0.8, boxstyle='round'),
                fontsize=10, fontweight='bold')
        
        ax1.set_title('1. RÉSIDUS vs PRÉDICTIONS (Échelle Originale FCFA)\n(Doit être aléatoire)', 
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('Valeurs prédites (Ŷ) [FCFA]', fontsize=11)
        ax1.set_ylabel('Résidus (Y - Ŷ) [FCFA]', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Formatter les axes en notation scientifique ou milliers
        ax1.ticklabel_format(style='plain', axis='both')
        
        # ========================================
        # GRAPHIQUE 2 : Réel vs Prédit (ÉCHELLE ORIGINALE)
        # ========================================
        ax2 = axes[0, 1]
        ax2.scatter(self.y_original, self.y_pred_original, alpha=0.5, s=30, 
                   edgecolors='black', linewidth=0.5, c='coral')
        
        lims = [min(self.y_original.min(), self.y_pred_original.min()), 
                max(self.y_original.max(), self.y_pred_original.max())]
        ax2.plot(lims, lims, 'r--', linewidth=2, label='Prédiction parfaite')
        
        ax2.text(0.05, 0.95, f'R² = {self.r2:.4f}\nMAE = {self.mae_original:,.0f} FCFA', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(facecolor='lightyellow', alpha=0.8, boxstyle='round'),
                fontsize=10, fontweight='bold')
        
        ax2.set_title('2. RÉEL vs PRÉDIT (Échelle Originale FCFA)\n(Points sur la ligne)', 
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('Valeurs réelles (Y) [FCFA]', fontsize=11)
        ax2.set_ylabel('Valeurs prédites (Ŷ) [FCFA]', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.ticklabel_format(style='plain', axis='both')
        
        # ========================================
        # GRAPHIQUE 3 : Distribution des résidus (ESPACE TRANSFORMÉ pour normalité)
        # ========================================
        ax3 = axes[1, 0]
        sns.histplot(self.residuals_transformed, kde=True, ax=ax3, color='steelblue', bins=30)
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
        
        # Test de normalité visuel
        mean_res = np.mean(self.residuals_transformed)
        std_res = np.std(self.residuals_transformed)
        
        ax3.text(0.05, 0.95, 
                f'Moyenne: {mean_res:.4f}\nÉcart-type: {std_res:.4f}\n(espace transformé)', 
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(facecolor='lightblue', alpha=0.8, boxstyle='round'),
                fontsize=9)
        
        ax3.set_title('3. DISTRIBUTION DES RÉSIDUS (Espace Transformé)\n(Doit être en cloche)', 
                     fontsize=12, fontweight='bold')
        ax3.set_xlabel('Résidus (transformés)', fontsize=11)
        ax3.set_ylabel('Fréquence', fontsize=11)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # ========================================
        # GRAPHIQUE 4 : Q-Q Plot (ESPACE TRANSFORMÉ)
        # ========================================
        ax4 = axes[1, 1]
        stats.probplot(self.residuals_transformed, dist="norm", plot=ax4)
        ax4.set_title('4. Q-Q PLOT (Espace Transformé)\n(Normalité des résidus)', 
                     fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('diagnostic_linearite.png', dpi=150, bbox_inches='tight')
        print("💾 Graphiques sauvegardés : diagnostic_linearite.png")
        plt.show()
        
        print("\n" + "─"*80)
        print("💡 INTERPRÉTATION VISUELLE :")
        print("─"*80)
        if is_linear_visuel:
            print("✅ Les résidus semblent aléatoires (bon pour linéarité)")
        else:
            print("⚠️ Un pattern est visible dans les résidus (possible non-linéarité)")
        print()
        
        return is_linear_visuel

    def etape_2_tests_formels(self):
        """Tests statistiques avec diagnostics détaillés"""
        print("="*80)
        print("📊 ÉTAPE 2 : TESTS STATISTIQUES FORMELS")
        print("="*80)
        print("ℹ️  Les tests sont effectués dans l'ESPACE TRANSFORMÉ (log1p + scaled)")
        print("   car les hypothèses de régression linéaire s'appliquent aux résidus transformés")
        print()
        
        resultats = {}
        details = {}
        
        # ========================================
        # TEST 1 : Harvey-Collier
        # ========================================
        print("━"*80)
        print("📐 TEST 1 : HARVEY-COLLIER (Linéarité directe)")
        print("━"*80)
        
        hc_success = False
        for skip in [3, 10, 20, 50, 100]:
            try:
                stat, p_val = linear_harvey_collier(self.model_sm, skip=skip)
                hc_success = True
                resultats['harvey_collier'] = p_val > 0.05
                details['harvey_collier'] = {
                    'stat': stat,
                    'p_value': p_val,
                    'skip': skip
                }
                
                print(f"✅ Test réussi (skip={skip})")
                print(f"   Statistique : {stat:.4f}")
                print(f"   P-value     : {p_val:.6f}")
                
                if p_val > 0.05:
                    print(f"   ✅ Conclusion : LINÉARITÉ confirmée (p > 0.05)")
                else:
                    print(f"   ❌ Conclusion : NON-LINÉARITÉ détectée (p < 0.05)")
                
                break
                
            except Exception as e:
                continue
        
        if not hc_success:
            print("❌ Test impossible à exécuter")
            print("   Raison : Multicollinéarité forte")
            print("   💡 Solution : Supprimer les variables corrélées > 0.95")
            resultats['harvey_collier'] = None
            details['harvey_collier'] = {'error': 'Multicollinéarité'}
        
        print()
        
        # ========================================
        # TEST 2 : Rainbow
        # ========================================
        print("━"*80)
        print("📐 TEST 2 : RAINBOW (Ajustement uniforme)")
        print("━"*80)
        
        try:
            stat, p_val = linear_rainbow(self.model_sm)
            resultats['rainbow'] = p_val > 0.05
            details['rainbow'] = {
                'stat': stat,
                'p_value': p_val
            }
            
            print(f"   Statistique F : {stat:.4f}")
            print(f"   P-value       : {p_val:.6f}")
            
            if p_val > 0.05:
                print(f"   ✅ Conclusion : LINÉARITÉ confirmée (p > 0.05)")
            else:
                print(f"   ❌ Conclusion : NON-LINÉARITÉ détectée (p < 0.05)")
            
        except Exception as e:
            print(f"❌ Erreur : {e}")
            resultats['rainbow'] = None
            details['rainbow'] = {'error': str(e)}
        
        print()
        
        # ========================================
        # TEST 3 : RESET de Ramsey
        # ========================================
        print("━"*80)
        print("📐 TEST 3 : RESET DE RAMSEY (Termes polynomiaux)")
        print("━"*80)
        
        try:
            reset_res = linear_reset(self.model_sm, power=3, use_f=True)
            p_val = reset_res.pvalue
            f_stat = reset_res.fvalue if hasattr(reset_res, 'fvalue') else reset_res.statistic
            
            resultats['reset'] = p_val > 0.05
            details['reset'] = {
                'stat': f_stat,
                'p_value': p_val
            }
            
            print(f"   Statistique F : {f_stat:.4f}")
            print(f"   P-value       : {p_val:.6f}")
            
            if p_val > 0.05:
                print(f"   ✅ Conclusion : X², X³ n'améliorent PAS (linéarité OK)")
            else:
                print(f"   ❌ Conclusion : X², X³ AMÉLIORENT (non-linéarité)")
                print(f"   💡 Suggestion : Essayer PolynomialFeatures(degree=2)")
            
        except Exception as e:
            print(f"❌ Erreur : {e}")
            resultats['reset'] = None
            details['reset'] = {'error': str(e)}
        
        print()
        
        return resultats, details

    def synthese_finale(self, visuel_ok, tests_res, details):
        """Synthèse avec recommandations actionnables"""
        print("="*80)
        print("🎯 SYNTHÈSE FINALE")
        print("="*80)
        print()
        
        # Compter les tests
        tests_valides = {k: v for k, v in tests_res.items() if v is not None}
        tests_passes = sum(1 for v in tests_valides.values() if v is True)
        total_tests = len(tests_valides)
        
        # Affichage détaillé
        print("📊 Résumé des tests :")
        print("─"*80)
        print(f"   Analyse visuelle : {'✅ Linéaire' if visuel_ok else '❌ Non-linéaire'}")
        
        for test_name, result in tests_res.items():
            nom = test_name.replace('_', ' ').title()
            if result is None:
                icon = "⚠️"
                status = "Non exécuté"
            elif result:
                icon = "✅"
                status = "Linéaire"
            else:
                icon = "❌"
                status = "Non-linéaire"
            
            p_val = details.get(test_name, {}).get('p_value', None)
            if p_val is not None:
                print(f"   {icon} {nom:20} : {status:15} (p = {p_val:.6f})")
            else:
                print(f"   {icon} {nom:20} : {status}")
        
        print()
        print("─"*80)
        
        # Score de linéarité
        if total_tests > 0:
            score_tests = (tests_passes / total_tests) * 100
            score_global = (score_tests * 0.6 + (100 if visuel_ok else 0) * 0.4)
        else:
            score_global = 100 if visuel_ok else 0
        
        print(f"📈 Score de linéarité : {score_global:.1f}%")
        print()
        
        # VERDICT
        print("="*80)
        print("💡 VERDICT ET RECOMMANDATIONS")
        print("="*80)
        
        if score_global >= 75:
            print("✅ RELATION FORTEMENT LINÉAIRE (dans l'espace transformé)")
            print()
            print("📌 Recommandations :")
            print("   1️⃣  La transformation log1p + RobustScaler est APPROPRIÉE")
            print("   2️⃣  Utiliser Ridge/Lasso sur les données transformées")
            print("   3️⃣  Ne pas oublier l'inverse_transform pour les prédictions finales")
            
        elif score_global >= 40:
            print("🟡 RELATION PARTIELLEMENT LINÉAIRE")
            print()
            print("📌 Recommandations :")
            print("   1️⃣  La transformation aide mais ne suffit pas complètement")
            print("   2️⃣  Essayer des modèles non-linéaires (XGBoost, Random Forest)")
            print("   3️⃣  Les modèles ensemblistes sont recommandés")
            
        else:
            print("⚠️ RELATION FORTEMENT NON-LINÉAIRE")
            print()
            print("📌 Recommandations :")
            print("   1️⃣  Privilégier XGBoost, Random Forest ou Gradient Boosting")
            print("   2️⃣  La transformation log1p aide mais modèle linéaire insuffisant")
            print("   3️⃣  Pas besoin d'inverse_transform avec les modèles non-linéaires")
        
        print()
        print("="*80)

    def tester_linearite(self):
        """Méthode principale orchestrant tous les tests"""
        print("\n")
        print("╔" + "="*78 + "╗")
        print("║" + " "*15 + "TEST DE LINÉARITÉ (avec transformations)" + " "*22 + "║")
        print("╚" + "="*78 + "╝")
        print()
        
        if self.log_transformed and self.scaler is not None:
            print("ℹ️  Données transformées : log1p + RobustScaler")
        elif self.log_transformed:
            print("ℹ️  Données transformées : log1p uniquement")
        elif self.scaler is not None:
            print("ℹ️  Données transformées : RobustScaler uniquement")
        else:
            print("ℹ️  Données non transformées")
        print()
        
        # Étape 1 : Visuel
        visuel_ok = self.etape_1_analyse_visuelle()
        
        # Étape 2 : Tests formels
        tests_res, details = self.etape_2_tests_formels()
        
        # Synthèse
        self.synthese_finale(visuel_ok, tests_res, details)
        
        return {
            'visual_linear': visuel_ok,
            'formal_tests': tests_res,
            'details': details,
            'r2': self.r2,
            'adj_r2': self.adj_r2,
            'mae_original': self.mae_original,
            'rmse_original': self.rmse_original
        }


# ========================================
# FONCTION UTILITAIRE
# ========================================

def supprimer_colonnes_correlees(X, seuil=0.95, verbose=True):
    """
    Supprime les colonnes fortement corrélées
    
    Parameters:
    -----------
    X : DataFrame
        Les features
    seuil : float
        Seuil de corrélation (0.95 par défaut)
    verbose : bool
        Afficher les détails
    
    Returns:
    --------
    DataFrame nettoyé
    """
    X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
    
    # Matrice de corrélation
    corr_matrix = X_df.corr().abs()
    
    # Triangle supérieur
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Colonnes à supprimer
    to_drop = [column for column in upper.columns if any(upper[column] > seuil)]
    
    if verbose and len(to_drop) > 0:
        print(f"🔍 Multicollinéarité détectée :")
        print(f"   {len(to_drop)} colonnes corrélées > {seuil}")
        print(f"   Colonnes supprimées : {to_drop[:5]}{'...' if len(to_drop) > 5 else ''}")
        print()
    
    X_clean = X_df.drop(columns=to_drop)
    
    if verbose:
        print(f"✅ Nettoyage terminé :")
        print(f"   Avant : {X_df.shape[1]} colonnes")
        print(f"   Après : {X_clean.shape[1]} colonnes")
        print()
    
    return X_clean


# ========================================
# EXEMPLE D'UTILISATION
# ========================================

if __name__ == "__main__":
    """
    IMPORTANT : Adapter selon votre pipeline de transformation !
    
    Exemple si vous avez :
    1. Appliqué log1p sur total_cost
    2. Utilisé RobustScaler pour normaliser
    """
    
    # Exemple de code d'utilisation (à adapter à votre cas)
    print("""
EXEMPLE D'UTILISATION :
=======================

# 1. Préparer vos données
X_combined = pd.concat([cat_final, num_data], axis=1)
X_combined = X_combined.drop('total_cost', axis=1).dropna()
y_original = num_data['total_cost'].loc[X_combined.index]

# 2. Appliquer les transformations (comme dans votre notebook)
from sklearn.preprocessing import RobustScaler

# Transformation log1p
y_log = np.log1p(y_original)

# Normalisation RobustScaler
scaler = RobustScaler()
y_scaled = scaler.fit_transform(y_log.values.reshape(-1, 1)).flatten()

# 3. Nettoyage features (optionnel)
X_clean = supprimer_colonnes_correlees(X_combined, seuil=0.95)

# 4. Test de linéarité avec transformations
tester = TestLineariteOptimal(
    X=X_clean, 
    y=y_scaled,           # Données transformées
    scaler=scaler,        # Le scaler utilisé
    log_transformed=True  # On a utilisé log1p
)

resultats = tester.tester_linearite()

# 5. Résultats
print(f"\\nR² = {resultats['r2']:.4f}")
print(f"R² ajusté = {resultats['adj_r2']:.4f}")
print(f"MAE (échelle originale) = {resultats['mae_original']:,.2f} FCFA")
    """)
