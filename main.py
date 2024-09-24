import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# Charger les données
data = pd.read_csv('toulouse_filtered_data.csv')

# Préparation des données
X = data[['Surface reelle bati', 'Nombre pieces principales', 'Surface terrain']]
y = pd.to_numeric(data['Valeur fonciere'].str.replace(',', '.'), errors='coerce')

# Imputation des valeurs manquantes
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Création d'une nouvelle caractéristique
data['Surface_per_Piece'] = data['Surface reelle bati'] / data['Nombre pieces principales']
X_imputed = pd.DataFrame(X_imputed, columns=['Surface reelle bati', 'Nombre pieces principales', 'Surface terrain'])
X_imputed['Surface_per_Piece'] = data['Surface_per_Piece']

# Remplacement des valeurs infinies
X_imputed.replace([np.inf, -np.inf], np.nan, inplace=True)
X_imputed.fillna(X_imputed.mean(), inplace=True)  # Remplacez les NaN par la moyenne

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Définition des modèles
rf_model = RandomForestRegressor(random_state=42)
gb_model = GradientBoostingRegressor(random_state=42)
lr_model = LinearRegression()

# Paramètres pour RandomizedSearchCV pour RandomForest
rf_param_distributions = {
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}

# Recherche d'hyperparamètres pour RandomForest
rf_random_search = RandomizedSearchCV(rf_model, rf_param_distributions, n_iter=20, cv=5, random_state=42, n_jobs=-1)
rf_random_search.fit(X_train, y_train)

# Évaluation du modèle RandomForest
rf_predictions = rf_random_search.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_predictions)

print(f'RandomForest - Mean Absolute Error: {rf_mae}')

# Sauvegarde du modèle RandomForest
joblib.dump(rf_random_search, 'modele_prediction_biens_immobiliers_RandomForest.joblib')

# Recherche d'hyperparamètres pour GradientBoosting
gb_param_distributions = {
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'learning_rate': [0.01, 0.1, 0.2, 0.5]
}

gb_random_search = RandomizedSearchCV(gb_model, gb_param_distributions, n_iter=20, cv=5, random_state=42, n_jobs=-1)
gb_random_search.fit(X_train, y_train)

# Évaluation du modèle GradientBoosting
gb_predictions = gb_random_search.predict(X_test)
gb_mae = mean_absolute_error(y_test, gb_predictions)

print(f'GradientBoosting - Mean Absolute Error: {gb_mae}')

# Sauvegarde du modèle GradientBoosting
joblib.dump(gb_random_search, 'modele_prediction_biens_immobiliers_GradientBoosting.joblib')

# Évaluation du modèle de Régression Linéaire
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_predictions)

print(f'LinearRegression - Mean Absolute Error: {lr_mae}')

# Sauvegarde du modèle de Régression Linéaire
joblib.dump(lr_model, 'modele_prediction_biens_immobiliers_LinearRegression.joblib')

# Sauvegarde des résultats de prédictions
predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted_RF': rf_predictions,
    'Predicted_GB': gb_predictions,
    'Predicted_LR': lr_predictions
})

predictions_df.to_csv('predictions.csv', index=False)
print("Les résultats de prédiction ont été sauvegardés dans 'predictions.csv'.")
