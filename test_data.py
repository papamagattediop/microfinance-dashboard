import pandas as pd
from utils.data_loader import load_data, get_basic_stats, get_numeric_columns

# Charger les données
df = load_data('data/microfinance_credit_risk.xlsx')

print("=" * 50)
print("APERÇU DES DONNÉES")
print("=" * 50)
print(df.head())
print("\n")

print("=" * 50)
print("INFORMATIONS GÉNÉRALES")
print("=" * 50)
print(df.info())
print("\n")

print("=" * 50)
print("STATISTIQUES DESCRIPTIVES")
print("=" * 50)
print(df.describe())
print("\n")

print("=" * 50)
print("VALEURS MANQUANTES")
print("=" * 50)
print(df.isnull().sum())
print("\n")

print("=" * 50)
print("DISTRIBUTION DE LA CIBLE (defaut_90j)")
print("=" * 50)
print(df['defaut_90j'].value_counts())
print(f"Taux de défaut : {df['defaut_90j'].mean() * 100:.2f}%")
print("\n")

print("=" * 50)
print("COLONNES NUMÉRIQUES")
print("=" * 50)
print(get_numeric_columns(df))
print("\n")

print("=" * 50)
print("COLONNES categorielles")
print("=" * 50)
print(df.select_dtypes(include=['object', 'category']).columns.tolist())
print("\n")

# Statistiques par région
print("=" * 50)
print("TAUX DE DÉFAUT PAR RÉGION")
print("=" * 50)
stats = get_basic_stats(df)
print(stats['taux_defaut_par_region'])

import pandas as pd

# Charger les données
df = pd.read_excel('data/microfinance_credit_risk.xlsx')

print("=" * 60)
print("DIAGNOSTIC DES DONNÉES")
print("=" * 60)

# Distribution de la cible
print("\nDistribution de la variable cible (defaut_90j) :")
print(df['defaut_90j'].value_counts())
print(f"\nPourcentages :")
print(df['defaut_90j'].value_counts(normalize=True) * 100)

# Ratio de déséquilibre
ratio = df['defaut_90j'].value_counts()[0] / df['defaut_90j'].value_counts()[1]
print(f"\nRatio de déséquilibre : {ratio:.2f}:1")

# Nombre total d'observations
print(f"\nNombre total de lignes : {len(df)}")