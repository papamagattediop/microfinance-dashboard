import pandas as pd
import numpy as np

def load_data(filepath='data/microfinance_credit_risk.xlsx'):
    """Charge et prépare les données"""
    df = pd.read_excel(filepath)  
    return df

def get_basic_stats(df):
    """Statistiques de base"""
    stats = {
        'nb_clients': len(df),
        'taux_defaut_global': df['defaut_90j'].mean() * 100,
        'taux_defaut_par_region': df.groupby('region')['defaut_90j'].mean() * 100,
        'taux_defaut_par_secteur': df.groupby('secteur_activite')['defaut_90j'].mean() * 100,
        'taux_defaut_par_canal': df.groupby('canal_octroi')['defaut_90j'].mean() * 100
    }
    return stats

def get_numeric_columns(df):
    """Retourne les colonnes numériques"""
    return df.select_dtypes(include=[np.number]).columns.tolist()

def get_categorical_columns(df):
    """Retourne les colonnes catégorielles"""
    return df.select_dtypes(include=['object']).columns.tolist()