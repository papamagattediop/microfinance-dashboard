# 🏦 Microfinance Credit Risk Dashboard

Dashboard interactif d'analyse et de prédiction du risque de crédit pour les institutions de microfinance.

---

## 🎯 Description

Application Dash permettant de :
- **Analyser** les facteurs de risque de crédit avec des visualisations interactives
- **Prédire** le risque de défaut avec deux modèles : LDA et QDA
- **Décider** automatiquement de l'octroi ou du refus d'un crédit

---

## 🛠️ Technologies

- Python 3.8+ | Dash | Plotly | Scikit-learn | Pandas

---

## 📂 Structure
```
microfinance-dashboard/
├── app.py                    # Application principale
├── assets/style.css          # Styles personnalisés
├── data/                     # Données Excel
├── models/                   # Modèles entraînés (généré)
├── pages/
│   ├── page1_analyse.py      # Page exploration
│   └── page2_model.py        # Page modélisation
└── utils/
    ├── data_loader.py        # Chargement données
    └── model_trainer.py      # Entraînement modèles
```

---

## 🚀 Installation & Lancement

### 1. Installation
```bash
# Créer environnement virtuel
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Installer dépendances
pip install pandas numpy plotly dash scikit-learn openpyxl dash-bootstrap-components joblib
```

### 2. Entraîner les modèles
```bash
python utils/model_trainer.py
```

### 3. Lancer le dashboard
```bash
python app.py
```

Ouvrir : **http://127.0.0.1:8050**

---

## 📊 Fonctionnalités

### Page 1 : Analyse & Exploration
- KPIs dynamiques (taux de défaut, clients à risque)
- Filtres interactifs (région, secteur, montant)
- Visualisations : Histogramme DSTI, Scatter plot, Corrélations
- Table de données filtrable

### Page 2 : Modélisation & Prédiction
- Comparaison LDA vs QDA (métriques, matrices de confusion, courbes ROC)
- Formulaire de prédiction client
- Décision automatique d'octroi de crédit

---

## 🎨 Design

Palette **verte forêt** professionnelle :
- Vert forêt : `#013328`
- Bois : `#CC8B65`
- Beige : `#E3DCD2`

Interface responsive et moderne.

---

## 👨‍💻 Auteur

**[Papa Magatte DIOP]**  
Licence 3 - Data Visualisation | 2024-2025

---

## 📄 Licence

Projet académique - Usage libre

---

**Bon scoring ! 🎯**