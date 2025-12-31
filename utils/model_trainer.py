import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class CreditRiskModels:
    """
    Classe pour entraîner des modèles LDA et QDA optimisés
    avec gestion avancée du déséquilibre des classes
    """
    
    def __init__(self, data_path='data/microfinance_credit_risk.xlsx'):
        """Initialisation"""
        self.df = pd.read_excel(data_path)
        self.features = ['age', 'revenu_mensuel_xof', 'dsti_pct', 'montant_pret_xof',
                        'duree_mois', 'taux_interet_annuel_pct', 'jours_retard_12m',
                        'anciennete_relation_mois', 'epargne_xof']
        
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        
        self.lda_model = None
        self.qda_model = None
        self.best_threshold_lda = 0.5
        self.best_threshold_qda = 0.5
        
        self.lda_metrics = {}
        self.qda_metrics = {}
        
    def prepare_data(self, test_size=0.2, random_state=42):
        """Prépare les données"""
        print("=" * 70)
        print("PRÉPARATION DES DONNÉES")
        print("=" * 70)
        
        available_features = [f for f in self.features if f in self.df.columns]
        print(f"Features sélectionnées : {len(available_features)}\n")
        print(f"Features disponibles : {available_features}")
        
        self.X = self.df[available_features].copy()
        self.y = self.df['defaut_90j'].copy()
        
        # Gérer les valeurs manquantes
        self.X = self.X.fillna(self.X.median())
        
        # Analyser le déséquilibre
        class_counts = self.y.value_counts()
        print(f"Distribution des classes :")
        print(f"  Classe 0 (non défaut) : {class_counts[0]} ({class_counts[0]/len(self.y)*100:.1f}%)")
        print(f"  Classe 1 (défaut)     : {class_counts[1]} ({class_counts[1]/len(self.y)*100:.1f}%)")
        
        if len(class_counts) < 2:
            raise ValueError("❌ Une seule classe présente dans les données !")
        
        ratio = class_counts[0] / class_counts[1]
        print(f"  Ratio de déséquilibre : {ratio:.2f}:1\n")
        
        # Split stratifié
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        print(f"Ensemble d'entraînement : {len(self.X_train)} échantillons")
        print(f"Ensemble de test        : {len(self.X_test)} échantillons")
        
        # Vérifier la distribution dans le test
        test_counts = self.y_test.value_counts()
        print(f"\nDistribution dans le test :")
        print(f"  Classe 0 : {test_counts.get(0, 0)}")
        print(f"  Classe 1 : {test_counts.get(1, 0)}\n")
        
        # Standardisation
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("✓ Données préparées\n")
        
    def find_optimal_threshold(self, y_proba, y_true, model_name="Model"):
        """Trouve le seuil optimal pour maximiser F1"""
        print(f"Optimisation du seuil pour {model_name}...")
        
        best_f1 = 0
        best_threshold = 0.5
        
        thresholds = np.arange(0.05, 0.95, 0.05)
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        print(f"  Seuil optimal : {best_threshold:.2f} (F1 = {best_f1:.4f})\n")
        return best_threshold, best_f1
        
    def train_lda(self):
        """Entraîne LDA avec ajustement du seuil"""
        print("=" * 70)
        print("ENTRAÎNEMENT LDA")
        print("=" * 70)
        
        # Entraîner avec priors ajustés selon la distribution réelle
        class_counts = self.y_train.value_counts()
        priors = [class_counts[0]/len(self.y_train), class_counts[1]/len(self.y_train)]
        
        print(f"Priors utilisés : {priors}\n")
        
        self.lda_model = LinearDiscriminantAnalysis(
            solver='lsqr',
            shrinkage='auto',
            priors=priors
        )
        self.lda_model.fit(self.X_train_scaled, self.y_train)
        
        # Prédictions de probabilité
        y_proba_test = self.lda_model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Optimiser le seuil
        self.best_threshold_lda, best_f1 = self.find_optimal_threshold(
            y_proba_test, self.y_test, "LDA"
        )
        
        # Prédictions finales avec seuil optimal
        y_pred_test = (y_proba_test >= self.best_threshold_lda).astype(int)
        
        # Métriques
        self.lda_metrics = {
            'accuracy_test': accuracy_score(self.y_test, y_pred_test),
            'precision_test': precision_score(self.y_test, y_pred_test, zero_division=0),
            'recall_test': recall_score(self.y_test, y_pred_test, zero_division=0),
            'f1_test': f1_score(self.y_test, y_pred_test, zero_division=0),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred_test),
            'threshold': self.best_threshold_lda
        }
        
        # ROC
        fpr, tpr, _ = roc_curve(self.y_test, y_proba_test)
        self.lda_metrics['auc'] = auc(fpr, tpr)
        self.lda_metrics['fpr'] = fpr
        self.lda_metrics['tpr'] = tpr
        
        print("RÉSULTATS LDA :")
        print(f"  Accuracy  : {self.lda_metrics['accuracy_test']*100:.2f}%")
        print(f"  Precision : {self.lda_metrics['precision_test']*100:.2f}%")
        print(f"  Recall    : {self.lda_metrics['recall_test']*100:.2f}%")
        print(f"  F1-Score  : {self.lda_metrics['f1_test']*100:.2f}%")
        print(f"  AUC       : {self.lda_metrics['auc']:.4f}")
        print(f"\nMatrice de confusion :")
        print(self.lda_metrics['confusion_matrix'])
        print("\n✓ LDA entraîné\n")
        
    def train_qda(self):
        """Entraîne QDA avec ajustement du seuil"""
        print("=" * 70)
        print("ENTRAÎNEMENT QDA")
        print("=" * 70)
        
        # Entraîner avec priors ajustés
        class_counts = self.y_train.value_counts()
        priors = [class_counts[0]/len(self.y_train), class_counts[1]/len(self.y_train)]
        
        print(f"Priors utilisés : {priors}\n")
        
        self.qda_model = QuadraticDiscriminantAnalysis(
            reg_param=0.1,
            priors=priors
        )
        self.qda_model.fit(self.X_train_scaled, self.y_train)
        
        # Prédictions de probabilité
        y_proba_test = self.qda_model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Optimiser le seuil
        self.best_threshold_qda, best_f1 = self.find_optimal_threshold(
            y_proba_test, self.y_test, "QDA"
        )
        
        # Prédictions finales avec seuil optimal
        y_pred_test = (y_proba_test >= self.best_threshold_qda).astype(int)
        
        # Métriques
        self.qda_metrics = {
            'accuracy_test': accuracy_score(self.y_test, y_pred_test),
            'precision_test': precision_score(self.y_test, y_pred_test, zero_division=0),
            'recall_test': recall_score(self.y_test, y_pred_test, zero_division=0),
            'f1_test': f1_score(self.y_test, y_pred_test, zero_division=0),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred_test),
            'threshold': self.best_threshold_qda
        }
        
        # ROC
        fpr, tpr, _ = roc_curve(self.y_test, y_proba_test)
        self.qda_metrics['auc'] = auc(fpr, tpr)
        self.qda_metrics['fpr'] = fpr
        self.qda_metrics['tpr'] = tpr
        
        print("RÉSULTATS QDA :")
        print(f"  Accuracy  : {self.qda_metrics['accuracy_test']*100:.2f}%")
        print(f"  Precision : {self.qda_metrics['precision_test']*100:.2f}%")
        print(f"  Recall    : {self.qda_metrics['recall_test']*100:.2f}%")
        print(f"  F1-Score  : {self.qda_metrics['f1_test']*100:.2f}%")
        print(f"  AUC       : {self.qda_metrics['auc']:.4f}")
        print(f"\nMatrice de confusion :")
        print(self.qda_metrics['confusion_matrix'])
        print("\n✓ QDA entraîné\n")
        
    def compare_models(self):
        """Compare les modèles"""
        print("=" * 70)
        print("COMPARAISON DES MODÈLES")
        print("=" * 70)
        
        comparison = pd.DataFrame({
            'Métrique': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'Seuil'],
            'LDA': [
                f"{self.lda_metrics['accuracy_test']*100:.2f}%",
                f"{self.lda_metrics['precision_test']*100:.2f}%",
                f"{self.lda_metrics['recall_test']*100:.2f}%",
                f"{self.lda_metrics['f1_test']*100:.2f}%",
                f"{self.lda_metrics['auc']:.4f}",
                f"{self.best_threshold_lda:.2f}"
            ],
            'QDA': [
                f"{self.qda_metrics['accuracy_test']*100:.2f}%",
                f"{self.qda_metrics['precision_test']*100:.2f}%",
                f"{self.qda_metrics['recall_test']*100:.2f}%",
                f"{self.qda_metrics['f1_test']*100:.2f}%",
                f"{self.qda_metrics['auc']:.4f}",
                f"{self.best_threshold_qda:.2f}"
            ]
        })
        
        print(comparison.to_string(index=False))
        print()
        
        best_model = 'LDA' if self.lda_metrics['f1_test'] >= self.qda_metrics['f1_test'] else 'QDA'
        print(f"🏆 Meilleur modèle : {best_model}\n")
        
        return best_model
        
    def save_models(self, output_dir='models'):
        """Sauvegarde les modèles"""
        print("=" * 70)
        print("SAUVEGARDE")
        print("=" * 70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        joblib.dump(self.lda_model, f'{output_dir}/lda_model.pkl')
        joblib.dump(self.qda_model, f'{output_dir}/qda_model.pkl')
        joblib.dump(self.scaler, f'{output_dir}/scaler.pkl')
        
        joblib.dump({
            'lda': self.lda_metrics,
            'qda': self.qda_metrics,
            'features': self.features,
            'threshold_lda': self.best_threshold_lda,
            'threshold_qda': self.best_threshold_qda
        }, f'{output_dir}/metrics.pkl')
        
        print(f"✓ Modèles sauvegardés\n")
        
    def train_all(self, test_size=0.2, random_state=42):
        """Pipeline complet"""
        self.prepare_data(test_size=test_size, random_state=random_state)
        self.train_lda()
        self.train_qda()
        best_model = self.compare_models()
        self.save_models()
        return best_model


def load_trained_models(models_dir='models'):
    """Charge les modèles"""
    lda_model = joblib.load(f'{models_dir}/lda_model.pkl')
    qda_model = joblib.load(f'{models_dir}/qda_model.pkl')
    scaler = joblib.load(f'{models_dir}/scaler.pkl')
    metrics = joblib.load(f'{models_dir}/metrics.pkl')
    return lda_model, qda_model, scaler, metrics


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("ENTRAÎNEMENT DES MODÈLES LDA VS QDA")
    print("=" * 70 + "\n")
    
    trainer = CreditRiskModels()
    best_model = trainer.train_all(test_size=0.2, random_state=42)
    
    print("=" * 70)
    print("✅ TERMINÉ")
    print("=" * 70 + "\n")