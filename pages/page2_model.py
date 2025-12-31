"""
Page de Modélisation et Prédiction
Dashboard Dash pour l'analyse comparative LDA vs QDA
"""

import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import sys

# Importer les fonctions de chargement des modèles
sys.path.append('..')
from utils.model_trainer import load_trained_models

# ============================================================================
# CONFIGURATION DE LA PAGE
# ============================================================================

dash.register_page(__name__, path='/modelisation', name='Modélisation')

# Vérifier si les modèles existent
MODELS_EXIST = os.path.exists('models/lda_model.pkl')

# Charger les modèles si disponibles
if MODELS_EXIST:
    lda_model, qda_model, scaler, metrics = load_trained_models()
    lda_metrics = metrics['lda']
    qda_metrics = metrics['qda']
    features = metrics['features']


# ============================================================================
# COMPOSANTS UI - HELPER FUNCTIONS
# ============================================================================

def create_alert_models_not_trained():
    """Crée l'alerte pour les modèles non entraînés"""
    return dbc.Alert([
        html.H5([
            html.I(className="fas fa-exclamation-triangle me-2"),
            "Modèles non entraînés"
        ], className="alert-heading"),
        html.P("Les modèles LDA et QDA n'ont pas encore été entraînés."),
        html.Hr(),
        html.P("Pour entraîner les modèles optimaux, exécutez dans le terminal :"),
        html.Code(
            "python utils/model_trainer.py",
            style={
                'backgroundColor': '#f1f3f5',
                'padding': '10px',
                'borderRadius': '5px',
                'display': 'block',
                'marginTop': '10px'
            }
        )
    ], color="warning", style={'borderRadius': '10px'})


def create_input_field(label, input_id, default_value, width=6):
    """Crée un champ de saisie formaté"""
    return dbc.Col([
        html.Label(label, className="filter-label"),
        dcc.Input(
            id=input_id,
            type='number',
            value=default_value,
            className="form-control mb-3",
            style={'borderRadius': '10px', 'border': '2px solid #E3DCD2'}
        )
    ], width=width)


def create_model_metrics_card(model_name, model_icon, metrics, color_scheme):
    """Crée une carte avec les métriques d'un modèle"""
    return html.Div([
        html.H5([
            html.I(className=f"fas {model_icon} me-2"),
            model_name
        ], className="text-center mb-3",
           style={'color': '#013328', 'fontWeight': '600'}),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.P("Exactitude", className="kpi-title"),
                    html.H3(
                        f"{metrics['accuracy_test']*100:.2f}%",
                        style={'color': '#013328', 'fontWeight': '700'}
                    )
                ], className="text-center")
            ], width=4),
            dbc.Col([
                html.Div([
                    html.P("Score F1", className="kpi-title"),
                    html.H3(
                        f"{metrics['f1_test']*100:.2f}%",
                        style={'color': '#CC8B65', 'fontWeight': '700'}
                    )
                ], className="text-center")
            ], width=4),
            dbc.Col([
                html.Div([
                    html.P("AUC", className="kpi-title"),
                    html.H3(
                        f"{metrics['auc']:.3f}",
                        style={'color': '#34a853', 'fontWeight': '700'}
                    )
                ], className="text-center")
            ], width=4),
        ])
    ], style={
        'padding': '20px',
        'backgroundColor': '#f9fafb',
        'borderRadius': '10px',
        'border': '2px solid #E3DCD2'
    })


def create_confusion_matrix(metrics, title):
    """Crée un graphique de matrice de confusion"""
    return html.Div([
        html.H5([
            html.I(className="fas fa-table me-2"),
            title
        ], className="text-center mb-3",
           style={'color': '#013328', 'fontWeight': '600'}),
        dcc.Graph(
            figure=px.imshow(
                metrics['confusion_matrix'],
                labels=dict(x="Prédit", y="Réel", color="Effectif"),
                x=['Non défaut', 'Défaut'],
                y=['Non défaut', 'Défaut'],
                text_auto=True,
                color_continuous_scale='Greens'
            ).update_layout(height=350, margin=dict(l=50, r=50, t=50, b=50)),
            config={'displayModeBar': False},
            style={'height': '350px'}
        )
    ])


def create_roc_curve():
    """Crée le graphique des courbes ROC"""
    return dcc.Graph(
        figure=go.Figure([
            go.Scatter(
                x=lda_metrics['fpr'],
                y=lda_metrics['tpr'],
                name=f'LDA (AUC = {lda_metrics["auc"]:.3f})',
                line=dict(color='#013328', width=3)
            ),
            go.Scatter(
                x=qda_metrics['fpr'],
                y=qda_metrics['tpr'],
                name=f'QDA (AUC = {qda_metrics["auc"]:.3f})',
                line=dict(color='#CC8B65', width=3)
            ),
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                name='Aléatoire (référence)',
                line=dict(color='gray', width=2, dash='dash')
            )
        ]).update_layout(
            xaxis_title="Taux de faux positifs (1 - Spécificité)",
            yaxis_title="Taux de vrais positifs (Sensibilité)",
            height=500,
            template='plotly_white',
            showlegend=True,
            legend=dict(x=0.6, y=0.1),
            margin=dict(l=50, r=50, t=50, b=50)
        ),
        config={'displayModeBar': False},
        style={'height': '500px'}
    )


# ============================================================================
# LAYOUT PRINCIPAL
# ============================================================================

layout = dbc.Container([

    # En-tête
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H2([
                    html.I(className="fas fa-brain me-3"),
                    "Modélisation & Prédiction"
                ], style={'color': '#013328', 'fontWeight': '700'}),
                html.P(
                    "Analyse comparative LDA vs QDA pour la prédiction de défaut de crédit",
                    style={'color': '#6b7280', 'fontSize': '1.1rem'}
                )
            ], className="text-center mb-4")
        ])
    ]),

    # Alerte si modèles non entraînés
    dbc.Row([
        dbc.Col([
            html.Div([
                create_alert_models_not_trained()
            ]) if not MODELS_EXIST else html.Div()
        ])
    ], className="mb-4"),

    # Section : Résultats des modèles (si disponibles)
    dbc.Row([
        dbc.Col([
            html.Div(id='model-display')
        ])
    ]) if MODELS_EXIST else html.Div(),

   # Section : Prédiction dossier client (si modèles disponibles)
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4([
                    html.I(className="fas fa-user-check me-2"),
                    "Simulation de scoring crédit"
                ], className="section-title mb-4 mt-5"),

                dbc.Row([
                    # Colonne formulaire
                    dbc.Col([
                        html.Div([
                            html.H5(
                                "Profil du demandeur",
                                className="mb-3",
                                style={'color': '#013328', 'fontWeight': '600'}
                            ),

                            # Ligne 1 : Âge et Revenu
                            dbc.Row([
                                create_input_field("Âge", 'input-age', 35, 6),
                                create_input_field("Revenu mensuel (XOF)", 'input-revenu', 150000, 6),
                            ]),

                            # Ligne 2 : Montant et DSTI
                            dbc.Row([
                                create_input_field("Montant emprunté (XOF)", 'input-montant', 500000, 6),
                                create_input_field("Ratio DSTI (%)", 'input-dsti', 30, 6),
                            ]),

                            # Ligne 3 : Durée et Taux
                            dbc.Row([
                                create_input_field("Durée du crédit (mois)", 'input-duree', 12, 6),
                                create_input_field("Taux d'intérêt annuel (%)", 'input-taux', 18, 6),
                            ]),

                            # Ligne 4 : Historique et relation client
                            dbc.Row([
                                create_input_field("Jours de retard (12 derniers mois)", 'input-retard', 0, 6),
                                create_input_field("Ancienneté relation (mois)", 'input-anciennete', 24, 6),
                            ]),

                            # Ligne 5 : Épargne
                            dbc.Row([
                                create_input_field("Épargne disponible (XOF)", 'input-epargne', 100000, 12),
                            ]),

                            # Bouton de calcul
                            dbc.Button(
                                [
                                    html.I(className="fas fa-calculator me-2"),
                                    "Calculer le score de risque"
                                ],
                                id='predict-button',
                                color="primary",
                                className="w-100 mt-3",
                                style={
                                    'borderRadius': '10px',
                                    'fontWeight': '600',
                                    'padding': '15px'
                                },
                                disabled=not MODELS_EXIST
                            )
                        ])
                    ], width=12, lg=6, className="mb-4"),

                    # Colonne résultats
                    dbc.Col([
                        html.Div(id='prediction-results')
                    ], width=12, lg=6)
                ])

            ], className="filter-section")
        ])
    ], className="mt-5 mb-4") if MODELS_EXIST else html.Div()

], fluid=True)


# ============================================================================
# CALLBACKS
# ============================================================================

@callback(
    Output('model-display', 'children'),
    Input('model-display', 'id')
)
def display_models(dummy):
    """Affiche les résultats des modèles entraînés"""
    if not MODELS_EXIST:
        return html.Div()

    return html.Div([
        # Section métriques
        html.Div([
            html.H4([
                html.I(className="fas fa-chart-line me-2"),
                "Performance des modèles entraînés"
            ], className="section-title mb-4"),

            dbc.Row([
                # Métriques LDA
                dbc.Col([
                    create_model_metrics_card(
                        "Linear Discriminant Analysis (LDA)",
                        "fa-project-diagram",
                        lda_metrics,
                        "lda"
                    )
                ], width=12, md=6, className="mb-3"),

                # Métriques QDA
                dbc.Col([
                    create_model_metrics_card(
                        "Quadratic Discriminant Analysis (QDA)",
                        "fa-bezier-curve",
                        qda_metrics,
                        "qda"
                    )
                ], width=12, md=6, className="mb-3"),
            ])
        ], className="filter-section mb-4"),

        # Matrices de confusion et courbes ROC
        html.Div([
            html.H4([
                html.I(className="fas fa-chart-area me-2"),
                "Visualisations des performances"
            ], className="section-title mb-4"),

            # Matrices de confusion
            dbc.Row([
                dbc.Col([
                    create_confusion_matrix(lda_metrics, "Matrice de confusion - LDA")
                ], width=12, md=6, className="mb-3"),

                dbc.Col([
                    create_confusion_matrix(qda_metrics, "Matrice de confusion - QDA")
                ], width=12, md=6, className="mb-3"),
            ]),

            # Courbes ROC
            dbc.Row([
                dbc.Col([
                    html.H5([
                        html.I(className="fas fa-chart-area me-2"),
                        "Courbes ROC - Analyse comparative"
                    ], className="text-center mb-3",
                       style={'color': '#013328', 'fontWeight': '600'}),
                    create_roc_curve()
                ])
            ])
        ], className="filter-section")
    ])

@callback(
    Output('prediction-results', 'children'),
    [Input('predict-button', 'n_clicks')],
    [
        State('input-age', 'value'),
        State('input-revenu', 'value'),
        State('input-montant', 'value'),
        State('input-dsti', 'value'),
        State('input-duree', 'value'),
        State('input-taux', 'value'),
        State('input-retard', 'value'),
        State('input-anciennete', 'value'),
        State('input-epargne', 'value')
    ]
)
def predict_default(
    n_clicks,
    age,
    revenu,
    montant,
    dsti,
    duree,
    taux,
    retard,
    anciennete,
    epargne
):
    """Effectue la prédiction de défaut pour un nouveau client"""

    # État initial
    if n_clicks is None or not MODELS_EXIST:
        return html.Div([
            html.Div([
                html.I(
                    className="fas fa-chart-line",
                    style={'fontSize': '60px', 'color': '#E3DCD2'}
                ),
                html.P(
                    "Saisissez les informations du profil client et lancez le calcul",
                    className="text-muted mt-3",
                    style={'fontSize': '1rem'}
                )
            ], className="text-center", style={'padding': '80px 20px'})
        ])

    # Préparer les données du nouveau client
    new_client = np.array([[
        age,
        revenu,
        dsti,
        montant,
        duree,
        taux,
        retard,
        anciennete,
        epargne
    ]])

    new_client_scaled = scaler.transform(new_client)

    # Prédictions (probabilités en %)
    proba_lda = lda_model.predict_proba(new_client_scaled)[0][1] * 100
    proba_qda = qda_model.predict_proba(new_client_scaled)[0][1] * 100

    # Récupérer les seuils optimaux
    threshold_lda = metrics.get('threshold_lda', 0.5) * 100
    threshold_qda = metrics.get('threshold_qda', 0.5) * 100

    # Prédictions finales
    pred_lda = "DÉFAUT" if proba_lda >= threshold_lda else "NON DÉFAUT"
    pred_qda = "DÉFAUT" if proba_qda >= threshold_qda else "NON DÉFAUT"

    # Sélection du meilleur modèle
    best_model = "LDA" if lda_metrics['f1_test'] >= qda_metrics['f1_test'] else "QDA"
    best_proba = proba_lda if best_model == "LDA" else proba_qda
    best_threshold = threshold_lda if best_model == "LDA" else threshold_qda
    best_prediction = pred_lda if best_model == "LDA" else pred_qda

    # Décision finale
    decision = "REFUSER LE CRÉDIT" if best_prediction == "DÉFAUT" else "ACCEPTER LE CRÉDIT"
    decision_color = "#ef4444" if best_prediction == "DÉFAUT" else "#34a853"
    decision_icon = "fa-times-circle" if best_prediction == "DÉFAUT" else "fa-check-circle"

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5([
                        html.I(className="fas fa-project-diagram me-2"),
                        "Modèle LDA"
                    ], className="text-center mb-3",
                       style={'color': '#013328', 'fontWeight': '600'}),
                    html.Div([
                        html.P("Probabilité de défaut", className="kpi-title"),
                        html.H2(
                            f"{proba_lda:.1f}%",
                            style={
                                'color': '#ef4444' if proba_lda > 50 else '#34a853',
                                'fontWeight': '700'
                            }
                        )
                    ], className="text-center")
                ], style={
                    'padding': '30px',
                    'backgroundColor': '#f9fafb',
                    'borderRadius': '10px',
                    'border': '2px solid #E3DCD2'
                })
            ], width=6, className="mb-3"),

            dbc.Col([
                html.Div([
                    html.H5([
                        html.I(className="fas fa-bezier-curve me-2"),
                        "Modèle QDA"
                    ], className="text-center mb-3",
                       style={'color': '#013328', 'fontWeight': '600'}),
                    html.Div([
                        html.P("Probabilité de défaut", className="kpi-title"),
                        html.H2(
                            f"{proba_qda:.1f}%",
                            style={
                                'color': '#ef4444' if proba_qda > 50 else '#34a853',
                                'fontWeight': '700'
                            }
                        )
                    ], className="text-center")
                ], style={
                    'padding': '30px',
                    'backgroundColor': '#f9fafb',
                    'borderRadius': '10px',
                    'border': '2px solid #E3DCD2'
                })
            ], width=6, className="mb-3"),
        ]),

        html.Div([
            html.H4([
                html.I(className="fas fa-gavel me-2"),
                "Décision d'octroi"
            ], className="text-center mb-3",
               style={'color': '#013328', 'fontWeight': '600'}),

            html.Div([
                html.P(
                    f"Basée sur le modèle optimal : {best_model} "
                    f"(F1-Score: {max(lda_metrics['f1_test'], qda_metrics['f1_test']) * 100:.1f}%)",
                    className="text-center text-muted mb-3"
                ),
                html.Div([
                    html.I(
                        className=f"fas {decision_icon}",
                        style={'fontSize': '50px', 'color': decision_color}
                    )
                ], className="text-center mb-3"),
                html.H1(
                    decision,
                    className="text-center",
                    style={
                        'color': decision_color,
                        'fontWeight': '700',
                        'fontSize': '2rem'
                    }
                ),
                html.P(
                    f"Score de risque : {best_proba:.1f}%",
                    className="text-center",
                    style={
                        'fontSize': '1.2rem',
                        'color': '#6b7280',
                        'marginTop': '15px'
                    }
                )
            ], style={
                'padding': '40px',
                'backgroundColor': 'white',
                'borderRadius': '15px',
                'border': f'3px solid {decision_color}',
                'boxShadow': f'0 4px 12px {decision_color}40'
            })
        ], className="mt-4")
    ])

