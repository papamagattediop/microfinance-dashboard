import dash
from dash import html, dcc, callback, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils.data_loader import load_data, get_basic_stats
from dash import no_update


dash.register_page(__name__, path='/', name='Analyse')

df = load_data()
stats = get_basic_stats(df)

# Dictionnaire de traduction des variables
LABELS_FR = {
    'age': 'Âge',
    'revenu_mensuel_xof': 'Revenu mensuel',
    'nb_dependants': 'Nombre de dépendants',
    'usage_mobile_money_score': 'Usage Mobile Money',
    'anciennete_relation_mois': 'Ancienneté (mois)',
    'historique_credit_mois': 'Historique crédit (mois)',
    'jours_retard_12m': 'Jours de retard (12m)',
    'montant_pret_xof': 'Montant du prêt',
    'duree_mois': 'Durée (mois)',
    'taux_interet_annuel_pct': 'Taux d\'intérêt (%)',
    'pret_groupe': 'Prêt de groupe',
    'dsti_pct': 'Ratio DSTI (%)',
    'defaut_90j': 'Défaut 90 jours',
    'taux_epargne_pct': 'Taux d\'épargne (%)',
    'montant_epargne_xof': 'Montant épargne'
}

def get_label(col):
    """Retourne le label français ou le nom de colonne si pas de traduction"""
    return LABELS_FR.get(col, col.replace('_', ' ').title())

# Récupérer les colonnes numériques pour le scatter
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
# Retirer la colonne cible
if 'defaut_90j' in numeric_cols:
    numeric_cols.remove('defaut_90j')

layout = dbc.Container([
    
    # Section KPIs Premium (dynamiques)
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.I(className="fas fa-exclamation-triangle", style={'fontSize': '24px'})
                ], className="kpi-icon"),
                html.P("Taux de défaut", className="kpi-title"),
                html.H2(id="kpi-taux-defaut", className="kpi-value"),
                html.P("Défauts sur 90 jours", className="kpi-change text-muted")
            ], className="kpi-card danger fade-in")
        ], width=12, md=6, lg=3, className="mb-3"),
        
        dbc.Col([
            html.Div([
                html.Div([
                    html.I(className="fas fa-users", style={'fontSize': '24px'})
                ], className="kpi-icon"),
                html.P("Total clients", className="kpi-title"),
                html.H2(id="kpi-nb-clients", className="kpi-value"),
                html.P("Portefeuille actif", className="kpi-change text-muted")
            ], className="kpi-card primary fade-in")
        ], width=12, md=6, lg=3, className="mb-3"),
        
        dbc.Col([
            html.Div([
                html.Div([
                    html.I(className="fas fa-check-circle", style={'fontSize': '24px'})
                ], className="kpi-icon"),
                html.P("Clients sains", className="kpi-title"),
                html.H2(id="kpi-clients-sains", className="kpi-value"),
                html.P("Remboursements à jour", className="kpi-change text-muted")
            ], className="kpi-card success fade-in")
        ], width=12, md=6, lg=3, className="mb-3"),
        
        dbc.Col([
            html.Div([
                html.Div([
                    html.I(className="fas fa-shield-alt", style={'fontSize': '24px'})
                ], className="kpi-icon"),
                html.P("Clients à risque", className="kpi-title"),
                html.H2(id="kpi-clients-risque", className="kpi-value"),
                html.P("Surveillance renforcée", className="kpi-change text-muted")
            ], className="kpi-card warning fade-in")
        ], width=12, md=6, lg=3, className="mb-3"),
    ], className="mb-4"),
    
    # Section principale avec filtres et graphiques
    dbc.Row([
        # Colonne filtres
        dbc.Col([
            html.Div([
                html.H4([
                    html.I(className="fas fa-filter me-2"),
                    "Filtres de recherche"
                ], className="filter-title"),
                
                html.Label("Région géographique", className="filter-label"),
                dcc.Dropdown(
                    id='filter-region',
                    options=[{'label': 'Toutes les régions', 'value': 'ALL'}] + 
                            [{'label': r, 'value': r} for r in sorted(df['region'].unique())],
                    value='ALL',
                    className="mb-3",
                    clearable=False
                ),
                
                html.Label("Secteur d'activité", className="filter-label"),
                dcc.Dropdown(
                    id='filter-secteur',
                    options=[{'label': 'Tous les secteurs', 'value': 'ALL'}] + 
                            [{'label': s, 'value': s} for s in sorted(df['secteur_activite'].unique())],
                    value='ALL',
                    className="mb-3",
                    clearable=False
                ),
                
                html.Label("Canal d'octroi", className="filter-label"),
                dcc.Dropdown(
                    id='filter-canal',
                    options=[{'label': 'Tous les canaux', 'value': 'ALL'}] + 
                            [{'label': c, 'value': c} for c in sorted(df['canal_octroi'].unique())],
                    value='ALL',
                    className="mb-3",
                    clearable=False
                ),
                
                html.Label("Montant du prêt (XOF)", className="filter-label"),
                dcc.RangeSlider(
                    id='filter-montant',
                    min=df['montant_pret_xof'].min(),
                    max=df['montant_pret_xof'].max(),
                    value=[df['montant_pret_xof'].min(), df['montant_pret_xof'].max()],
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                    className="mb-4"
                ),
                
                html.Div([
                    html.I(className="fas fa-info-circle me-2"),
                    html.Span(id="info-nb-dossiers", className="text-muted small")
                ])
                
            ], className="filter-section")
        ], width=12, lg=3, className="mb-4"),
        
        # Colonne graphiques
        dbc.Col([
            html.Div([
                dbc.Tabs([
                    dbc.Tab([
                        html.I(className="fas fa-chart-bar me-2"),
                        "Distribution DSTI"
                    ], tab_id="tab-dsti"),
                    dbc.Tab([
                        html.I(className="fas fa-circle me-2"),
                        "Analyse bivariée"
                    ], tab_id="tab-scatter"),
                    dbc.Tab([
                        html.I(className="fas fa-th me-2"),
                        "Corrélations"
                    ], tab_id="tab-corr"),
                ], id="tabs", active_tab="tab-dsti", className="custom-tabs"),
            ]),
            
            # Conteneur DSTI - toujours présent dans le DOM
            html.Div(id="graph-dsti", className="graph-container fade-in"),
            
            # Conteneur Scatter - toujours présent dans le DOM
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Label("Variable X (axe horizontal):", className="filter-label"),
                        dcc.Dropdown(
                            id='scatter-x',
                            options=[{'label': get_label(col), 'value': col} for col in numeric_cols],
                            value='revenu_mensuel_xof',
                            clearable=False,
                            className="mb-3"
                        )
                    ], width=6),
                    dbc.Col([
                        html.Label("Variable Y (axe vertical):", className="filter-label"),
                        dcc.Dropdown(
                            id='scatter-y',
                            options=[{'label': get_label(col), 'value': col} for col in numeric_cols],
                            value='montant_pret_xof',
                            clearable=False,
                            className="mb-3"
                        )
                    ], width=6),
                ]),
                html.Div(id='scatter-plot-graph')
            ], id="graph-scatter", className="graph-container fade-in"),
            
            # Conteneur Corrélations - toujours présent dans le DOM
            html.Div(id="graph-corr", className="graph-container fade-in")
            
        ], width=12, lg=9)
    ]),
    
    # Section tableau
    dbc.Row([
        dbc.Col([
            html.H4([
                html.I(className="fas fa-table me-2"),
                "Aperçu des données"
            ], className="section-title"),
            html.Div(id="data-table", className="fade-in")
        ])
    ], className="mt-4")
    
], fluid=True)

# Callback pour mettre à jour les KPIs en fonction des filtres
@callback(
    [Output("kpi-taux-defaut", "children"),
     Output("kpi-nb-clients", "children"),
     Output("kpi-clients-sains", "children"),
     Output("kpi-clients-risque", "children"),
     Output("info-nb-dossiers", "children")],
    [Input("filter-region", "value"),
     Input("filter-secteur", "value"),
     Input("filter-canal", "value"),
     Input("filter-montant", "value")]
)
def update_kpis(region, secteur, canal, montant):
    dff = df.copy()
    
    if region != 'ALL':
        dff = dff[dff['region'] == region]
    if secteur != 'ALL':
        dff = dff[dff['secteur_activite'] == secteur]
    if canal != 'ALL':
        dff = dff[dff['canal_octroi'] == canal]
    
    dff = dff[(dff['montant_pret_xof'] >= montant[0]) & 
              (dff['montant_pret_xof'] <= montant[1])]
    
    # Calcul des KPIs
    taux_defaut = dff['defaut_90j'].mean() * 100
    nb_clients = len(dff)
    taux_sains = 100 - taux_defaut
    nb_risque = int(nb_clients * taux_defaut / 100)
    
    return (
        f"{taux_defaut:.2f}%",
        f"{nb_clients:,}",
        f"{taux_sains:.2f}%",
        f"{nb_risque:,}",
        f"{nb_clients} dossiers disponibles"
    )

# Callback pour gérer la visibilité des onglets (sans détruire le DOM)
@callback(
    [Output("graph-dsti", "style"),
     Output("graph-scatter", "style"),
     Output("graph-corr", "style")],
    Input("tabs", "active_tab")
)
def toggle_tabs_visibility(active_tab):
    """Affiche/masque les conteneurs sans les détruire"""
    dsti_style = {'display': 'block'} if active_tab == "tab-dsti" else {'display': 'none'}
    scatter_style = {'display': 'block'} if active_tab == "tab-scatter" else {'display': 'none'}
    corr_style = {'display': 'block'} if active_tab == "tab-corr" else {'display': 'none'}
    
    return dsti_style, scatter_style, corr_style


# Callback pour mettre à jour le graphique DSTI
@callback(
    Output("graph-dsti", "children"),
    [Input("filter-region", "value"),
     Input("filter-secteur", "value"),
     Input("filter-canal", "value"),
     Input("filter-montant", "value")]
)
def update_dsti_graph(region, secteur, canal, montant):
    dff = df.copy()
    
    if region != 'ALL':
        dff = dff[dff['region'] == region]
    if secteur != 'ALL':
        dff = dff[dff['secteur_activite'] == secteur]
    if canal != 'ALL':
        dff = dff[dff['canal_octroi'] == canal]
    
    dff = dff[(dff['montant_pret_xof'] >= montant[0]) & 
              (dff['montant_pret_xof'] <= montant[1])]
    
    fig = px.histogram(
        dff, x='dsti_pct', color='defaut_90j',
        title="Distribution du ratio DSTI par statut de défaut",
        labels={'dsti_pct': 'Ratio DSTI (%)', 'defaut_90j': 'Statut'},
        barmode='overlay',
        template='plotly_white',
        color_discrete_map={0: '#86efac', 1: '#fca5a5'}  # Vert très clair et rouge très clair
    )
    fig.update_layout(
        title_font_size=18,
        title_font_family="Inter",
        showlegend=True,
        legend_title_text="Statut",
        height=500,
        legend=dict(
            itemsizing='constant',
            traceorder='normal'
        )
    )
    # Renommer les légendes pour plus de clarté
    fig.for_each_trace(lambda t: t.update(name='Pas de défaut' if t.name == '0' else 'Défaut'))
    
    return html.Div([
        html.Div([
            html.P([
                html.I(className="fas fa-info-circle me-2"),
                html.Strong("Qu'est-ce que le ratio DSTI ? "),
                "Le ratio DSTI (Debt Service-to-Income) est un indicateur financier clé mesurant la part des revenus mensuels nets d'un emprunteur consacrée au remboursement de ses dettes."
            ], style={'fontSize': '0.9rem', 'color': '#4b5563', 'marginBottom': '15px', 
                     'padding': '15px', 'backgroundColor': '#f9fafb', 'borderRadius': '10px',
                     'borderLeft': '4px solid #CC8B65'})
        ]),
        dcc.Graph(figure=fig, config={'displayModeBar': False})
    ])

# Callback pour le scatter plot dynamique - VERSION CORRIGÉE
@callback(
    Output('scatter-plot-graph', 'children'),
    [Input('scatter-x', 'value'),
     Input('scatter-y', 'value'),
     Input("filter-region", "value"),
     Input("filter-secteur", "value"),
     Input("filter-canal", "value"),
     Input("filter-montant", "value")]
)
def update_scatter_plot(x_var, y_var, region, secteur, canal, montant):
    dff = df.copy()
    
    if region != 'ALL':
        dff = dff[dff['region'] == region]
    if secteur != 'ALL':
        dff = dff[dff['secteur_activite'] == secteur]
    if canal != 'ALL':
        dff = dff[dff['canal_octroi'] == canal]
    
    dff = dff[(dff['montant_pret_xof'] >= montant[0]) & 
              (dff['montant_pret_xof'] <= montant[1])]
    
    # Convertir defaut_90j en catégorie pour forcer une légende discrète
    dff['defaut_90j_cat'] = dff['defaut_90j'].map({0: 'Pas de défaut', 1: 'Défaut'})
    
    fig = px.scatter(
        dff, x=x_var, y=y_var,
        color='defaut_90j_cat',
        title=f"Relation entre {get_label(x_var)} et {get_label(y_var)}",
        labels={
            x_var: get_label(x_var),
            y_var: get_label(y_var),
            'defaut_90j_cat': 'Statut'
        },
        template='plotly_white',
        color_discrete_map={'Pas de défaut': '#10b981', 'Défaut': '#f87171'},
        category_orders={'defaut_90j_cat': ['Pas de défaut', 'Défaut']},
        opacity=0.6
    )
    fig.update_layout(
        title_font_size=18,
        title_font_family="Inter",
        height=500,
        legend_title_text="Statut",
        legend=dict(
            itemsizing='constant',
            traceorder='normal'
        )
    )
    
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


# Callback pour mettre à jour le graphique de corrélation
@callback(
    Output("graph-corr", "children"),
    [Input("filter-region", "value"),
     Input("filter-secteur", "value"),
     Input("filter-canal", "value"),
     Input("filter-montant", "value")]
)
def update_corr_graph(region, secteur, canal, montant):
    dff = df.copy()
    
    if region != 'ALL':
        dff = dff[dff['region'] == region]
    if secteur != 'ALL':
        dff = dff[dff['secteur_activite'] == secteur]
    if canal != 'ALL':
        dff = dff[dff['canal_octroi'] == canal]
    
    dff = dff[(dff['montant_pret_xof'] >= montant[0]) & 
              (dff['montant_pret_xof'] <= montant[1])]
    
    # Matrice de corrélation avec labels explicites
    numeric_data = dff.select_dtypes(include=['number'])
    
    # Sélectionner les variables les plus importantes (max 12 variables)
    important_vars = ['age', 'revenu_mensuel_xof', 'montant_pret_xof', 
                      'dsti_pct', 'duree_mois', 'taux_interet_annuel_pct',
                      'jours_retard_12m', 'anciennete_relation_mois', 
                      'nb_dependants', 'usage_mobile_money_score', 'defaut_90j']
    
    # Garder seulement les variables présentes dans les données
    available_vars = [v for v in important_vars if v in numeric_data.columns]
    
    # Calculer la corrélation
    corr = numeric_data[available_vars].corr()
    
    # Renommer les colonnes et index avec les labels français
    corr_renamed = corr.copy()
    corr_renamed.columns = [get_label(col) for col in corr.columns]
    corr_renamed.index = [get_label(col) for col in corr.index]
    
    fig = px.imshow(
        corr_renamed, 
        text_auto='.2f',
        aspect="auto",
        title="Matrice de corrélation des variables clés",
        template='plotly_white',
        color_continuous_scale='RdYlGn',
        zmin=-1,
        zmax=1
    )
    
    fig.update_layout(
        title_font_size=18,
        title_font_family="Inter",
        height=700,
        xaxis_title="",
        yaxis_title="",
        font=dict(size=11)
    )
    
    # Ajouter une légende explicative
    return html.Div([
        html.Div([
            html.P([
                html.I(className="fas fa-info-circle me-2"),
                html.Strong("Interprétation : "),
                "Les valeurs proches de +1 (vert) indiquent une corrélation positive forte. "
                "Les valeurs proches de -1 (rouge) indiquent une corrélation négative forte. "
                "Les valeurs proches de 0 (jaune) indiquent une absence de corrélation linéaire."
            ], style={'fontSize': '0.9rem', 'color': '#4b5563', 'marginBottom': '15px', 
                     'padding': '15px', 'backgroundColor': '#f9fafb', 'borderRadius': '10px',
                     'borderLeft': '4px solid #CC8B65'})
        ]),
        dcc.Graph(figure=fig, config={'displayModeBar': False})
    ])

# Callback pour le tableau
@callback(
    Output("data-table", "children"),
    [Input("filter-region", "value"),
     Input("filter-secteur", "value"),
     Input("filter-canal", "value"),
     Input("filter-montant", "value")]
)
def update_table(region, secteur, canal, montant):
    dff = df.copy()
    
    if region != 'ALL':
        dff = dff[dff['region'] == region]
    if secteur != 'ALL':
        dff = dff[dff['secteur_activite'] == secteur]
    if canal != 'ALL':
        dff = dff[dff['canal_octroi'] == canal]
    
    dff = dff[(dff['montant_pret_xof'] >= montant[0]) & 
              (dff['montant_pret_xof'] <= montant[1])]
    
    display_cols = ['region', 'secteur_activite', 'age', 'revenu_mensuel_xof', 
                   'montant_pret_xof', 'dsti_pct', 'jours_retard_12m', 'defaut_90j']
    
    return dash_table.DataTable(
        data=dff[display_cols].head(100).to_dict('records'),
        columns=[{"name": get_label(i), "id": i} for i in display_cols],
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '12px',
            'fontFamily': 'Inter',
            'fontSize': '14px'
        },
        style_header={
            'backgroundColor': '#013328',
            'color': 'white',
            'fontWeight': '600',
            'border': 'none'
        },
        style_data={
            'border': 'none',
            'borderBottom': '1px solid #E3DCD2'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 250, 252)'
            },
            {
                'if': {'column_id': 'defaut_90j', 'filter_query': '{defaut_90j} = 1'},
                'backgroundColor': '#fee2e2',
                'color': '#991b1b',
                'fontWeight': '600'
            }
        ],
        page_size=10,
        sort_action="native",
        filter_action="native"
    )