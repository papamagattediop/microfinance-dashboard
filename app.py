import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

# Font Awesome pour les icônes
app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
    ],
    suppress_callback_exceptions=True
)

app.layout = dbc.Container([
    # En-tête Premium
    html.Div([
        html.Div([
            html.H1("Microfinance Credit Risk Dashboard", className="main-title"),
            html.P("Analyse prédictive et segmentation des risques de crédit", 
                  className="main-subtitle")
        ], className="main-header fade-in")
    ]),*
    
    # Navigation Premium
    html.Div([
        dbc.Nav([
            dbc.NavLink([
                html.I(className="fas fa-chart-line me-2"),
                "Analyse & Exploration"
            ], href="/", active="exact", className="fade-in"),
            dbc.NavLink([
                html.I(className="fas fa-brain me-2"),
                "Modélisation & Prédiction"
            ], href="/modelisation", active="exact", className="fade-in"),
        ], pills=True, className="custom-nav")
    ]),
    
    # Conteneur des pages
    dash.page_container
    
], fluid=True, className="fade-in")

server = app.server

if __name__ == '__main__':
    app.run(debug=True, port=8050)