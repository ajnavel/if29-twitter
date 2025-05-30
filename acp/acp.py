import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import joblib

# Définition du chemin racine du projet
root = Path(__file__).resolve().parent.parent

# Chargement des données utilisateurs enrichies de scores
csv_path = root / "data" / "processed" / "user_profiles_with_scores.csv"
df = pd.read_csv(csv_path, dtype={"user_id": str}).fillna(0)

# Liste des variables utilisées pour la PCA
FEATURES = [
    "mean_text_length", "mean_text_upper_ratio", "mean_text_exclam_ratio",
    "mean_nb_hashtags", "mean_nb_mentions", "mean_is_retweet",
    "mean_tweet_hour", "user.followers_count", "user.friends_count",
    "user.statuses_count", "log_ratio_followers_friends",
    "mean_hashtag_spam_score", "mean_mention_repetition_score",
    "mean_text_repetition_score", "account_age_days",
    "followers_per_tweet", "mean_engagement_rate",
    "mean_late_night_tweet"
]
X = df[FEATURES]

# Standardisation des données et réduction de dimension à 3 composantes
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
print("Variance expliquée (PC1–3) :", pca.explained_variance_ratio_)

# Définition des catégories de type et couleurs associées
type_order = ["normal", "bot", "spam", "influenceur", "media", "spam_star", "atypique_generique"]
color_map = {
    "normal": "#2ecc71",
    "bot": "#3498db",
    "spam": "#e74c3c",
    "influenceur": "#f39c12",
    "media": "#e67e22",
    "spam_star": "#c0392b",
    "atypique_generique": "#9b59b6"
}
df['type_user'] = df['type_user'].astype(pd.CategoricalDtype(type_order, ordered=True))

# Définition des groupes pour affichage groupé
group_normal = ["normal", "influenceur"]
group_atypique = ["bot", "spam", "media", "spam_star", "atypique_generique"]
group_colors = {"normal": "#2ecc71", "atypique": "#e74c3c"}

# Comptage des types (individuels et groupés) + calcul des pourcentages
counts = df['type_user'].value_counts().reindex(type_order).fillna(0)
percentages = (counts / counts.sum() * 100).round(1)

grouped_counts = {
    "normal": counts[group_normal].sum(),
    "atypique": counts[group_atypique].sum()
}
grouped_percentages = {
    k: round((v / sum(grouped_counts.values())) * 100, 1)
    for k, v in grouped_counts.items()
}

# Création de la figure Plotly avec 2 sous-graphes (3D et histogramme)
fig = make_subplots(
    rows=2, cols=1,
    specs=[[{"type": "scene"}], [{"type": "xy"}]],
    row_heights=[0.7, 0.3],
    vertical_spacing=0.08,
    subplot_titles=("ACP 3D des profils Twitter", "Répartition des types (%)")
)

# Traces 3D pour chaque type individuel
for t in type_order:
    mask = df['type_user'] == t
    fig.add_trace(
        go.Scatter3d(
            x=X_pca[mask, 0], y=X_pca[mask, 1], z=X_pca[mask, 2],
            mode='markers',
            name=t,
            marker=dict(size=1.5, color=color_map[t], opacity=0.6, line=dict(width=0)),            
            customdata=df.loc[mask, 'user_id']
        ),
        row=1, col=1
    )

# Traces 3D pour les deux groupes agrégés (cachées par défaut)
mask_normal = df['type_user'].isin(group_normal)
fig.add_trace(
    go.Scatter3d(
        x=X_pca[mask_normal, 0], y=X_pca[mask_normal, 1], z=X_pca[mask_normal, 2],
        mode='markers',
        name="normal (groupé)",
        marker=dict(size=1.5, color=group_colors["normal"], opacity=0.6, line=dict(width=0)),            
        
        visible=False,
        customdata=df.loc[mask_normal, 'user_id']
    ),
    row=1, col=1
)

mask_atypique = df['type_user'].isin(group_atypique)
fig.add_trace(
    go.Scatter3d(
        x=X_pca[mask_atypique, 0], y=X_pca[mask_atypique, 1], z=X_pca[mask_atypique, 2],
        mode='markers',
        name="atypique (groupé)",
        marker=dict(size=1.5, color=group_colors["atypique"], opacity=0.6, line=dict(width=0)),            
        
        visible=False,
        customdata=df.loc[mask_atypique, 'user_id']
    ),
    row=1, col=1
)


# 1. Trier du + petit % au + grand
sorted_percentages = percentages.sort_values()
sorted_types       = sorted_percentages.index.tolist()
sorted_counts      = counts.loc[sorted_types]

# 2. Créer la trace
fig.add_trace(
    go.Bar(
        x=sorted_types,
        y=sorted_percentages.values,
        customdata=sorted_counts.values,              
        hovertemplate="Total : %{customdata}<br>% : %{y}%<extra></extra>",
        marker_color=[color_map[t] for t in sorted_types],
        text=[f"{v}%" for v in sorted_percentages.values],
        textposition='outside',
        showlegend=False,
        name="Répartition types (séparés)"
    ),
    row=2, col=1
)

# Histogramme groupé (normal / atypique), caché par défaut
fig.add_trace(
    go.Bar(
        x=["normal", "atypique"],
        y=[grouped_percentages["normal"], grouped_percentages["atypique"]],
        marker_color=[group_colors["normal"], group_colors["atypique"]],
        text=[f"{grouped_percentages['normal']}%", f"{grouped_percentages['atypique']}%"],
        textposition='outside',
        showlegend=False,
        name="Répartition types (groupé)",
        visible=False
    ),
    row=2, col=1
)

# Affichage du nombre total de comptes analysés
fig.add_annotation(
    text=f"Total comptes affichés : {len(df)}",
    xref='paper', yref='paper',
    x=0.01, y=0.95,
    showarrow=False,
    font=dict(size=14)
)

# Boutons d'interaction pour basculer entre vue groupée et détaillée
n_types = len(type_order)
trace_grouped_3d = [False]*n_types + [True, True]
trace_individual_3d = [True]*n_types + [False, False]
bar_individual = [True, False]
bar_grouped = [False, True]

fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            buttons=[
                dict(
                    label="Afficher tous les types",
                    method="update",
                    args=[{"visible": trace_individual_3d + bar_individual}]
                ),
                dict(
                    label="Afficher types groupés (normal / atypique)",
                    method="update",
                    args=[{"visible": trace_grouped_3d + bar_grouped}]
                )
            ],
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.5,
            xanchor="center",
            y=1.1,
            yanchor="top"
        )
    ]
)

# Configuration finale du layout
fig.update_layout(
    height=1000,
    margin=dict(l=0, r=0, t=60, b=0),
    title_x=0.5
)
fig.update_scenes(
    xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'
)

# Export du graphique interactif en HTML
output_path = root / "visualisations" / "acp_visualisation_all_small.html"
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(
    str(output_path),
    include_plotlyjs='cdn'
)

# Sauvegarde des modèles PCA et scaler pour réutilisation ultérieure
models_dir = root / "models"
models_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(pca, models_dir / "pca_3d_all_small.joblib")
joblib.dump(scaler, models_dir / "scaler_all_small.joblib")

print(f"✅ Visualisation exportée dans : {output_path}")
print(f"✅ PCA et scaler sauvegardés dans : {models_dir}")

# ── AFFICHAGE SIMPLE : normal vs atypique ──

# 1) Calcul PCA 2D (inchangé)
from sklearn.decomposition import PCA as PCA2D

pca2 = PCA2D(n_components=2)
X_pca2 = pca2.fit_transform(X_scaled)
var2 = pca2.explained_variance_ratio_

# 2) Définition des deux masques
mask_normal   = df['type_user'] == 'normal'
mask_atypique = df['type_user'] != 'normal'

# 3) Création de la figure
fig2 = go.Figure()

# trace “normal”
fig2.add_trace(
    go.Scatter(
        x=X_pca2[mask_normal, 0],
        y=X_pca2[mask_normal, 1],
        mode='markers',
        name='normal',
        marker=dict(color='#2ecc71', size=6, opacity=0.7, line=dict(width=0))
    )
)

# trace “atypique”
fig2.add_trace(
    go.Scatter(
        x=X_pca2[mask_atypique, 0],
        y=X_pca2[mask_atypique, 1],
        mode='markers',
        name='atypique',
        marker=dict(color='crimson', size=6, opacity=0.7, line=dict(width=0))
    )
)

# 4) Mise en forme
fig2.update_layout(
    title="ACP 2D : normal vs atypique",
    xaxis=dict(title=f"PC1 ({var2[0]*100:.1f}% de variance)", zeroline=False),
    yaxis=dict(title=f"PC2 ({var2[1]*100:.1f}% de variance)", zeroline=False),
    legend_title="Catégorie",
    width=800, height=600, margin=dict(l=40, r=40, t=60, b=40)
)

# 5) Affichage
fig2.show()
