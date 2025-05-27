import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import joblib

# 1. Chargement des données
root = Path(__file__).resolve().parent.parent
csv_path = root / "data" / "processed" / "user_profiles_with_scores.csv"
df = pd.read_csv(csv_path, dtype={"user_id": str}).fillna(0)

# 2. Sélection des features pour la PCA
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

# 3. Standardisation + PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
print("Variance expliquée (PC1–3) :", pca.explained_variance_ratio_)

# 4. Préparation des couleurs et ordre des catégories
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

df['final_type'] = df['final_type'].astype(pd.CategoricalDtype(type_order, ordered=True))

# 5. Comptage et pourcentages
counts = df['final_type'].value_counts().reindex(type_order).fillna(0)
percentages = (counts / counts.sum() * 100).round(1)




# 6. Création d'une figure en deux sous-plots (3D + bar)
fig = make_subplots(
    rows=2, cols=1,
    specs=[[{"type": "scene"}], [{"type": "xy"}]],
    row_heights=[0.7, 0.3],
    vertical_spacing=0.08,
    subplot_titles=("ACP 3D des profils Twitter", "Répartition des types (%)")
)


# 6a. Scatter 3D par type
for t in type_order:
    mask = df['final_type'] == t
    fig.add_trace(
        go.Scatter3d(
            x=X_pca[mask, 0], y=X_pca[mask, 1], z=X_pca[mask, 2],
            mode='markers',
            name=t,
            marker=dict(size=4, color=color_map[t], opacity=0.8, line=dict(width=0)),
            customdata=df.loc[mask, 'user_id']
        ),
        row=1, col=1
    )

# 6b. Diagramme en barres
fig.add_trace(
    go.Bar(
        x=type_order,
        y=percentages.values,
        marker_color=[color_map[t] for t in type_order],
        text=[f"{v}%" for v in percentages.values],
        textposition='outside',
        showlegend=False
    ),
    row=2, col=1
)

# 7. Annotation du total de comptes affichés
fig.add_annotation(
    text=f"Total comptes affichés : {len(df)}",
    xref='paper', yref='paper',
    x=0.01, y=0.95,
    showarrow=False,
    font=dict(size=14)
)

# 8. Layout et export
fig.update_layout(
    height=1000,
    margin=dict(l=0, r=0, t=60, b=0),
    title_x=0.5
)
fig.update_scenes(
    xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'
)

output_path = root / "visualisations" / "acp_visualisation.html"
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(
    str(output_path),
    include_plotlyjs='cdn'
)

# 9. Sauvegarde des objets PCA et scaler
models_dir = root / "models"
models_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(pca, models_dir / "pca_3d.joblib")
joblib.dump(scaler, models_dir / "scaler.joblib")
print(f"✅ Visualisation exportée dans : {output_path}")
print(f"✅ PCA et scaler sauvegardés dans : {models_dir}")
