# pca_visualization.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import joblib

# 1. Chargement des données 
df = pd.read_csv(
    "data/processed/user_profiles_with_scores.csv",
    dtype={"user_id": str}
)

# 2. Sélection des features
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

# 3. Imputation éventuelle
df = df.fillna(0)

# 4. Standardisation + PCA
X = df[FEATURES]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=3)
X_pca_full = pca.fit_transform(X_scaled)
print("Variance expliquée (PC1–3) :", pca.explained_variance_ratio_)

# 4.b. Récupérer les poids des features pour chaque composante
pca_components = pd.DataFrame(
    pca.components_,
    columns=FEATURES,
    index=["PC1", "PC2", "PC3"]
).T

# Affichage des contributions triées par importance pour chaque PC
for pc in pca_components.columns:
    print(f"\nTop features pour {pc} :")
    print(pca_components[pc].abs().sort_values(ascending=False).head(5))

# 5. Préparation et export du nuage 3D
export_df = pd.DataFrame({
    "PC1": X_pca_full[:, 0],
    "PC2": X_pca_full[:, 1],
    "PC3": X_pca_full[:, 2],
    "type": df["final_type"],
    "user_id": df["user_id"]
})

color_map = {
    "normal": "#2ecc71",  
    "bot": "#3498db",
    "spam": "#e74c3c",
    "influenceur": "#f39c12",
    "spam_star": "#c0392b",  
    "atypique_generique": "#9b59b6",
    "media": "#e67e22"
}
type_order = ["normal","bot","spam","influenceur","media","spam_star","atypique_generique"]

fig = px.scatter_3d(
    export_df, x="PC1", y="PC2", z="PC3",
    color="type",
    color_discrete_map=color_map,
    category_orders={"type": type_order},
    custom_data=["user_id"],
    title="ACP 3 dimensions des comptes Twitter",
    width=1200, height=800
)

# Personnalisation supplémentaire
fig.update_traces(
    marker=dict(size=4, opacity=0.8, line=dict(width=0)),
    selector=dict(mode='markers')
)

# 6. Script JS pour l'interactivité des ID
post_script = """
var gd = document.getElementsByClassName('plotly-graph-div')[0];
gd.on('plotly_click', function(data) {
  var userId = data.points[0].customdata[0];
  prompt('User ID (CTRL+C pour copier)', userId);
});
"""

# 7. Export final
html_path = "visualisations/acp_vizualisation.html"
fig.write_html(
    html_path,
    include_plotlyjs="cdn",
    post_script=post_script
)

joblib.dump(pca, "models/pca_3d.joblib")
joblib.dump(scaler, "models/scaler.joblib")
print("PCA et scaler sauvegardés.")
