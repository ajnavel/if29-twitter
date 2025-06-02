import pandas as pd
import joblib
import os
import plotly.graph_objects as go

# --- Chargement des modèles ---
scaler = joblib.load("models/scaler.joblib")
pca = joblib.load("models/pca.joblib")
kmeans = joblib.load("models/kmeans_from_agglo.pkl")

# --- Chargement des nouvelles données (A CHANGER ABSOLUMENT JE TEST JUSTE AVCE CE QU'ON A) ---
new_df = pd.read_csv("data/processed/user_profiles_with_scores.csv", dtype={"user_id": str})
new_df = new_df.fillna(0)

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

# Extraction features + transformation
X_new = new_df[FEATURES]
X_scaled = scaler.transform(X_new)
X_pca = pca.transform(X_scaled)

# Prédiction clusters
cluster_labels = kmeans.predict(X_pca)
new_df["cluster"] = cluster_labels

# --- Création des heatmaps ---

# Cross tab : cluster vs classe réelle (final_type)
crosstab = pd.crosstab(new_df["cluster"], new_df["final_type"])
crosstab_percent = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
crosstab_percent_by_class = crosstab.div(crosstab.sum(axis=0), axis=1) * 100

# Heatmap 1 : Répartition des classes dans chaque cluster (%)
fig1 = go.Figure()
fig1.add_trace(go.Heatmap(
    z=crosstab_percent.values,
    x=crosstab.columns.tolist(),
    y=[f"Cluster {i}" for i in crosstab.index],
    colorscale="Oranges",
    zmin=0, zmax=100,
    colorbar=dict(title="% dans cluster")
))
for i, row in enumerate(crosstab_percent.values):
    for j, val in enumerate(row):
        fig1.add_annotation(
            x=crosstab.columns[j],
            y=f"Cluster {crosstab.index[i]}",
            text=f"{val:.1f}%",
            showarrow=False,
            font=dict(color="black")
        )
fig1.update_yaxes(autorange="reversed")
fig1.update_layout(
    title="Répartition des classes dans chaque cluster (%) - nouvelles données",
    width=800,
    height=600
)

# Heatmap 2 : Répartition des clusters dans chaque classe réelle (%)
fig2 = go.Figure()
fig2.add_trace(go.Heatmap(
    z=crosstab_percent_by_class.values,
    x=crosstab_percent_by_class.columns.tolist(),
    y=[f"Cluster {i}" for i in crosstab_percent_by_class.index],
    colorscale="Blues",
    zmin=0, zmax=100,
    colorbar=dict(title="% dans classe")
))
for i, row in enumerate(crosstab_percent_by_class.values):
    for j, val in enumerate(row):
        fig2.add_annotation(
            x=crosstab_percent_by_class.columns[j],
            y=f"Cluster {crosstab_percent_by_class.index[i]}",
            text=f"{val:.1f}%",
            showarrow=False,
            font=dict(color="black")
        )
fig2.update_yaxes(autorange="reversed")
fig2.update_layout(
    title="Répartition des clusters dans chaque classe réelle (%) - nouvelles données",
    width=800,
    height=600
)

# --- Sauvegarde heatmaps ---
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
fig1.write_html(os.path.join(output_dir, "heatmap_classes_par_cluster_new.html"), include_plotlyjs="cdn")
fig2.write_html(os.path.join(output_dir, "heatmap_clusters_par_classe_new.html"), include_plotlyjs="cdn")

print("Heatmaps créées et enregistrées dans le dossier 'results'")
