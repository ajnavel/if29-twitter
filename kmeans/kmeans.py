import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# --- Paramètres ---
N_COMPONENTS = 18
K_CLUSTERS = 2

# 1. Chargement des données
df = pd.read_csv("data/processed/user_profiles_with_scores.csv", dtype={"user_id": str})
df = df.fillna(0)

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
X = df[FEATURES]
y_true_text = df["final_type"]

# 3. Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. PCA
pca = PCA(n_components=N_COMPONENTS)
X_pca = pca.fit_transform(X_scaled)

# 5. KMeans Clustering
kmeans = KMeans(n_clusters=K_CLUSTERS, init='k-means++', n_init='auto', random_state=42)
cluster_labels = kmeans.fit_predict(X_pca)
df["cluster"] = cluster_labels

# 6. Matrices de correspondance
crosstab = pd.crosstab(df["cluster"], y_true_text)
crosstab_percent = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
crosstab_percent_by_class = crosstab.div(crosstab.sum(axis=0), axis=1) * 100

# 7. Heatmap 1 : Répartition des classes dans chaque cluster
fig1 = go.Figure()
fig1.add_trace(go.Heatmap(
    z=crosstab_percent.values,
    x=crosstab.columns.tolist(),
    y=[f"Cluster {i}" for i in crosstab.index],
    colorscale="Oranges",
    zmin=0, zmax=100,
    colorbar=dict(title="% dans cluster")
))

# Annotations
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
    title="Répartition des classes dans chaque cluster (%)",
    width=800,
    height=600
)

# 8. Heatmap 2 : Répartition des clusters dans chaque classe réelle
fig2 = go.Figure()
fig2.add_trace(go.Heatmap(
    z=crosstab_percent_by_class.values,
    x=crosstab_percent_by_class.columns.tolist(),  # noms des classes simples (pas "Cluster ...")
    y=[f"Cluster {i}" for i in crosstab_percent_by_class.index],  # affichage comme "Cluster 0", "Cluster 1", ...
    colorscale="Blues",
    zmin=0, zmax=100,
    colorbar=dict(title="% dans classe")
))

# Annotations
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
    title="Répartition des clusters dans chaque classe réelle (%)",
    width=800,
    height=600
)


# 9. Visualisation 3D (ACP)
if N_COMPONENTS < 3:
    raise ValueError("Pour la visualisation 3D, PCA doit avoir au moins 3 composantes.")

viz_df = pd.DataFrame(X_pca[:, :3], columns=["PC1", "PC2", "PC3"])
viz_df["cluster"] = cluster_labels.astype(str)
viz_df["true_type"] = y_true_text
viz_df["user_id"] = df["user_id"]

fig_3d = px.scatter_3d(
    viz_df, x="PC1", y="PC2", z="PC3",
    color="cluster",
    hover_data=["user_id", "true_type"],
    title="K-Means initialisé (ACP)",
    width=1200, height=800
)

# 10. Export des visualisations
output_dir = "C:/Users/yohan/OneDrive/Documents/UTT Cours/ISI2/IF29/projet/visualisations"
os.makedirs(output_dir, exist_ok=True)

# Heatmaps
fig1.write_html(os.path.join(output_dir, "heatmap_classes_par_cluster.html"), include_plotlyjs="cdn")
fig2.write_html(os.path.join(output_dir, "heatmap_clusters_par_classe.html"), include_plotlyjs="cdn")
print("Deux heatmaps exportées avec succès.")

# Visualisation 3D
fig_3d.write_html(
    os.path.join(output_dir, "kmeans_custom_visualisation.html"),
    include_plotlyjs="cdn",
    post_script="""
    var gd = document.getElementsByClassName('plotly-graph-div')[0];
    gd.on('plotly_click', function(data) {
      var userId = data.points[0].customdata[0];
      prompt('User ID (CTRL+C pour copier)', userId);
    });
    """
)
print("Visualisation 3D exportée avec succès.")

# 11. ARI + sauvegarde modèle
y_true_encoded = y_true_text.factorize()[0]
ari = adjusted_rand_score(y_true_encoded, cluster_labels)
print(f"\nAdjusted Rand Index (ARI) : {ari:.3f}")

joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(pca, "models/pca.joblib")
joblib.dump(kmeans, "models/kmeans.joblib")
print("Modèle, Scaler et PCA sauvegardés.")

# 12. Barplot clusters
sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
cluster_counts = df['cluster'].value_counts().sort_index()
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, hue=cluster_counts.index, palette="Set2", legend=False)
plt.xlabel("Cluster")
plt.ylabel("Nombre d'individus")
plt.title("Répartition du nombre d'individus par cluster")
plt.tight_layout()
plt.show()
