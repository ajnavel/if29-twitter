import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# 1. Chargement des données
df = pd.read_csv("../data/processed/user_profiles_with_scores.csv", dtype={"user_id": str})

# 2. Selection des variables pertinentes pour le clustering
FEATURES = [
  "user.followers_count",
  "user.friends_count",
  "user.statuses_count",
#   "account_age_days", retiré car ne marche pas
  "log_ratio_followers_friends",
  "followers_per_tweet",
  "is_media",
  "mean_text_length",
  "mean_text_upper_ratio",
  "mean_text_exclam_ratio",
  "mean_nb_hashtags",
  "mean_nb_mentions",
  "mean_is_retweet",
  "mean_hashtag_spam_score",
  "mean_mention_repetition_score",
  "mean_text_repetition_score",
  "mean_text_readability",
  "mean_text_complexity",
  "mean_tweet_hour",
  "mean_late_night_tweet",
  "mean_engagement_rate"
]

X = df[FEATURES]

# 3. Normalisation des variables

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Méthode du coude pour obtenir le nombre de clusters optimal
inertias = []
silhouettes = []
K_range = range(2, 7)
for k in K_range:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X_scaled, model.labels_))

# Visualisation
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(K_range, inertias, marker='o')
plt.title("Méthode du coude")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Inertie")

plt.subplot(1,2,2)
plt.plot(K_range, silhouettes, marker='s')
plt.title("Silhouette score")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Score")

plt.tight_layout()
plt.show()

# 5. Application de KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_scaled)

# 6. Ajout des labels de clusters au DataFrame
df['cluster'] = kmeans.labels_

# 7. Analyse des clusters
print("\nCluster distribution:")
print(df['cluster'].value_counts())
cluster_summary = df.groupby('cluster')[FEATURES].mean().round(2)
print(cluster_summary)

# 8. Application d'un PCA pour la visualisation des clusters
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['pca1'], df['pca2'] = X_pca[:, 0], X_pca[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='viridis', alpha=0.7)
plt.title('Clusters avec PCA')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(title='Cluster')
plt.show()



