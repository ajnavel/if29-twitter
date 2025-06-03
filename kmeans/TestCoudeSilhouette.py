import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Chargement des données
df = pd.read_csv("data/processed/user_profiles_with_scores.csv", dtype={"user_id": str})
df = df.fillna(0)

# Sélection des caractéristiques
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

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calcul des métriques pour différentes valeurs de k
inertias = []
silhouette_scores = []
k_range = range(2, 15)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Détection "manuelle" du coude (ex: ici on suppose k=4 est optimal visuellement ou par heuristique)
elbow_k = 4  # <-- Ajuste cette valeur selon le plot ou calcule-le

# Visualisation
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Nombre de clusters k')
ax1.set_ylabel('Inertie', color=color)
ax1.plot(k_range, inertias, 'o-', color=color, label='Inertie')
ax1.tick_params(axis='y', labelcolor=color)
ax1.axvline(x=elbow_k, color='black', linestyle='--', label=f"Coude à k={elbow_k}")

ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('Score silhouette', color=color)
ax2.plot(k_range, silhouette_scores, 's--', color=color, label='Silhouette')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Méthode du coude et score silhouette')
fig.tight_layout()
plt.legend(loc='upper right')
plt.show()