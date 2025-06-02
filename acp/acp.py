# pca_visualization.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
import plotly.express as px
import plotly.io as pio
import seaborn as sns

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
pca = PCA(n_components=18)
X_pca_full = pca.fit_transform(X_scaled)
eigenvalues = pca.explained_variance_ratio_*100 # pour avoir en % les valeurs propres
print("Variance expliquée (PC1–3) :", eigenvalues)


plt.figure(figsize=(10, 4))
bars = plt.bar(range(1, len(eigenvalues) + 1), eigenvalues, alpha=0.5, align='center')
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='-', color='r')
plt.ylabel('Pourcentage de variance expliquée')
plt.xlabel('Composantes principales')
plt.title('Pourcentage de variance expliquée')
for bar, percentage in zip(bars, eigenvalues):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{percentage:.2f}%',
             ha='center', va='bottom', color='black', fontsize=8)
plt.show()

plt.figure(figsize=(10, 4))
plt.bar(range(1, len(eigenvalues) + 1), np.cumsum(eigenvalues), alpha=0.5, align='center')
plt.ylabel('Pourcentage de variance cumulée')
plt.xlabel('Composantes principales')
plt.title('Pourcentage de variance cumulée')
plt.axhline(y=80, color='r', linestyle='--')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), cmap='coolwarm_r', annot=True, fmt=".2f", square=True)
plt.title('Matrice des Corrélations')
plt.show()

def plot_pca_var_arrow(acp, data, ax1=1, ax2=2):
    dim1, dim2 = ax1 - 1, ax2 - 1

    # Coordonnées des variables dans le plan factoriel
    coord_var = acp.components_.T * np.sqrt(acp.explained_variance_)

    plt.figure(figsize=(8, 8))
    plt.gca().add_artist(plt.Circle((0, 0), 1, color='gray', fill=False))

    for i, (x, y) in enumerate(coord_var[:, [dim1, dim2]]):
        arrow = FancyArrowPatch((0, 0), (x, y), color='red', arrowstyle='->', mutation_scale=10)
        plt.gca().add_patch(arrow)
        plt.text(x * 1.1, y * 1.1, data.columns[i], color='black', ha='center', va='center')

    # Axes et titres
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.xlabel(f"Dimension {ax1} ({acp.explained_variance_ratio_[dim1]*100:.2f}%)")
    plt.ylabel(f"Dimension {ax2} ({acp.explained_variance_ratio_[dim2]*100:.2f}%)")
    plt.title(f"Cercle des variables : Dim {ax1} vs Dim {ax2}")
    plt.axhline(0, color='grey', lw=1)
    plt.axvline(0, color='grey', lw=1)
    plt.grid(True)
    plt.gca().set_aspect('equal')
    plt.show()


# Liste des couples d’axes à tracer
axes_to_plot = [(1, 2), (1, 3), (2, 3)]

for ax1, ax2 in axes_to_plot:
    plot_pca_var_arrow(pca, X, ax1=ax1, ax2=ax2)

# 4.b. Récupérer les poids des features pour chaque composante
pca_components = pd.DataFrame(
    pca.components_,
    columns=FEATURES,
    index=[f"PC{i+1}" for i in range(18)]
).T


# Affichage des contributions triées par importance pour chaque PC
for pc in pca_components.columns:
    loadings = pca_components[pc]
    squared_loadings = loadings ** 2
    contributions = squared_loadings / squared_loadings.sum() * 100
    print(f"\nTop contributions pour {pc} :")
    print(contributions.sort_values(ascending=False).head(5))


# New code :
pc_names = [f'PC{i+1}' for i in range(pca.n_components_)]

# Calcul des loadings = corrélations entre variables et axes (si données centrées-réduites)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)  # shape (n_features, n_components)
loadings_df = pd.DataFrame(loadings, index=X.columns, columns=pc_names)

# Contributions en pourcentage = carrés des loadings normalisés colonne par colonne
squared_loadings = loadings_df ** 2
contributions_df = squared_loadings.div(squared_loadings.sum(axis=0), axis=1) * 100

# Fusion en un seul tableau multi-index
final_df = pd.concat({
    'Correlation': loadings_df,
    'Contribution (%)': contributions_df
}, axis=1)

# Affichage propre
pd.set_option('display.max_columns', None)
print(final_df.round(3))    


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

# Graphe des individus en 2D
# Forcer l'ouverture dans le navigateur
pio.renderers.default = "browser"

def plot_pca_2d(df, x_axis, y_axis, color_col="type", id_col="user_id",
                color_map=None, type_order=None, width=1000, height=700):
    fig = px.scatter(
        df,
        x=x_axis,
        y=y_axis,
        color=color_col,
        color_discrete_map=color_map,
        category_orders={color_col: type_order},
        custom_data=[id_col],
        title=f"ACP - Plan {x_axis} vs {y_axis}",
        width=width,
        height=height
    )

    fig.update_traces(
        marker=dict(size=5, opacity=0.7, line=dict(width=0)),
        selector=dict(mode='markers'),
        hovertemplate="<b>ID</b>: %{customdata[0]}<br><b>" + x_axis + "</b>: %{x:.2f}<br><b>" + y_axis + "</b>: %{y:.2f}<extra></extra>"
    )

    fig.update_layout(
        xaxis_title=x_axis,
        yaxis_title=y_axis,
        legend_title="Type"
    )

    fig.show()

# Exécution
plot_pca_2d(export_df, "PC1", "PC2", color_map=color_map, type_order=type_order)
plot_pca_2d(export_df, "PC1", "PC3", color_map=color_map, type_order=type_order)
plot_pca_2d(export_df, "PC2", "PC3", color_map=color_map, type_order=type_order)


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

# Liste des types définie
type_order = ["normal", "bot", "spam", "influenceur", "media", "spam_star", "atypique_generique"]

# Compter les individus par type
available_types = df["final_type"].unique()
valid_type_order = [t for t in type_order if t in available_types]

type_counts = df["final_type"].value_counts().loc[valid_type_order]

# Création du barplot
plt.figure(figsize=(10, 6))
sns.barplot(x=type_counts.index, y=type_counts.values, palette="Set2")

plt.xlabel("Type de compte")
plt.ylabel("Nombre d'individus")
plt.title("Répartition des individus par type de compte")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
