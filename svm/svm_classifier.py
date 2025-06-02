"""svm_classifier_with_pca.py : SVM avec sélection du nombre de composantes principales d'ACP."""

import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import time
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Démarrage du timer
start_time = time.time()

# Chargement des données
df = pd.read_csv("data/processed/user_profiles_with_scores.csv", dtype={"user_id": str})
df = df.fillna(0)

# Définition des features
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
y_text = df["final_type"]

# Encodage des labels
le = LabelEncoder()
y = le.fit_transform(y_text)
print("Classes =", list(le.classes_))

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
test_indices = X_test.index

# Chargement du scaler et de la PCA globaux
scaler = joblib.load("models/scaler.joblib")
pca = joblib.load("models/pca_3d.joblib")

# Choix du nombre de composantes principales à utiliser
n_components = 18

# Transformation des données
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_pca = pca.transform(X_train_scaled)[:, :n_components]
X_test_pca = pca.transform(X_test_scaled)[:, :n_components]

# Entraînement du SVM
clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
clf.fit(X_train_pca, y_train)

# Prédictions
y_pred = clf.predict(X_test_pca)

# Évaluation
print("Accuracy :", accuracy_score(y_test, y_pred))
print("\nClassification report :")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
print("Confusion matrix :")
print(confusion_matrix(y_test, y_pred))

# === HEATMAP 1 : Répartition des classes réelles dans chaque prédiction (matrice de confusion) ===
df_preds = pd.DataFrame({
    "true_type": le.inverse_transform(y_test),
    "predicted_type": le.inverse_transform(y_pred)
})

crosstab_pred = pd.crosstab(df_preds["predicted_type"], df_preds["true_type"])
crosstab_percent_pred = crosstab_pred.div(crosstab_pred.sum(axis=1), axis=0) * 100

fig1 = go.Figure()
fig1.add_trace(go.Heatmap(
    z=crosstab_percent_pred.values,
    x=crosstab_percent_pred.columns.tolist(),
    y=[f"Classe prédite : {i}" for i in crosstab_percent_pred.index],
    colorscale="Oranges",
    zmin=0, zmax=100,
    colorbar=dict(title="% dans prédiction")
))

for i, row in enumerate(crosstab_percent_pred.values):
    for j, val in enumerate(row):
        fig1.add_annotation(
            x=crosstab_percent_pred.columns[j],
            y=f"Classe prédite : {crosstab_percent_pred.index[i]}",
            text=f"{val:.1f}%",
            showarrow=False,
            font=dict(color="black")
        )

fig1.update_yaxes(autorange="reversed")
fig1.update_layout(
    title="Répartition des classes réelles dans chaque prédiction (%)",
    width=850,
    height=600
)

# === HEATMAP 2 corrigée : % des prédictions dans chaque classe réelle (normalisation colonne) ===
crosstab_real = pd.crosstab(df_preds["predicted_type"], df_preds["true_type"])
crosstab_percent_by_class = crosstab_real.div(crosstab_real.sum(axis=0), axis=1) * 100

fig2 = go.Figure()
fig2.add_trace(go.Heatmap(
    z=crosstab_percent_by_class.values,
    x=crosstab_percent_by_class.columns.tolist(),  # Classes réelles (en X)
    y=[f"Classe prédite : {i}" for i in crosstab_percent_by_class.index],  # Classes prédites (en Y)
    colorscale="Blues",
    zmin=0, zmax=100,
    colorbar=dict(title="% dans vraie classe")
))

# Annotations
for i, row in enumerate(crosstab_percent_by_class.values):
    for j, val in enumerate(row):
        fig2.add_annotation(
            x=crosstab_percent_by_class.columns[j],
            y=f"Classe prédite : {crosstab_percent_by_class.index[i]}",
            text=f"{val:.1f}%",
            showarrow=False,
            font=dict(color="black")
        )

fig2.update_yaxes(autorange="reversed")
fig2.update_layout(
    title="Répartition des prédictions dans chaque classe réelle (%), Matrice de confusion normalisée",
    width=850,
    height=600
)



# === Visualisation 3D si possible ===
if n_components >= 3:
    viz_df = pd.DataFrame(X_test_pca[:, :3], columns=["PC1", "PC2", "PC3"])
    viz_df["predicted_type"] = le.inverse_transform(y_pred)
    viz_df["true_type"] = le.inverse_transform(y_test)
    viz_df["user_id"] = df.loc[test_indices, "user_id"].values

    fig_3d = px.scatter_3d(
        viz_df,
        x="PC1", y="PC2", z="PC3",
        color="predicted_type",
        hover_data=["user_id", "true_type"],
        title=f"SVM - Visualisation 3D des classes prédites ({n_components} CP utilisées)",
        width=1200,
        height=800
    )

    fig_3d.write_html("visualisations/svm_prediction_visualisation.html", include_plotlyjs="cdn")
    print("Visualisation 3D exportée.")
else:
    print(f"Visualisation 3D non générée (n_components = {n_components})")

# === Exports ===
os.makedirs("visualisations", exist_ok=True)
fig1.write_html("visualisations/svm_heatmap_pred_vs_true.html", include_plotlyjs="cdn")
fig2.write_html("visualisations/svm_heatmap_true_vs_pred.html", include_plotlyjs="cdn")

print("Heatmaps exportées.")

# Fin du timer
end_time = time.time()
print(f"\nTemps total d'exécution : {end_time - start_time:.2f} secondes")

# Sauvegarde du modèle
joblib.dump(clf, "models/svm_classifier.joblib")
joblib.dump(le, "models/label_encoder.joblib")
print("Modèle SVM et encodage des labels sauvegardés.")
