import pandas as pd
import numpy as np
import joblib
import os
import plotly.figure_factory as ff
import plotly.graph_objects as go
import time

# Démarrage du timer
start_time = time.time()

# === Chargement des modèles ===
scaler = joblib.load("models/scaler.joblib")
pca = joblib.load("models/pca_3d.joblib")
clf = joblib.load("models/svm_classifier.joblib")
le = joblib.load("models/label_encoder.joblib")

# === Chargement des nouvelles données (PAREIL A CHANGER JE TEST JUST PAS D'ERREUR) ===
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

# Transformation des données
X_new = new_df[FEATURES]
X_scaled = scaler.transform(X_new)
X_pca = pca.transform(X_scaled)[:, :18]  # même n_components que l'entraînement

# Prédictions
y_pred = clf.predict(X_pca)
y_pred_labels = le.inverse_transform(y_pred)
y_true_labels = new_df["final_type"]  # on suppose que les vraies classes sont présentes

# === HEATMAP 1 : Répartition des classes réelles dans chaque prédiction (%) ===
df_preds = pd.DataFrame({
    "true_type": y_true_labels,
    "predicted_type": y_pred_labels
})

# Table de contingence : prédiction vs classe réelle
crosstab_pred = pd.crosstab(df_preds["predicted_type"], df_preds["true_type"])

# Normalisation par ligne (chaque prédiction = 100%)
crosstab_percent_pred = crosstab_pred.div(crosstab_pred.sum(axis=1), axis=0) * 100

fig1 = go.Figure()
fig1.add_trace(go.Heatmap(
    z=crosstab_percent_pred.values,
    x=crosstab_percent_pred.columns.tolist(),  # Classes réelles (en X)
    y=[f"Classe prédite : {i}" for i in crosstab_percent_pred.index],  # Prédictions (en Y)
    colorscale="Oranges",
    zmin=0, zmax=100,
    colorbar=dict(title="% dans prédiction")
))

# Annotations
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

# === HEATMAP 2 : Répartition des prédictions dans chaque classe réelle (%) ===
# Table de contingence inversée : prédiction vs vraie classe
crosstab_real = pd.crosstab(df_preds["predicted_type"], df_preds["true_type"])

# Normalisation par colonne (chaque vraie classe = 100%)
crosstab_percent_by_class = crosstab_real.div(crosstab_real.sum(axis=0), axis=1) * 100

fig2 = go.Figure()
fig2.add_trace(go.Heatmap(
    z=crosstab_percent_by_class.values,
    x=crosstab_percent_by_class.columns.tolist(),  # Classes réelles (X)
    y=[f"Classe prédite : {i}" for i in crosstab_percent_by_class.index],  # Prédictions (Y)
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
    title="Répartition des prédictions dans chaque classe réelle (%) - Matrice de confusion normalisée",
    width=850,
    height=600
)

# === Sauvegarde des figures ===
os.makedirs("results", exist_ok=True)
fig1.write_html("results/svm_appli_prediction_distribution_new.html", include_plotlyjs="cdn")
fig2.write_html("results/svm_appli_confusion_heatmap_new.html", include_plotlyjs="cdn")

print("Heatmaps SVM pour nouvelles données générées et sauvegardées.")

# Fin du timer
end_time = time.time()
print(f"\n Temps total d'exécution : {end_time - start_time:.2f} secondes")
