"""svm_classifier.py : entraînement et évaluation d'un SVM supervisé."""
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.figure_factory as ff

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Chargement des données
df = pd.read_csv("data/processed/user_profiles_with_scores.csv", dtype={"user_id": str})
df = df.fillna(0)

# Features + Label
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
# On garde aussi les indices correspondants
test_indices = X_test.index

# Chargement du scaler et de la PCA globaux
scaler = joblib.load("models/scaler.joblib")
pca = joblib.load("models/pca_3d.joblib")

# Transformation des données
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entraînement du SVM
clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
clf.fit(X_train_scaled, y_train)

# Prédictions
y_pred = clf.predict(X_test_scaled)

# Évaluation
print("Accuracy :", accuracy_score(y_test, y_pred))
print("\nClassification report :")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
print("Confusion matrix :")
print(confusion_matrix(y_test, y_pred))

# Matrice de confusion (normalisée)
cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_percent = cm_normalized * 100

fig_cm = ff.create_annotated_heatmap(
    z=cm_percent,
    x=list(le.classes_),
    y=list(le.classes_),
    colorscale='Blues',
    showscale=True,
    annotation_text=[[f"{val:.1f}%" for val in row] for row in cm_percent],
)
fig_cm.update_layout(
    title='Matrice de confusion normalisée (SVM) en %',
    xaxis_title='Prédit',
    yaxis_title='Vrai',
    width=700,
    height=700
)
fig_cm.write_html("visualisations/svm_visualisation.html", include_plotlyjs="cdn")
print("Matrice de confusion exportée.")

# Visualisation 3D avec la PCA globale
X_test_pca = pca.transform(X_test_scaled)
viz_df = pd.DataFrame(X_test_pca, columns=["PC1", "PC2", "PC3"])
viz_df["predicted_type"] = le.inverse_transform(y_pred)
viz_df["true_type"] = le.inverse_transform(y_test)
viz_df["user_id"] = df.loc[test_indices, "user_id"].values

fig_3d = px.scatter_3d(
    viz_df,
    x="PC1", y="PC2", z="PC3",
    color="predicted_type",
    hover_data=["user_id", "true_type"],
    title="SVM - Visualisation 3D des classes prédites (ACP globale)",
    width=1200,
    height=800
)

viz_path = "visualisations/svm_prediction_visualisation.html"
fig_3d.write_html(viz_path, include_plotlyjs="cdn")
print(f"Visualisation 3D exportée dans : {viz_path}")
