
"""svm_classifier.py : entraînement et évaluation d'un SVM supervisé."""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


"""
C'est un "début" de SVM supervisé pour classifier les utilisateurs 
Twitter en fonction de leurs caractéristiques.
Faudra approfondir 
"""
# 1. Chargement des données
df = pd.read_csv("data/processed/user_profiles_with_scores.csv", dtype={"user_id": str})

# 2. Sélection des features et du label
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
y_text = df["final_type"] # en gros le label à prédire 

# 3. Encodage des labels
le = LabelEncoder()
y = le.fit_transform(y_text)
print("Classes =", list(le.classes_))

# 4. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 6. Entraînement du SVM
clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
clf.fit(X_train_scaled, y_train)

# 7. Évaluation
y_pred = clf.predict(X_test_scaled)
print("Accuracy :", accuracy_score(y_test, y_pred))
print("\nClassification report :")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion matrix :")
print(confusion_matrix(y_test, y_pred))