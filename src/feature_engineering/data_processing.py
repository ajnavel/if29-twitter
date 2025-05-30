import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import numpy as np
import re
from tqdm import tqdm
from textstat import flesch_reading_ease, lexicon_count


from regles.def_spam_or_star import compute_spam_and_influencer_flags
from regles.def_media import compute_media_flag
from regles.def_type import def_type_tweet, def_type_user

###########################
###### 0. Paramètres ######
###########################

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "twitterdb"
COLLECTION = "tweets"
LIMIT = 0


###########################
## 1. Charger les tweets ##
###########################

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
col = db[COLLECTION]
projection = {
     "_id": 0,
    "text": 1,
    "created_at": 1,
    "entities.hashtags": 1,
    "entities.user_mentions": 1,
    "retweeted_status.id": 1,
    "user.id_str": 1,             
    "user.followers_count": 1,
    "user.friends_count": 1,
    "user.statuses_count": 1,
    "user.created_at": 1,
    "user.description": 1,   
    "favorite_count": 1,
    "retweet_count": 1,
    "id_str": 1,
    "user.listed_count": 1                   
}
cursor = col.find({}, projection).limit(LIMIT)
data = list(tqdm(cursor, total=LIMIT))
df = pd.json_normalize(data)
df = df.rename(columns={"user.description": "description"})

###########################
###### 2. Nettoyage #######
###########################

fill = {k: 0 for k in [
    "user.followers_count",
    "user.statuses_count",
    "favorite_count",
    "retweet_count",
    "user.listed_count"     
]}
fill.update({
    "user.friends_count": 1,
    "text": ""
})
df = df.fillna(fill)

for c in ['entities.hashtags','entities.user_mentions']:
    df[c] = df[c].apply(lambda x: x if isinstance(x,list) else [])
df["entities.hashtags"] = df["entities.hashtags"].apply(lambda x: x if isinstance(x, list) else [])
df["entities.user_mentions"] = df["entities.user_mentions"].apply(lambda x: x if isinstance(x, list) else [])

# 2bis) Calcul automatique des deux flags via le module externe
df["cond_influencer"], df["cond_star_spammer"] = compute_spam_and_influencer_flags(df)
df["is_media"] = compute_media_flag(df, description_col="description")

# Feature engineering 
print("Traitement des features...")
tqdm.pandas()

# Fonction pour extraire l'heure du tweet
def extract_hour(created_at):
    try:
        return datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y").hour
    except:
        return -1

# Features existantes
df["text_length"] = df["text"].progress_apply(len)
df["text_upper_ratio"] = df["text"].progress_apply(lambda t: sum(1 for c in t if c.isupper()) / len(t) if len(t) > 0 else 0)
df["text_exclam_ratio"] = df["text"].progress_apply(lambda t: t.count("!") / len(t) if len(t) > 0 else 0)
df["nb_hashtags"] = df["entities.hashtags"].progress_apply(len)
df["nb_mentions"] = df["entities.user_mentions"].progress_apply(len)
df["is_retweet"] = df["retweeted_status.id"].apply(lambda x: 1 if pd.notna(x) else 0)
df["hashtag_spam_score"] = df["entities.hashtags"].progress_apply(
    lambda x: len([h for h in x if re.search(r'(follow|like|retweet|win|free|giveaway)', h['text'], re.IGNORECASE)]) / len(x) if len(x) > 0 else 0
)

df["mention_repetition_score"] = df["entities.user_mentions"].progress_apply(
    lambda x: 1 if len(x) > 3 else 0  # Plus de 3 mentions = suspect
)

# Analyse sémantique basique
if 'user.id_str' in df.columns:
    df["text_repetition_score"] = df.groupby("user.id_str")["text"].transform(
        lambda x: x.duplicated().sum() / len(x) if len(x) > 0 else 0
    )
elif 'user.id' in df.columns:
    df["text_repetition_score"] = df.groupby("user.id")["text"].transform(
        lambda x: x.duplicated().sum() / len(x) if len(x) > 0 else 0
    )
else:
    df["text_repetition_score"] = 0
    print("Aucun identifiant utilisateur trouvé (user.id ou user.id_str)")

df["text_readability"] = df["text"].progress_apply(
    lambda t: flesch_reading_ease(t) if len(t) > 10 else 60  # 60 = valeur par défaut pour texte court
)

df["text_complexity"] = df["text"].progress_apply(
    lambda t: lexicon_count(t) / len(t.split()) if len(t.split()) > 0 else 0
)

# Calcul de l'âge du compte (en jours)
def account_age(created_at):
    try:
        account_date = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
        tweet_date = datetime.strptime("2018-12-31", "%Y-%m-%d")  # Fin 2018 comme référence vu qu'on a pris début 2019
        return (tweet_date - account_date).days
    except:
        return np.nan

df["account_age_days"] = df["user.created_at"].progress_apply(account_age)

# Ratio followers/tweets (activité suspecte)
df["followers_per_tweet"] = df["user.followers_count"] / (df["user.statuses_count"].replace(0, 1))

# Nouveau calcul du ratio followers/friends avec log pour réduire l'échelle
df["log_ratio_followers_friends"] = np.log10(
    (df["user.followers_count"] + 1) / (df["user.friends_count"] + 1))
df["log_ratio_followers_friends"].replace([np.inf, -np.inf], np.nan, inplace=True)

# Heure du tweet et régularité
df["tweet_hour"] = df["created_at"].progress_apply(extract_hour)
df["late_night_tweet"] = ((df["tweet_hour"] >= 0) & (df["tweet_hour"] <= 5)).astype(int)

# Score d'engagement
df["engagement_rate"] = (df["favorite_count"] + df["retweet_count"]) / (df["user.followers_count"] + 1)

# Calcul du score atypique avec pondération
weights = {
    "low_followers": 1,          # < 10 followers
    "high_followers": 1,         # > 10k followers
    "extreme_ratio": 2,          # ratio followers/friends extrême
    "high_mentions": 1,          # > 5 mentions
    "high_hashtags": 1,          # > 5 hashtags
    "spam_hashtags": 2,          # hashtags spammy
    "repetitive_mentions": 2,    # mêmes mentions répétées
    "high_exclam": 1,            # beaucoup de !
    "high_upper": 1,             # beaucoup de majuscules
    "late_night": 1,             # tweets très tard
    "low_engagement": 2,         # engagement faible
    "repetitive_content": 3,     # contenu répétitif
    "suspicious_activity": 3,    # followers/tweets suspect
    "low_readability": 2,        # texte difficile à lire
    "high_readability": 1,       # texte anormalement simple
    "low_complexity": 2          # vocabulaire répétitif
}

df["score_atypique"] = 0
df["score_atypique"] += (df["user.followers_count"] < 10).astype(int) * weights["low_followers"]
df["score_atypique"] += (df["user.followers_count"] > 10000).astype(int) * weights["high_followers"]
df["score_atypique"] += ((df["log_ratio_followers_friends"] < -2) | (df["log_ratio_followers_friends"] > 2)).astype(int) * weights["extreme_ratio"]
df["score_atypique"] += (df["nb_mentions"] > 5).astype(int) * weights["high_mentions"]
df["score_atypique"] += (df["nb_hashtags"] > 5).astype(int) * weights["high_hashtags"]
df["score_atypique"] += (df["hashtag_spam_score"] > 0.5).astype(int) * weights["spam_hashtags"]
df["score_atypique"] += (df["mention_repetition_score"] > 0).astype(int) * weights["repetitive_mentions"]
df["score_atypique"] += (df["text_exclam_ratio"] > 0.05).astype(int) * weights["high_exclam"]
df["score_atypique"] += (df["text_upper_ratio"] > 0.5).astype(int) * weights["high_upper"]
df["score_atypique"] += (df["late_night_tweet"] > 0).astype(int) * weights["late_night"]
df["score_atypique"] += (df["engagement_rate"] < 0.001).astype(int) * weights["low_engagement"]
df["score_atypique"] += (df["text_repetition_score"] > 0.3).astype(int) * weights["repetitive_content"]
df["score_atypique"] += (df["followers_per_tweet"] > 100).astype(int) * weights["suspicious_activity"]
df["score_atypique"] += (df["text_readability"] < 30).astype(int) * weights["low_readability"]
df["score_atypique"] += (df["text_readability"] > 90).astype(int) * weights["high_readability"]
df["score_atypique"] += (df["text_complexity"] < 0.5).astype(int) * weights["low_complexity"]

# Normalisation du score sur 100
max_score = sum(weights.values())
df["score_atypique"] = (df["score_atypique"] / max_score) * 100

# Marquage comme atypique si le score dépasse 50%
seuil_raw = 3
seuil_pct = (seuil_raw / max_score) * 100  
df["label"] = (df["score_atypique"] >= seuil_pct).astype(int)

# Application de la classification
df["type"] = df.apply(lambda row: def_type_tweet(row, seuil_pct), axis=1)

# Renommer la colonne d'ID utilisateur pour plus de clarté
df = df.rename(columns={"user.id_str": "user_id"})

###################################
## 3. Agrégation par utilisateur ##
###################################

user_features = [
    "user.followers_count",
    "user.friends_count",
    "user.statuses_count",
    "account_age_days",
    "log_ratio_followers_friends",
    "followers_per_tweet",
    "is_media" 
]

# On garde les premières valeurs pour les features stables (qui ne changent pas entre tweets d'un même utilisateur)
user_stats = df.groupby('user_id')[user_features].first()

# On calcule les moyennes pour les features qui varient entre tweets
variable_features = [
    "text_length",
    "text_upper_ratio",
    "text_exclam_ratio",
    "nb_hashtags",
    "nb_mentions",
    "is_retweet",
    "hashtag_spam_score",
    "mention_repetition_score",
    "text_repetition_score",
    "text_readability",
    "text_complexity",
    "tweet_hour",
    "late_night_tweet",
    "engagement_rate",
    "score_atypique",
    "label"
]

user_variable_stats = df.groupby('user_id')[variable_features].mean().add_prefix('mean_')

# On calcule aussi le pourcentage de tweets atypiques par utilisateur
user_label_stats = df.groupby('user_id')['label'].agg(
    total_tweets='count',
    atypical_tweets='sum'
).reset_index()
user_label_stats['pct_atypical'] = (user_label_stats['atypical_tweets'] / user_label_stats['total_tweets']) * 100

# Fusion des données utilisateur
user_df = pd.merge(user_stats, user_variable_stats, left_index=True, right_index=True)
user_df = pd.merge(user_df, user_label_stats, on='user_id')



#################################################
###### 4. Calcul du score final utilisateur######
#################################################


# Pondération entre le score moyen et le pourcentage de tweets atypiques
user_df['final_score'] = (user_df['mean_score_atypique'] * 0.7) + (user_df['pct_atypical'] * 0.3)

# Seuil pour déterminer si un utilisateur est atypique (ajustable)
final_seuil = 30  # par exemple 30/100
user_df['final_label'] = (user_df['final_score'] >= final_seuil).astype(int)

#################################################
### 5. Classification finale des utilisateurs ###
#################################################

user_df["type_user"] = user_df.apply(def_type_user, axis=1)


# 6. Export des données utilisateur
user_df.to_csv('data/processed/user_profiles_with_scores.csv', index=True)
print("Fichier des profils utilisateur exporté dans data/processed/user_profiles_with_scores.csv")

# 7. Export des données de tweets
tweet_features = [
    "id_str", "user_id", "text_length", "text_upper_ratio", "text_exclam_ratio",
    "nb_hashtags", "nb_mentions", "is_retweet", "tweet_hour", 
    "user.followers_count", "user.friends_count", "user.statuses_count",
    "log_ratio_followers_friends", "hashtag_spam_score", 
    "mention_repetition_score", "text_repetition_score", 
    "account_age_days", "followers_per_tweet", "engagement_rate",
    "late_night_tweet", "score_atypique", "label", "type"
]

df[tweet_features].to_csv("data/processed/tweet_features.csv", index=False)
print("Fichier des features par tweet exporté dans data/processed/tweet_features.csv")