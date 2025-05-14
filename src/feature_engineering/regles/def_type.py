import numpy as np


def def_type_tweet(row, seuil_pct):
    """
    Catégorise un TWEET selon ses features et un seuil de score atypique.
    """
    # 0) Priorité aux médias
    if row.get("is_media", 0) == 1:
        return "media"

    # 1) Comptes normaux sous le seuil
    if row.get("score_atypique", 0) < seuil_pct:
        return "normal"

    # 2) Overrides influenceur / star_spammer
    if row.get("cond_influencer", 0) == 1:
        return "influenceur"
    if row.get("cond_star_spammer", 0) == 1:
        return "spam_star"

    # 3) Détection bots vs spammeurs
    is_bot = (
        (row.get("text_repetition_score", 0) > 0.5) or
        (row.get("log_ratio_followers_friends", 0) < -2) or
        (row.get("late_night_tweet", 0) == 1) or
        (row.get("followers_per_tweet", 0) > 100)
    )
    is_spam = (
        (row.get("hashtag_spam_score", 0) > 0.5) or
        (row.get("mention_repetition_score", 0) == 1) or
        (row.get("text_exclam_ratio", 0) > 0.1) or
        (row.get("text_upper_ratio", 0) > 0.5)
    )
    if is_bot and not is_spam:
        return "bot"
    if is_spam:
        return "spam"

    # 4) Sinon, profil atypique générique
    return "atypique_generique"


def def_type_user(row):
    """
    Catégorise un UTILISATEUR en fonction de son score final et de ses metrics agrégées.

    """
    # 0) Normaux purs
    if row.get('final_label', 0) == 0:
        return 'normal'

    # 1) Médias en override
    if row.get('is_media', 0) == 1:
        return 'media'

    # 2) Calcul des scores partiels
    bot_score = (
        (row.get('mean_text_repetition_score', 0) > 0.5) +
        (row.get('log_ratio_followers_friends', 0) < -2) +
        (row.get('mean_late_night_tweet', 0) > 0.3) +
        (row.get('followers_per_tweet', 0) > 100)
    )
    spam_score = (
        (row.get('mean_hashtag_spam_score', 0) > 0.5) +
        (row.get('mean_mention_repetition_score', 0) > 0.5) +
        (row.get('mean_text_exclam_ratio', 0) > 0.1) +
        (row.get('mean_text_upper_ratio', 0) > 0.5)
    )
    influencer_score = (
        (row.get('user.followers_count', 0) > 10000) +
        (row.get('mean_engagement_rate', 0) < 0.001) +
        (row.get('account_age_days', 0) < 365)
    )
    # 3) Choix de la catégorie dominante
    scores = {'bot': bot_score, 'spam': spam_score, 'influenceur': influencer_score}
    max_label = max(scores, key=scores.get)
    max_value = scores[max_label]
    if max_value == 0:
        return 'atypique_generique'
    return max_label
