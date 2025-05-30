import numpy as np
import pandas as pd

def compute_spam_and_influencer_flags(
    df,
    high_followers=20000,
    low_statuses=500,
    new_account_days=180,
    min_engagement_rate=0.005
):
    """
    Compute two boolean flags on the DataFrame:
    - cond_influencer: Vrai si l'utilisateur est considéré comme un influenceur légitime
    - cond_star_spammer: vrai si l'utilisateur est un jeune star-spammer ( ex
     un compte qui vient d'être créer) et pas un influenceur
    """
    # Calculer le taux d'engagement
    followers = df['user.followers_count'].replace(0, np.nan)
    df['engagement_rate'] = (df['retweet_count'] + df['favorite_count']) / followers

    # Calculer le seuil du top 5% pour listed_count
    listed = df.get('user.listed_count', pd.Series(0, index=df.index)).fillna(0)
    min_listed_count = listed.quantile(0.95)

    # Conditions pour être un influenceur
    cond_verified = df.get('user.verified', pd.Series(False, index=df.index)).fillna(False)
    cond_listed = listed >= min_listed_count
    cond_influencer = (
        cond_verified
        | (
            (df['user.followers_count'] > high_followers)
            & (df['user.statuses_count'] >= low_statuses)
            & (df['engagement_rate'] >= min_engagement_rate)
            & cond_listed
        )
    )

    # Condition sur l'âge du compte, 
    if 'account_age_days' in df.columns:
        cond_new_account = df['account_age_days'] < new_account_days
    else:
        cond_new_account = pd.Series(False, index=df.index)

    # Conditions pour être un star_spammer
    cond_star_spammer = (
        (df['user.followers_count'] > high_followers)
        & (df['user.statuses_count']   < low_statuses)
        & cond_new_account
        & (cond_influencer)
    )
    return cond_influencer.astype(int), cond_star_spammer.astype(int)
