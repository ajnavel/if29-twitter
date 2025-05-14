import re
import pandas as pd

# Liste de mots-clés pour détecter un compte média ou institution
MEDIA_KEYWORDS = [
    ## Anglais
    "journal", "magazine", "press", "news", "radio",
    "tv", "television", "éditorial", "reporter", "reportage",

    ## Ecole etc en anglais 
    "university", "college", "school", "institute", "academy",
    
    ## Français
    "info", "télévision", "média", "reporter", "éditorial", "actualité",
    "article", "interview",

    ## Ecole en Français
    "université", "école", "institut", "académie", "études",
    
    ## Arabe
    "خبر", "إعلام", "صحافة", "تلفزيون", "راديو", 
    "قناة", "فضائية", "نشرة", "khabar", "ealam", "sahafa",
    "talfaz", "rida", "qanat",
    "جامعة", "مدرسة", "معهد", "أكاديمية", "دراسات",

    ## Espagnol
    "noticia", "prensa", "medio", "radio", "televisión",
    "periodista", "información", "canal", "comunicación",
    "reportaje", "emisora",

    ## Allemand
    "nachrichten", "medien", "presse", "fernsehen",
    "radio", "sender", "reporter", "zeitung", "bericht",
    "übertragung"
]


def compute_media_flag(
    df: pd.DataFrame,
    description_col: str = "description",
    keywords: list[str] | None = None,
    ## Ici on doit match au minimum 2 mots-clés 
    min_matches: int = 2
) -> pd.Series:
    """
    Retourne un indicateur (1/0) s'il s'agit vraisemblablement d'un compte média ou institution.
    Renvoie 1 si la description contient au moins `min_matches` mots-clés distincts parmi `keywords`.
    """
    # Sélection de la liste de mots-clés
    if keywords is None:
        keywords = MEDIA_KEYWORDS

    # Préparation des textes (remplace NaN par chaîne vide)
    texts = df[description_col].fillna("").astype(str)

    # Pour chaque mot-clé, on teste la présence du mot complet (avec \b)
    flags = pd.DataFrame(
        {
            kw: texts.str.contains(rf"\b{re.escape(kw)}\b", case=False, regex=True)
            for kw in keywords
        },
        index=df.index
    )

    # Somme des correspondances
    count_matches = flags.sum(axis=1)

    # Retourne 1 si au moins min_matches mots-clés sont présents
    return (count_matches >= min_matches).astype(int)
