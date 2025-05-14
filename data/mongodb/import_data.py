import os
import json
from pymongo import MongoClient

# Configuration
uri = "mongodb://localhost:27017/"
db_name = "twitterdb"
collection_name = "tweets"
data_folder = r"C:XXX\Woldcup2008\Tweet Worldcup 200Twets"

# Connexion MongoDB
client = MongoClient(uri)
db = client[db_name]
collection = db[collection_name]

# Parcours tous les fichiers JSON
for filename in os.listdir(data_folder):
    if filename.endswith('.json'):
        filepath = os.path.join(data_folder, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            try:
                lines = file.readlines()
                documents = [json.loads(line) for line in lines if line.strip()]
                collection.insert_many(documents)
                print(f"{filename} importé avec succès.")
            except Exception as e:
                print(f"Erreur sur {filename} : {e}")

print("Importation terminée.")
