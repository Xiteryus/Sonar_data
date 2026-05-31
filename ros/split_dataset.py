"""
Decoupe sonar.csv en deux fichiers DISJOINTS :
  - sonar_train.csv : pour entrainer le modele (ai.py)
  - sonar_simu.csv  : pour la simulation (lignes JAMAIS vues par le modele)

Ainsi la simulation evalue le modele sur des donnees inconnues
-> des erreurs naturelles apparaissent, sans avoir besoin d'ajouter du bruit.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

# Chargement
sonar = pd.read_csv("sonar.csv", header=None)

X = sonar.drop(60, axis=1)
y = sonar[60]

# 70% entrainement / 30% simulation, en gardant le ratio Mines/Rochers (stratify)
df_train, df_simu = train_test_split(
    sonar,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Sauvegarde SANS en-tete ni index (meme format que sonar.csv d'origine)
df_train.to_csv("sonar_train.csv", header=False, index=False)
df_simu.to_csv("sonar_simu.csv", header=False, index=False)

print("--- Decoupage termine ---")
print(f"sonar_train.csv : {len(df_train)} lignes  {df_train[60].value_counts().to_dict()}")
print(f"sonar_simu.csv  : {len(df_simu)} lignes  {df_simu[60].value_counts().to_dict()}")
print("\nLes deux ensembles sont disjoints : aucune ligne de simulation")
print("n'a servi a l'entrainement.")
