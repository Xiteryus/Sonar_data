import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib

# 1. Chargement des donnees D'ENTRAINEMENT uniquement.
#    (sonar_simu.csv est mis de cote pour la simulation -> jamais vu par le modele)
sonar = pd.read_csv("sonar_train.csv", header=None)

# Affichage des informations de base
print("--- Dimensions (jeu d'entrainement) ---")
print(f"Lignes : {sonar.shape[0]}")
print(f"Colonnes de caractéristiques : {sonar.shape[1] - 1}")

print("\n--- Aperçu des données ---")
print(sonar.head())

# 2. Visualisation de la signature acoustique moyenne
plt.figure(figsize=(12, 6))

mean_mine = sonar[sonar[60] == 'M'].iloc[:, :60].mean()
mean_rock = sonar[sonar[60] == 'R'].iloc[:, :60].mean()

plt.plot(mean_mine.index.tolist(), mean_mine.values.tolist(), label='Mine', color='red')
plt.plot(mean_rock.index.tolist(), mean_rock.values.tolist(), label='Rocher', color='blue')

plt.title('Signature acoustique moyenne : Mine vs Rocher')
plt.xlabel('Bande de fréquence (0 à 59)')
plt.ylabel('Intensité moyenne du signal')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 3. Préparation des données
X = sonar.drop(60, axis=1)
y = sonar[60]

# Encodage M/R en 1/0
le = LabelEncoder()
y = le.fit_transform(y)

# Séparation interne (pour mesurer la performance pendant le developpement)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Normalisation des données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Entraînement et évaluation des modèles
results = {}

model_lr = LogisticRegression()
scores_lr = cross_val_score(model_lr, X_train, y_train, cv=5)
results['Reg. Logistique'] = np.mean(scores_lr)
model_lr.fit(X_train, y_train)

model_svm = SVC()
scores_svm = cross_val_score(model_svm, X_train, y_train, cv=5)
results['SVM'] = np.mean(scores_svm)
model_svm.fit(X_train, y_train)

model_knn = KNeighborsClassifier(n_neighbors=5)
scores_knn = cross_val_score(model_knn, X_train, y_train, cv=5)
results['KNN'] = np.mean(scores_knn)
model_knn.fit(X_train, y_train)

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
scores_rf = cross_val_score(model_rf, X_train, y_train, cv=5)
results['Random Forest'] = np.mean(scores_rf)
model_rf.fit(X_train, y_train)

# 5. Affichage du classement
df_results = pd.DataFrame(list(results.items()), columns=['Modèle', 'Score CV Moyen'])
df_results = df_results.sort_values(by='Score CV Moyen', ascending=False).reset_index(drop=True)

print("\n--- Classement des Modèles ---")
print(df_results)

# 6. Re-entrainement du SVM final sur TOUT le jeu d'entrainement disponible,
#    avec un scaler ajuste sur ces memes donnees (important pour l'inference).
final_scaler = StandardScaler()
X_full = final_scaler.fit_transform(X)
final_svm = SVC()
final_svm.fit(X_full, y)

# 7. Sauvegarde pour l'inférence ROS 2
print("\n--- Sauvegarde pour l'inférence ROS 2 ---")
joblib.dump(final_svm, 'sonar_svm_model.pkl')
joblib.dump(final_scaler, 'sonar_scaler.pkl')

print("Fichiers 'sonar_svm_model.pkl' et 'sonar_scaler.pkl' générés avec succès.")
print("Le modele n'a JAMAIS vu les lignes de sonar_simu.csv.")
