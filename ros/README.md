# Détecteur Sonar Mine / Rocher (ROS 2 + RViz)

Simulation d'un sonar qui balaye un environnement et utilise une IA (SVM)
pour classer chaque obstacle détecté : **Mine** ou **Rocher**.

---

## 1. Fichiers du projet

| Fichier | Rôle |
|---|---|
| `sonar.csv` | Dataset complet d'origine (208 lignes) |
| `split_dataset.py` | Découpe `sonar.csv` en train + simu |
| `sonar_train.csv` | Données pour entraîner le modèle |
| `sonar_simu.csv` | Données pour la simulation (jamais vues par le modèle) |
| `machine_learning.py` | Entraîne le SVM et génère les fichiers `.pkl` |
| `sonar_svm_model.pkl` | Le modèle entraîné |
| `sonar_scaler.pkl` | Le normaliseur des données |
| `simu_env.py` | Simulateur : génère les obstacles et le faisceau sonar |
| `ros_2d.py` | L'IA : classe les détections et les affiche |

---

## 2. Préparation (à faire une seule fois)

Avant chaque commande, charger l'environnement ROS :

```bash
source /opt/ros/humble/setup.bash
```

**Étape A — Découper le dataset** (si pas déjà fait) :

```bash
python3 split_dataset.py
```

**Étape B — Entraîner le modèle** :

```bash
python3 machine_learning.py
```

> Une fenêtre de graphique s'ouvre. **Fermez-la** pour que le script finisse
> et crée `sonar_svm_model.pkl` et `sonar_scaler.pkl`.

---

## 3. Lancer la simulation

Ouvrir **3 terminaux**. Dans chacun, faire d'abord :

```bash
source /opt/ros/humble/setup.bash
```

**Terminal 1 — L'IA (à lancer en premier)** :

```bash
python3 ros_2d.py
```

**Terminal 2 — Le simulateur** :

```bash
python3 simu_env.py
```

**Terminal 3 — La visualisation** :

```bash
rviz2
```

---

## 4. Configuration de RViz

Dans RViz, une fois ouvert :

1. En bas à gauche, cliquer sur **Add**.
2. Onglet **By topic**, ajouter les affichages suivants :
   - `/environment` → **MarkerArray** (les vrais obstacles)
   - `/sonar_beam_viz` → **Marker** (le faisceau du sonar)
   - `/sonar/detections_2d_viz` → **Marker** (les détections de l'IA)
3. En haut à gauche, dans **Global Options**, mettre **Fixed Frame** sur `map`.

---

## 5. Comment lire l'affichage

**Les cubes = la vérité (vrais obstacles) :**

| Couleur | Signification |
|---|---|
| 🔴 Rouge | C'est vraiment une Mine |
| 🔵 Bleu | C'est vraiment un Rocher |

**Les cylindres = ce que l'IA a prédit :**

| Couleur | Signification |
|---|---|
| 🔴 Rouge | L'IA pense que c'est une Mine |
| 🔵 Bleu | L'IA pense que c'est un Rocher |
| 🟡 Jaune | **Erreur** : l'IA s'est trompée |

Le faisceau cyan tourne et « scanne » les obstacles un par un.

---

## 6. Fonctionnement

- Toutes les **10 secondes**, 10 nouveaux obstacles sont générés au hasard.
  Les anciens cubes **et** les anciens cercles de détection sont effacés
  automatiquement.
- Les données de simulation n'ont jamais servi à l'entraînement : les erreurs
  (jaunes) sont donc naturelles et réalistes (~80 % de réussite).
- Chaque détection est enregistrée dans `simulation_results.csv`.

---

## 7. Réglages utiles

Dans `simu_env.py` :

- **Durée avant régénération** : ligne `self.create_timer(10.0, self.regenerate_environment)`
  → changer `10.0` (en secondes).
- **Ajouter du bruit** (tâche plus difficile) : ligne `self.noise_std = 0.0`
  → mettre par exemple `0.02`.

---

## 8. Arrêter / Réinitialiser

- **Arrêter** : `Ctrl + C` dans chaque terminal.
  (Le message `rcl_shutdown already called` à la fermeture est sans gravité.)
- **Repartir de zéro** pour le fichier de résultats :

```bash
rm simulation_results.csv
```
![Démo de la simulation](Sonar.gif)
