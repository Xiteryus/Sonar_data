# Détection Sonar par Machine Learning

Ce projet s'inscrit dans le domaine de la perception autonome pour la robotique sous-marine. Dans les environnements aquatiques où la visibilité optique est souvent nulle, les robots doivent s'appuyer sur la télémétrie acoustique (Sonar) pour naviguer et interagir avec leur environnement.

L'objectif de ce dépôt est d'implémenter et de comparer plusieurs modèles d'apprentissage automatique (Régression Logistique, SVM, Forêt Aléatoire) capables d'analyser des signatures acoustiques complexes. Le système exploite les données de 60 bandes de fréquences pour distinguer de manière autonome et fiable des objets métalliques (Mines) de simples obstacles naturels (Rochers), une fonctionnalité critique pour la sécurisation des trajectoires robotisées.

## Installation et Exécution

- Clonez ce dépôt sur votre machine locale :

```bash
git clone https://github.com/Xiteryus/Sonar_data.git
```
- Déplacez-vous dans le répertoire contenant le programme de Machine Learning :

```bash
cd Sonar_Data
```

- Installez l'ensemble des dépendances requises via le fichier de configuration :


```bash
pip install -r requirements.txt
```

## Exécution et Résultats


- Lancez le notebook Jupyter principal pour entraîner les algorithmes, visualiser les matrices de confusion et afficher le classement final des modèles :

```bash
jupyter notebook ML_dev/main.ipynb
```

- L'exécution de cette commande va démarrer un serveur local. Si votre navigateur web ne s'ouvre pas automatiquement, un lien http://localhost:8888/ apparaîtra dans votre terminal. Cliquez simplement sur ce lien pour accéder à l'interface et voir les résultats.

- Le code est également disponible sur [Google Colab](https://colab.research.google.com/drive/1eQLsB-q40bpODR6FsR-rhZisyvaHqOdb?usp=sharing)



## Robotics Integration (ROS)

Ce projet inclut également une intégration robotique. Pour lancer et configurer cette partie, veuillez vous rendre dans le dossier ros.

```bash
cd ros
```

Vous y trouverez un README.md spécifique qui détaille l'architecture logicielle et les commandes nécessaires au lancement des nœuds ROS.