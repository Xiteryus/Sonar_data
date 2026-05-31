# Détection Sonar par Machine Learning

Ce projet implémente des modèles d'apprentissage automatique pour analyser des signaux sonar, permettant à un système robotisé de distinguer de manière autonome des mines sous-marines de simples rochers.

## Installation et Exécution

- Clonez ce dépôt sur votre machine locale :

```bash
git clone https://github.com/Xiteryus/Sonar_data.git
```
- Déplacez-vous dans le répertoire contenant le programme de Machine Learning :

```bash
cd Sonar_Data
```

- Lancez le script principal pour entraîner les modèles et afficher les résultats :

```bash
jupyter notebook ML_dev/main.ipynb
```

- Le code est également disponible sur [Google Colab](https://colab.research.google.com/drive/1eQLsB-q40bpODR6FsR-rhZisyvaHqOdb?usp=sharing)

## Prérequis et Dépendances

Assurez-vous d'avoir Python 3.12.3 installé sur votre système. Le programme s'appuie sur les librairies d'analyse de données et de machine learning standards.

Vous pouvez installer toutes les dépendances requises via la commande suivante :


```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Robotics Integration (ROS)

Ce projet inclut également une intégration robotique. Pour lancer et configurer cette partie, veuillez vous rendre dans le dossier ros.

```bash
cd ros
```

Vous y trouverez un README.md spécifique qui détaille l'architecture logicielle et les commandes nécessaires au lancement des nœuds ROS.