# -prediction-prix-immobilier
Projet de Machine Learning pour la prédiction des prix de l'immobilier

## 📌 Présentation du projet

Ce projet a été réalisé dans le cadre de notre **Projet de Fin d'Études (PFE) de Licence en Sciences Mathématiques et Informatique** à la Faculté des Sciences Ben M'Sik (FSBM).

L'objectif est de développer un système de **prédiction des prix immobiliers à Melbourne, en Australie**, à partir des caractéristiques des biens immobiliers, en utilisant des techniques de Machine Learning.

Le projet comprend la préparation des données, l'entraînement et la comparaison de plusieurs modèles de régression, ainsi que le développement d'une interface permettant d'effectuer des prédictions.

## 🎯 Objectifs

- Explorer et préparer les données immobilières.
- Appliquer des techniques de prétraitement et de feature engineering.
- Entraîner et comparer plusieurs modèles de régression.
- Évaluer les performances des modèles à l'aide de métriques adaptées.
- Identifier le modèle présentant les meilleurs résultats expérimentaux.
- Développer une interface de prédiction des prix immobiliers.

## 📊 Données utilisées

Le projet utilise un jeu de données sur les prix immobiliers de Melbourne.

- **Variable cible :** `Price`
- **Type de problème :** régression supervisée.
- **Préparation des données :** traitement des valeurs manquantes, encodage des variables catégorielles et préparation des variables numériques.

Le jeu de données original n'est pas inclus dans ce dépôt pour le moment.

## ⚙️ Méthodologie

### 1. Prétraitement des données

- Suppression des doublons.
- Traitement des valeurs manquantes.
- Traitement des valeurs extrêmes.
- Encodage des variables catégorielles.
- Mise à l'échelle des variables numériques, selon les besoins des modèles.

### 2. Feature engineering

Création de variables dérivées à partir des caractéristiques immobilières afin d'enrichir les informations disponibles pour les modèles.

### 3. Entraînement et comparaison

Plusieurs algorithmes de régression ont été étudiés :

- Gradient Boosting
- XGBoost
- CatBoost
- LightGBM
- Random Forest
- Support Vector Regression (SVR)
- Decision Tree
- AdaBoost
- K-Nearest Neighbors (KNN)
- Artificial Neural Network (ANN)

### 4. Évaluation

Les modèles sont comparés à l'aide des métriques suivantes :

- **R² (coefficient de détermination)** : mesure la capacité du modèle à expliquer les variations des valeurs cibles.
- **MAE (Mean Absolute Error)** : mesure l'erreur absolue moyenne entre les valeurs réelles et les prédictions.

## 📈 Résultats expérimentaux

Le tableau suivant reprend les résultats obtenus dans le cadre du projet.

| Modèle | R² Train | R² Test | MAE Train | MAE Test |
|---|---:|---:|---:|---:|
| Gradient Boosting | 99,99 % | 99,88 % | 0,00525 | 0,00890 |
| XGBoost | 99,98 % | 99,56 % | 0,0072 | 0,0142 |
| CatBoost | 99,91 % | 99,51 % | 0,0180 | 0,0207 |
| LightGBM | 99,92 % | 99,49 % | 0,0126 | 0,0167 |
| Random Forest | 99,50 % | 99,30 % | 0,0457 | 0,0488 |
| SVR | 99,70 % | 98,96 % | 0,0456 | 0,0562 |
| Decision Tree | 98,67 % | 98,32 % | 0,0817 | 0,0856 |
| AdaBoost | 96,03 % | 95,87 % | 0,1470 | 0,1461 |
| KNN | 91,56 % | 87,29 % | 0,1981 | 0,2435 |
| ANN | 99,92 % | 99,85 % | 0,01835 | 0,0231 |

**Résultat principal :** selon le tableau expérimental, Gradient Boosting présente le plus faible MAE sur le jeu de test, tandis que l'ANN obtient un R² de test légèrement supérieur à celui de Gradient Boosting.

*Remarque : les résultats sont ceux rapportés dans le projet et restent à vérifier à partir du notebook, notamment concernant les variables utilisées pour l'entraînement et l'échelle de calcul du MAE.*

## 🖥️ Application de prédiction

Le projet comprend une partie dédiée au déploiement d'une interface de prédiction. L'utilisateur peut renseigner les caractéristiques d'un bien immobilier afin d'obtenir une estimation de son prix.

Les instructions d'installation et d'exécution seront précisées en fonction des fichiers réellement présents dans le dossier de déploiement.

## 🛠️ Technologies utilisées

- Python
- Pandas et NumPy
- Scikit-learn
- XGBoost
- CatBoost
- LightGBM
- Matplotlib et Seaborn
- Streamlit

## 📁 Contenu du projet

Les principaux livrables comprennent :

- Un notebook Jupyter pour l'analyse des données et les expérimentations.
- Un jeu de données au format CSV.
- Un rapport de PFE.
- Une présentation PowerPoint.
- Un dossier consacré au déploiement du système.

L'organisation exacte des fichiers sera documentée selon la structure finale du dépôt.

## 👥 Équipe du projet

- Hajar Benlaidi

## 🎓 Contexte académique

**Formation :** Licence Sciences Mathématiques et Informatique  
**Établissement :** Faculté des Sciences Ben M'Sik (FSBM)  
**Année de présentation :** 2025
## 📁 Accès aux fichiers du projet

Les documents complémentaires du projet sont disponibles sur Google Drive :

- [📂 Accéder au dossier complet du projet](https://drive.google.com/drive/folders/1sqxIeqkkkZwpR225HllltJcMi3WGk-Fr?usp=drive_link)

Le dossier contient le rapport de PFE, la présentation, le notebook Jupyter, le jeu de données et les fichiers liés au déploiement du système.
