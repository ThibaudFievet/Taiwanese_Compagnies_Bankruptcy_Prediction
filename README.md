*(English version below)*

# Prédiction de la faillite d'entreprises taïwanaises

**Projet de data mining et de machine learning - EDHEC Business School - Double diplôme en ingénierie digitale et management (majeure finance)**

![Python](https://img.shields.io/badge/Language-Python-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange?style=flat&logo=scikit-learn)
![NumPy](https://img.shields.io/badge/Library-NumPy-blue?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Library-Matplotlib-ffffff?style=flat&logo=python)
![Seaborn](https://img.shields.io/badge/Library-Seaborn-blue?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Library-Pandas-150458?style=flat&logo=pandas&logoColor=white)

## Présentation du projet

Ce projet applique des techniques de **machine learning** afin de prédire et d'analyser la faillite d'entreprises à partir d'un **jeu de données sur les faillites d'entreprises taïwanaises**. Le principal défi de cette analyse était de traiter un **jeu de données fortement déséquilibré**, dans lequel seulement 3,2 % des entreprises étaient déclarées en faillite. Chaque partie de code est précédée d'une cellule Markdown expliquant le raisonnement suivi lors de son écriture. Ce travail a été réalisé individuellement dans le cadre du cours de Data Mining & Machine Learning à l'EDHEC Business School.

La démarche combine de l'**apprentissage supervisé** pour prédire des scores de risque et de l'**apprentissage non supervisé** pour identifier des profils financiers distincts au sein du marché.

---

## Avertissement

> **Remarque :** Comme il s'agissait de mon tout premier projet en machine learning, l'objectif principal était de comprendre les concepts et méthodologies fondamentaux. Je suis conscient que le code, sa modularité et la présentation visuelle peuvent encore être largement améliorés.

---

## Consignes du projet

Le projet devait analyser un jeu de données choisi librement, en respectant les contraintes suivantes :

* **Sélection du jeu de données :**
    * Il devait être disponible publiquement (par exemple sur UCI ou Kaggle).
    * **Taille minimale :** au moins 5 000 observations.
    * **Dimensionnalité :** au moins 15 variables.

* **Travaux d'analyse :**
    * Réaliser une **analyse descriptive** complète (univariée et bivariée).
    * Appliquer au moins **deux techniques de machine learning** parmi la classification, la régression et le clustering.

* **Exigences méthodologiques :**
    * Justifier le choix du jeu de données et des méthodes.
    * Appliquer une **mise à l'échelle des variables** (standardisation ou normalisation) lorsque cela était pertinent.
    * Mettre en place une **validation croisée** afin de limiter le surapprentissage.

---

## Le jeu de données

* **Source :** [UCI Machine Learning Repository - Taiwanese Bankruptcy Prediction](https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction)
* **Observations :** 6 819 entreprises
* **Variables :** 95 ratios financiers (par exemple : ROA, ratio d'endettement, résultat net sur actifs)
* **Variable cible :** binaire (`1` = Faillite, `0` = Entreprise solvable)
* **Déséquilibre :** 96,8 % d'entreprises solvables contre 3,2 % en faillite

---

## Méthodologie

### 1. Nettoyage des données et analyse exploratoire
* Suppression des colonnes constantes (variance nulle ou quasi nulle) afin de réduire le bruit.
* Réalisation d'analyses univariées et bivariées pour identifier les principaux facteurs de risque.
* Application de **StandardScaler** pour normaliser les 94 variables financières.

### 2. Classification (apprentissage supervisé)
* **Modèle :** régression logistique.
* **Validation :** validation croisée stratifiée en 5 parties sur un échantillon d'entraînement représentant 80 % des données.
* **Métrique :** optimisation de la **ROC-AUC** en raison du déséquilibre des classes, avec analyse complémentaire du score F1, du rappel et de la précision.
* **Sélection des variables :** mise en œuvre d'une régression Lasso (L1), qui a éliminé 35 variables redondantes et identifié 58 indicateurs financiers pertinents.

### 3. Clustering (apprentissage non supervisé)
* **Algorithme :** clustering K-Means sur l'ensemble des données.
* **Optimisation :** utilisation de la méthode du coude pour déterminer le nombre optimal de clusters ($K=6$).
* **Profilage :** analyse des profils financiers moyens de chaque cluster afin d'identifier différents archétypes de marché (par exemple, « endettement élevé / risque élevé » contre « fonds propres élevés / situation saine »).

---

## Principaux résultats

### Performance de la classification

Le modèle de régression logistique a démontré une bonne stabilité (**écart-type en validation croisée : 0,0263**) ainsi qu'une bonne capacité de classement :

* **Métriques principales :** **ROC-AUC : 0,91** et **score F1 macro : 0,61**, une métrique plus pertinente que le taux de précision biaisé de 96 %.
* **Conclusion :** avec le seuil par défaut, le modèle est opérationnellement conservateur (**rappel : 0,18** / **précision : 0,38**). Le contraste entre une AUC élevée et un rappel faible indique que, même si le modèle classe correctement les entreprises risquées, le seuil de décision doit être abaissé pour détecter la majorité des cas de faillite.

### Résultats du clustering

Le modèle K-Means non supervisé ($K=6$) a réussi à isoler le risque sans jamais utiliser la variable cible :

* **Cluster 3 (« risque élevé ») :** groupe présentant un **taux de faillite de 75 %**, caractérisé par un endettement élevé et une faible efficacité.
* **Clusters 0 et 2 (« situation saine ») :** groupes présentant un **taux de faillite de 0 %**.
* **Impact :** ces résultats montrent que la difficulté financière produit une signature distincte et mathématiquement observable, même en l'absence de données étiquetées.

---

## Difficultés et apprentissages

Bien que les concepts fondamentaux du projet aient été accessibles, plusieurs aspects ont nécessité un approfondissement et une évolution de ma manière de travailler :

* **Passage à Scikit-Learn :** la transition entre Python académique et les bibliothèques de machine learning a constitué une étape importante. J'ai consacré du temps à déconstruire de nouvelles structures de code et syntaxes, car l'implémentation du machine learning diffère de la programmation procédurale enseignée dans les cours d'introduction.
* **Gestion du déséquilibre des données :** le déséquilibre du jeu de données a représenté un enjeu technique majeur. J'ai étudié et mis en œuvre des stratégies adaptées afin de préserver la robustesse prédictive du modèle et d'éviter un biais en faveur de la classe majoritaire.
* **Priorité donnée à la documentation :** l'utilisation intensive de Markdown a constitué une évolution par rapport à mes habitudes. J'ai traité les annotations du notebook comme un véritable rapport de projet. Malgré le temps nécessaire, j'ai compris l'importance de cette bonne pratique pour la reproductibilité et la clarté, et je souhaite la conserver dans mes futurs travaux techniques.

---

## Améliorations futures

* **Scoring probabiliste :** utiliser les probabilités produites par le modèle pour évaluer le prix de **Credit Default Swaps (CDS)**.
* **Modèles avancés :** tester des méthodes d'ensemble comme **Random Forest** ou **XGBoost** afin d'améliorer potentiellement le rappel.

---

# Taiwanese Bankruptcy Prediction Analysis

**Data Mining & Machine Learning Project - EDHEC Business School - Dual Degree in Digital ENgineering & Management (finance major)**

![Python](https://img.shields.io/badge/Language-Python-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange?style=flat&logo=scikit-learn)
![NumPy](https://img.shields.io/badge/Library-NumPy-blue?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Library-Matplotlib-ffffff?style=flat&logo=python)
![Seaborn](https://img.shields.io/badge/Library-Seaborn-blue?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Library-Pandas-150458?style=flat&logo=pandas&logoColor=white)

## Project Overview

This project applies macine learning techniques to predict and analyze corporate bankruptcy using a **Taiwanese Bankruptcy Prediction dataset**. The primary challenge of this analysis was dealing with a **highly imbalanced dataset**, where only 3.2% of companies were labeled as bankrupt. Each code is predede by a markdown cell that explain my thought when writing it. It has been realised as an individual assignment for my grade of Data Mining & Machine Learning course at EDHEC Business School.

The workflow combines **Supervised Learning** to predict risk scores and **Unsupervised Learning** to discover hidden financial profiles within the market.

---

## Disclaimer

> **Note:** As this was my very first project in Machine Learning, the primary focus was on understanding the core concepts and methodologies. I am aware that there is still significant room for improvement regarding code optimization, modularity, and visual presentation.

---

## Project Requirements

The assignment required analyzing a dataset of our choice under the following constraints:

* **Dataset Selection:**
    * Must be publicly available (e.g., UCI, Kaggle).
    * **Minimum Size:** At least 5,000 instances (rows).
    * **Dimensionality:** At least 15 features (variables).

* **Analysis Tasks:**
    * Perform a complete **Descriptive Analysis** (Univariate & Bivariate).
    * Apply at least **two different Machine Learning techniques** (chosen from Classification, Regression, or Clustering).

* **Methodological Requirements:**
    * Justify the choice of dataset and methods.
    * Apply **Feature Scaling** (Standardization/Normalization) where appropriate.
    * Implement **Cross-Validation** to prevent overfitting.
  
---

## The Dataset

* **Source:** [UCI Machine Learning Repository - Taiwanese Bankruptcy Prediction](https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction)
* **Instances:** 6,819 companies
* **Features:** 95 financial ratios (e.g., ROA, Debt Ratio, Net Income to Assets)
* **Target:** Binary (`1` = Bankrupt, `0` = Solvent)
* **Imbalance:** 96.8% Solvent vs. 3.2% Bankrupt

---

## Methodology

### 1. Data Cleaning & EDA
* Removed constant columns (zero and almost zero variance) to reduce noise.
* Performed univariate and bivariate analysis to identify key risk drivers.
* Applied **StandardScaler** to normalize the 94 financial features.

### 2. Classification (Supervised Learning)
* **Model:** Logistic Regression.
* **Validation:** 5-Fold Stratified Cross-Validation on an 80/20 split.
* **Metric:** Optimized for **ROC-AUC** due to class imbalance, also F1 Score, Recall and Precision.
* **Feature Selection:** Implemented **Lasso Regression (L1)** which successfully eliminated 35 redundant features, identifying 58 not-redondant financial indicators.

### 3. Clustering (Unsupervised Learning)
* **Algorithm:** K-Means Clustering on the full dataset.
* **Optimization:** Used the **Elbow Method** to determine the optimal number of clusters ($K=6$).
* **Profiling:** Analyzed the mean financial profiles of each cluster to identify distinct market archetypes (e.g., "High-Debt/High-Risk" vs. "High-Equity/Healthy").

---

## Key Results & Findings

### Classification Performance
The Logistic Regression model demonstrated good stability (**CV Std Dev: 0.0263**) and ranking capability:
* **Key Metrics:** **ROC-AUC: 0.91** and **Macro F1-Score: 0.61** (a more honest metric than the biased 96% Accuracy).
* **Conclusion:** The model is operationally conservative at the default threshold (**Recall: 0.18** / **Precision: 0.38**). The strong contrast between the high AUC and low Recall indicates that while the model correctly ranks risky companies, the decision threshold must be tuned downward to capture the majority of bankruptcy cases.

### Clustering Insights
The unsupervised K-Means model ($K=6$) successfully isolated risk without ever seeing the target label:
* **Cluster 3 ("High Risk"):** Identified a group with a **75% bankruptcy rate**. Characterized by high debt and low efficiency.
* **Cluster 0 & 2 ("Healthy"):** Identified groups with a **0% bankruptcy rate**.
* **Impact:** Proves that financial distress creates a distinct, mathematically observable footprint even without labeled data.

---

## Challenges & Learning Journey

While the core concepts of the project were straightforward, several areas required a deeper dive and a shift in mindset:

* **Bridging the gap to Scikit-Learn:** Transitioning from academic Python to Machine Learning libraries was a key step. I spent significant time deconstructing new code structures and syntax within `Scikit-Learn`, as ML implementation often differs from standard procedural programming taught in introductory courses.
* **Addressing Dataset Imbalance:** A major technical focus was handling the imbalanced nature of the dataset. I dedicated time to researching and implementing appropriate strategies to ensure the model's predictive power remained robust and wasn't biased toward the majority class.
* **Documentation as a Priority:** A departure from my usual workflow was the extensive use of Markdown. I treated notebook annotations as a formal project report. While time-consuming, I recognized this as a "best practice" for reproducibility and clarity—a habit I intend to maintain for all future technical work.

---

## Future Improvements
* **Probability Scoring:** Use the model's probability outputs to price **Credit Default Swaps (CDS)**.
* **Advanced Models:** Test ensemble methods like **Random Forest** or **XGBoost** to potentially improve Recall.
