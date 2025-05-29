# Estimation multi-modèle de la volatilité

## Contexte

Ce projet a pour objectif de développer un modèle d’ensemble combinant trois approches complémentaires pour prévoir la volatilité réalisée à court terme sur des séries de prix financiers :

1. **HAR–RV** (Heterogeneous Autoregressive Realized Volatility)
2. **GARCH** (Generalized Autoregressive Conditional Heteroskedasticity)
3. **LSTM** (Long Short-Term Memory)

En tirant parti de la mémoire longue de la volatilité, de l’hétéroscédasticité conditionnelle et des capacités non linéaires des réseaux de neurones, ce modèle vise à offrir des prévisions plus robustes et précises.

---

## Fonctionnalités

* Prétraitement et calcul de la volatilité réalisée (RV) à partir de données m1.
* Implémentation de modèles HAR–RV, GARCH et LSTM.
* Combinaison des prévisions par moyenne pondérée optimisée via validation croisée.
* Évaluation des performances avec RMSE.
* Visualisation des résultats et comparaison entre modèles.

---

## Installation

1. Cloner le dépôt :

   ```bash
   git clone https://github.com/ArthurJouannin/prevision-volatilite.git
   ```

2. Créer et activer un environnement virtuel :

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Installer les dépendances :

   ```bash
   pip install pandas numpy scikit-learn arch tensorflow matplotlib 
   ```

---

## Résultats 

---
