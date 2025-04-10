# Outil d'Analyse d'Infrastructure et Recommandations

Ce script Python utilise LangGraph et LangChain pour analyser des données de monitoring d'infrastructure, détecter des anomalies et générer des recommandations d'optimisation à l'aide d'un LLM (OpenAI GPT).

## Fonctionnalités

* **Ingestion de données :** Lit les métriques d'infrastructure depuis un fichier JSON ou à partir d'un dictionnaire.
* **Détection d'anomalies :** Identifie des anomalies simples basées sur des règles prédéfinies (ex: CPU > 80%, Mémoire > 80%, Statut de service != "online").
* **Génération de recommandations :** Utilise un LLM (via l'API OpenAI) pour suggérer des actions d'optimisation basées sur les anomalies détectées.
* **Workflow modulaire :** Le processus est structuré en étapes séquentielles (nœuds) grâce à LangGraph.
* **Exportation :** Un nœud existe pour sauvegarder l'état final (données, anomalies, recommandations) dans un fichier JSON.

## Technologies Utilisées

* Python 3.12
* LangGraph
* LangChain 
* Pydantic (pour la structuration des données)
* OpenAI API (pour l'accès au LLM)
* Streaamlit (pour l'IHM)

## Les différents fichiers 

* test_app.ipynb : notebook de creation du script, pour avoir une approche pas à pas et corriger les erreurs plus facilement.
* app.py : script python qui reprend presque tous les éléments du notebook.
* app_v2.py : script python avec interface graphique simple à l'aide de la bibliothèque Streamlit.
