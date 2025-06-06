# Outil d'Analyse d'Infrastructure et Recommandations

Ce script Python utilise LangGraph et LangChain pour analyser des données de monitoring d'infrastructure, détecter des anomalies et générer des recommandations d'optimisation à l'aide d'un LLM (OpenAI GPT).


## Fonctionnalités

* **Ingestion de données :** Lit les métriques d'infrastructure depuis un fichier JSON ou à partir d'un dictionnaire.
* **Détection d'anomalies :** Identifie des anomalies simples basées sur des règles prédéfinies (ex: CPU > 80%, Mémoire > 80%, Statut de service != "online").
* **Génération de recommandations :** Utilise un LLM (via l'API OpenAI) pour suggérer des actions d'optimisation basées sur les anomalies détectées.
* **Workflow modulaire :** Le processus est structuré en étapes séquentielles (nœuds) grâce à LangGraph.
* **Exportation :** Un nœud existe pour sauvegarder l'état final (données, anomalies, recommandations) dans un fichier JSON.


## Les différents fichiers 

* **test_app.ipynb :** notebook de creation du script, pour avoir une approche pas à pas et corriger les erreurs plus facilement.
* **app.py :** script python qui reprend presque tous les éléments du notebook, mais avec les modifications apportées lors de l'entretien le 10/04 + l'exercice à finir (prédictions).
* **app_v2.py :** script python avec interface graphique simple à l'aide de la bibliothèque Streamlit, ne fonctionne que pour la première version du test, càd sans les prédictions.
* **Dossier Data :** contient le fichier rapport.json .
* **Dossier Recommendations :** contient le fichier .json en sortie du script app.py.

  
## Technologies Utilisées

* Python 3.12
* LangGraph
* LangChain 
* Pydantic (pour la structuration des données)
* OpenAI API (pour l'accès au LLM)
* Streaamlit (pour l'IHM)
