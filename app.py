## Imports
import os
import json

# LangGraph pour l'architecture en étapes/noeuds
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# LangChain pour la génération de recommandations
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


## Récuépration des variables d'environnement (clé API)
openai_api_key = os.environ['OPENAI_API_KEY']

## Création d'un client OpenAI pour la génération de recommandations
client = OpenAI()
llm = ChatOpenAI(model="gpt-4o-mini")


class Anomalies(BaseModel):
    """
    Sortie pour les anomalies
    """
    metric: str = Field(description="Métrique ou élément où il y a une anomalie")
    value: Any = Field(description="La valeur qui pose problème")
    issue: str = Field(description="Les conséquences ou explications de la valeur problématique")
   
   
class Recommendation(BaseModel):
    """
    Sortie pour les recommendations
    """
    anomalie: str = Field(description="Description d'anomalie.s rencontrée.S")
    suggestion: str = Field(description="Action.s suggérée.s pour optimiser l'infrastructure")

class RecommendationList(BaseModel):
    """
    Liste de Recommandations s'il y en a plusieurs pour 1 rapport
    """
    recommendations: List[Recommendation] = Field(description="Une liste de recommandations d'optimisation basées sur les anomalies détectées.")

## Tendances émergentes

## si un service (database, api_gateay, cache) est 'degraded' ==> latency élevée,
## donc comprendre pourquoi degraded, trouver un pattern

## timestamp : regarder si une certaine heure pose pb
class PatternAnomalies(BaseModel):
    """
    Si on trouve un pattern/une tendance qui est en lien avec une anomalie
    alors, on peut prévoir ce qui peut se passer avec les recommandations faites en fonction des anomalies
    """
    pattern: str  = Field(description="Tendances/Pattern qui peuvent être significatif d'une anomalie")
    recommendation: RecommendationList = Field(description="Une liste de recommandations d'optimisation basées sur les anomalies détectées.")

class PatternAnomaliesList(BaseModel):
    predictions: List[PatternAnomalies] = Field(description="Une liste de prédiction de tendances qui pourraient entraîner des anomaliess")
    
class State(TypedDict):
    """
    Définition de l'état du graphe : InputState pour la gestion de l'entrée et OutputState pour la gestion de la sortie
    """
    input_path: str                 
    input_data: Optional[List[Dict[str, Any]]] 
    small_list_anomalies: List[Anomalies]     
    anomalies: List[Anomalies]     
    recommendations: List[RecommendationList]
    predictions: List[PatternAnomaliesList]
    error: Optional[str]   


## Fonctions pour chaque étape de l'architecture
def data_ingestion(state):
    """ Noeud d'ingestion des données """
    print("### Noeud en cours : Ingestion des données ###")
    #print("\tSTATE : ", state)
    try :
        if ".json" in state["input_data"] : 
            with open(state["input_data"], 'r') as file :
                data = json.load(file)
            print(f"Data ingestion complétée pour le fichier : { state['input_data'] }\n")

        else :
            data = state["input_data"]
            print(f"Data ingestion complétée")
        return {**state, "input_data": data, "error": None}
        
    except Exception as e :
        print(f"Erreur pendant l'ingestion du fichier : {e}\n")
        return {**state, "input_data": {}, "error": f"Ingestion failed: {str(e)}"}
    

def anomalies_detection(state):
    """ Noeud de détection d'anomalies dans les donées """
    print("\n\n### Noeud en cours : Détection des anomalies ###")
    #print("\tSTATE : ", state)

    if state["error"] is not None:
        print("Erreur détectée dans un noeud précédent. Arrêt de la génération de recommandations\n")
        return state
    
    if isinstance(state["input_data"], dict) : 
        state["input_data"] = [state["input_data"]]
    data = state["input_data"]
    small_list_anomalies: List[Anomalies] = []
    anomalies_detected: List[Anomalies] = []
    number_anomalies = 0

    for d in data :
        if d["cpu_usage"] > 80 :
            small_list_anomalies.append({"metric": "cpu_usage","value": d["cpu_usage"],"issue": "High CPU Usage (> 80%)"})
            anomalies_detected.append({"metric": "cpu_usage","value": d["cpu_usage"],"issue": "High CPU Usage (> 80%)"})
            number_anomalies += 1
        if d["memory_usage"] > 80 :
            small_list_anomalies.append({"metric": "memory_usage","value": d["memory_usage"],"issue": "High Memory Usage (> 80%)"})
            anomalies_detected.append({"metric": "memory_usage","value": d["memory_usage"],"issue": "High Memory Usage (> 80%)"})
            number_anomalies += 1
        else :
            anomalies_detected.append({})

        service_status = d["service_status"]
        for service, status in service_status.items():
            if status != "online":
                small_list_anomalies.append({"metric": f"service_status : {service}", "value": status, "issue": f"Service {service} is {status}"})
                anomalies_detected.append({"metric": f"service_status : {service}", "value": status, "issue": f"Service {service} is {status}"})
                number_anomalies += 1
            else :
                anomalies_detected.append({})
        
    print(f"{number_anomalies} anomalie.s détectée.s")
    # print("\n\tANOMALIES DETECTED ", anomalies_detected)
    state["anomalies"] = anomalies_detected
    state["small_list_anomalies"] = small_list_anomalies
    return state



def recommandations_generation(state):
    """ Noeud de génération de recommandations à partir des anomalies détectées """
    print("\n\n### Noeud en cours : Génération de recommandations ###")
    # print("STATE : ", state)

    if state["error"] is not None:
        print("Erreur détectée dans un noeud précédent. Arrêt de la génération de recommandations")
        return state
    
    anomalies = state["anomalies"]
    recommendations: List[Recommendation] = []
    parser = PydanticOutputParser(pydantic_object=RecommendationList)
    prompt_template = ChatPromptTemplate.from_messages([("system",
                                                         "Tu es un ingénieur infrastructure expert analysant des anomalies de monitoring. "
                                                         "Ta tâche est de fournir des recommandations d'optimisation concrètes et actionnables basées sur les anomalies détectées,. "
                                                         "Structure ta réponse exactement selon le schéma JSON fourni."
                                                         "\n{format_instructions}"), 
                                                         ("human", 
                                                          "Voici une liste de rapports d'infrastructue avec des anomalies détectées :\n"
                                                          "{anomaly_list}\n\n"
                                                          "Génère une liste de recommandations pour traiter ces problèmes, dans un seul texte. Pour chaque anomalie, fournis un seul objet `Recommendation` "
                                                          "où le champ 'suggestion' regroupe toutes les actions proposées pour cette anomalie, séparées par un saut de ligne et listées par des numéros."
                                                          "S'il n'ya pas d'anomalies, toutes les valeurs sont `None`, alors retourne None aussi pour les suggestions.")
                                                        ])
        
    chain = prompt_template | llm | parser

    if anomalies == []:
        print("Aucune anomalie trouvée dans l'état. Pas de recommandations à générer.")
        return {**state, "recommendations": []}
    
    try:
        print("Appel du LLM (gpt-4o-mini) pour générer les recommandations...")
        response = chain.invoke({
            "format_instructions": parser.get_format_instructions(), 
            "anomaly_list": anomalies
        })

        # state['recommendations'] = response.recommendations
        # print("\n\nappel fini ")

        for a in anomalies :
            if a == {}:
                recommendations.append({})
            elif "CPU" in a['metric'] :
                recommendations.append({"anomalie" : response.recommendations[0].anomalie, "suggestion": response.recommendations[0].suggestion})
            elif "Memory" in a['metric'] :
                recommendations.append({"anomalie" : response.recommendations[1].anomalie, "suggestion": response.recommendations[1].suggestion})
            elif "api_gateway" in a['metric'] :
                recommendations.append({"anomalie" : response.recommendations[2].anomalie, "suggestion": response.recommendations[2].suggestion})
            elif "cache" in a['metric'] :
                recommendations.append({"anomalie" : response.recommendations[3].anomalie, "suggestion": response.recommendations[3].suggestion})
            else :
                recommendations.append({"anomalie" : response.recommendations[4].anomalie, "suggestion": response.recommendations[4].suggestion})
        print(f"Génération réussie de recommandations.")

    except Exception as e:
        print(f"Erreur lors de l'appel LLM ou du parsing de la réponse : {e}")
        return {**state, "recommendations": [], "error": f"Génération de recommandations a échoué: {str(e)}"}

    state["recommendations"] = recommendations
    #print("FINAL STATE : ", state)
    return state


def predict_anomalies_patern(state):
    """ 
    Noeud de prédiction de tendances dans le rapport complet (données historiques)
    à partir des anomalies détectées et de patterns déterminés
    """
    
    print("\n\n### Noeud en cours : Prédiction ###")
    
    if state["error"] is not None:
        print("Erreur détectée dans un noeud précédent. Arrêt de la prédiction de tendances.")
        return state

    historic_data = state["input_data"]
    anomaly_list = state["small_list_anomalies"]
    predictions: List[PatternAnomalies] = []
    parser = PydanticOutputParser(pydantic_object = PatternAnomaliesList)
    prompt_template = ChatPromptTemplate.from_messages([("system",
                                                         "Tu es un ingénieur infrastructure expert analysant des anomalies de monitoring. "
                                                         "Ta tâche est de trouver des tendances dans les timestamps ou ailleurs dans le rapport"
                                                         "qui peuvent entraîner les anomalies détectées. "
                                                         "Structure ta réponse exactement selon le schéma JSON fourni."
                                                         "\n{format_instructions}"), 

                                                         ("human", 
                                                          "Voici une liste d'anomalies rencontrées :\n"
                                                          "{anomaly_list}\n\n"
                                                          "une liste de rapports sur l'infrastructure :\n"
                                                          "{historic_data}\n\n"
                                                          "Génère une liste de patterns dans les rapports d'infrastructure qui ont pour conséquences ces anomalies, dans un seul texte. "
                                                          "Pour chaque pattern, fournis une liste de recommandations pour éviter les différentes anomalies."
                                                          "Pour chaque anomalie, fournis un seul objet `Recommendation` où le champ 'suggestion' regroupe "
                                                          "toutes les actions proposées pour cette anomalie, séparées par un saut de ligne et listées par des numéros."
                                                          "S'il n'ya pas d'anomalies, toutes les valeurs sont `None`, alors retourne None aussi pour les suggestions.")
                                                        ])
        
    chain = prompt_template | llm | parser

    print("Appel du LLM (gpt-4o-mini) pour repérer des tendances et patterns...")
    response = chain.invoke({
        "format_instructions": parser.get_format_instructions(), 
        "anomaly_list": anomaly_list,
        "historic_data": historic_data,
        })
    
    #print(len(response.predictions))
    for pred in response.predictions :
        print("PRED : ",pred)
        predictions.append({"pattern" : pred.pattern, "recommendation": pred.recommendation})
        
    #print(predictions)
    state["predictions"] = predictions
    return state


def file_creation(state):
    """
    Fonction qui récupère le dictionnaire de l'état du graphe et l'enregistre au format .JSON
    """
    
    print("\n\n### Noeud en cours : Création d'un fichier avec les anomalies et les recommandations (état final du graphe) ###")
    #print("FINAL STATE : ", state)
    
    with open("Recommendations/recommendations.json", "w") as outfile:
        json.dump(state, outfile, indent=4, sort_keys=False, ensure_ascii=False) #.encode('utf8')



## Définition du graphe
# Initialisation du graphe
workflow = StateGraph(State)

# Définition des noeuds du graphe
workflow.add_node("ingestion", data_ingestion)
workflow.add_node("analyze", anomalies_detection)
workflow.add_node("recommend", recommandations_generation)
workflow.add_node("predict", predict_anomalies_patern)
workflow.add_node("file_creation", file_creation)

# Début du graphe
workflow.set_entry_point("ingestion")

# Définition des arêtes du graphe
workflow.add_edge("ingestion", "analyze")
workflow.add_edge("analyze", "predict")
workflow.add_edge("predict", "recommend")
workflow.add_edge("recommend", "file_creation")
workflow.add_edge("file_creation", END)


# Compilation du graphe
graph = workflow.compile()


### Exécution
if __name__ == "__main__":
    JSON_FILE = "Data/rapport.json"
    graph.invoke({"input_data" : JSON_FILE}) 