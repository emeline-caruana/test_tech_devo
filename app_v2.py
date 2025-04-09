## Imports
import os
import json
import streamlit as st

# LangGraph pour l'architecture en étapes/noeuds
from pydantic import BaseModel, Field
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

# LangChain pour la génération de recommandations
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


## Récuépration des variables d'environnement (clé API) ##
openai_api_key = os.environ['OPENAI_API_KEY']

## Création d'un client OpenAI pour la génération de recommandations ##
client = OpenAI()
llm = ChatOpenAI(model="gpt-4o-mini")


## Sortie pour les anomalies ##
class Anomalies(BaseModel):
    metric: str = Field(description="Métrique ou élément où il y a une anomalie")
    value: Any = Field(description="La valeur qui pose problème")
    issue: str = Field(description="Les conséquences ou explications de la valeur problématique")
    
## Sortie pour les recommendations ##
class Recommendations(BaseModel):
    anomalie: str = Field(description="Description d'anomalie.s rencontrée.S")
    suggestion: str = Field(description="Action.s suggérée.s pour optimiser l'infrastructure")

class RecommendationList(BaseModel):
    recommendations: List[Recommendations] = Field(description="Une liste de recommandations d'optimisation basées sur les anomalies détectées.")

    
## Définition de l'état du graphe : InputState pour la gestion de l'entrée et OutputState pour la gestion de la sortie ##
class State(TypedDict):
    input_path: str                 
    input_data: Optional[List[Dict[str, Any]]] 
    anomalies: List[Anomalies]     
    recommendations: List[RecommendationList]
    error: Optional[str]   


## Fonctions pour chaque étape de l'architecture ##

def data_ingestion(state):
    """ Noeud d'ingestion des données """
    #print("### Noeud en cours : Ingestion des données ###")
    #print("\tSTATE : ", state)
    try :
        if ".json" in state["input_data"] : 
            with open(state["input_data"], 'r') as file :
                data = json.load(file)
            #print(f"Data ingestion complétée pour le fichier : { state['input_data'] }\n")

        else :
            data = state["input_data"]
            #print(f"Data ingestion complétée")
        return {**state, "input_data": data, "error": None}
        
    except Exception as e :
        #print(f"Erreur pendant l'ingestion du fichier : {e}\n")
        return {**state, "input_data": {}, "error": f"Ingestion failed: {str(e)}"}
    

def anomalies_detection(state):
    """ Noeud de détection d'anomalies dans les donées """
    #print("\n\n### Noeud en cours : Détection des anomalies ###")
    #print("\tSTATE : ", state)

    if state["error"] != None:
        #print("Erreur détectée dans un noeud précédent. Arrêt de la génération de recommandations\n")
        return state
    
    if type(state["input_data"]) == dict : 
        state["input_data"] = [state["input_data"]]
    data = state["input_data"]
    anomalies_detected: List[Anomalies] = []
        
    for d in data :
        if d["cpu_usage"] > 80 :
            anomalies_detected.append({"metric": "cpu_usage","value": d["cpu_usage"],"issue": "High CPU Usage (> 80%)"})
        if d["memory_usage"] > 80 :
            anomalies_detected.append({"metric": "memory_usage","value": d["memory_usage"],"issue": "High Memory Usage (> 80%)"})

        service_status = d["service_status"]
        for service, status in service_status.items():
            if status != "online":
                anomalies_detected.append({"metric": f"service_status : {service}", "value": status, "issue": f"Service {service} is {status}"})
        
    #print(f"{len(anomalies_detected)} anomalie.s détectée.s")
    #print("\n\tANOMALIES DETECTED ", anomalies_detected)
    state["anomalies"] = anomalies_detected
    return state

def recommandations_generation(state):
    """ Noeud de génération de recommandations à partir des anomalies détectées """
    #print("\n\n### Noeud en cours : Génération de recommandations ###")
    #print("STATE : ", state)
    
    if state["error"] != None:
        #print("Erreur détectée dans un noeud précédent. Arrêt de la génération de recommandations")
        return state
    
    anomalies = state["anomalies"]    
    recommendations: List[Recommendations] = [] 
    parser = PydanticOutputParser(pydantic_object=RecommendationList)
    prompt_template = ChatPromptTemplate.from_messages([("system",
                                                         "Tu es un ingénieur infrastructure expert analysant des anomalies de monitoring. "
                                                         "Ta tâche est de fournir des recommandations d'optimisation concrètes et actionnables basées sur les anomalies détectées,. "
                                                         "Structure ta réponse exactement selon le schéma JSON fourni."
                                                         "\n{format_instructions}"), 
                                                         ("human", 
                                                          "Voici les anomalies détectées sur l'infrastructure :\n"
                                                          "{anomaly_list}\n\n"
                                                          "Génère une liste de recommandations pour traiter ces problèmes, dans un seul texte.  Pour chaque anomalie, fournis un seul objet `Recommendations` "
                                                          "où le champ 'suggestion' regroupe toutes les actions proposées pour cette anomalie, séparées par un saut de ligne et listées par des numéros.")
                                                        ])
        
    chain = prompt_template | llm | parser
    

    if anomalies == []:
        #print("Aucune anomalie trouvée dans l'état. Pas de recommandations à générer.")
        return {**state, "recommendations": []}
    
    for a in anomalies:
        formatted_anomalies = f"- Métrique: {a['metric']}, Valeur: {a['value']}, Problème: {a['issue']}"
        try:
            #print("Appel du LLM (gpt-4o-mini) pour générer les recommandations...")
            response = chain.invoke({
                "format_instructions": parser.get_format_instructions(), 
                "anomaly_list": formatted_anomalies
            })
            
            # recommendations.append(response.recommendations)
            recommendations.append({"anomalie" : response.recommendations[0].anomalie, "suggestion": response.recommendations[0].suggestion})
            #print(f"Génération réussie de {len(recommendations)} recommandations.")

        except Exception as e:
            #print(f"Erreur lors de l'appel LLM ou du parsing de la réponse : {e}")
            return {**state, "recommendations": [], "error": f"Génération de recommandations a échoué: {str(e)}"}

    state["recommendations"] = recommendations
    #print("FINAL STATE : ", state)
    return state

def file_creation(state):
    #print("\n\n### Noeud en cours : Création d'un fichier avec les anomalies et les recommandations (état final du graphe) ###")
    #print("FINAL STATE : ", state)

    with open("Recommendations/recommandations.json", "w") as outfile:
        json.dump(state, outfile, indent=4, sort_keys=False, ensure_ascii=False)#.encode('utf8')



## Définition du graphe ##
# Initialisation du graphe
workflow = StateGraph(State)

# Définition des noeuds du graphe
workflow.add_node("ingestion", data_ingestion)
workflow.add_node("analyze", anomalies_detection)
workflow.add_node("recommend", recommandations_generation)
workflow.add_node("file_creation", file_creation)

# Début du graphe
workflow.set_entry_point("ingestion")

# Définition des arêtes du graphe
workflow.add_edge("ingestion", "analyze")
workflow.add_edge("analyze", "recommend")
workflow.add_edge("recommend", END)
# workflow.add_edge("recommend", "file_creation")
# workflow.add_edge("file_creation", END)

# Compilation du graphe
graph = workflow.compile()
graph = workflow.compile()


## Fonction d'invocation du graphe pour l'application Streamlit ##
def invoke_graph(input):
    if not isinstance(input, dict):
        if not isinstance(input, str):
            raise TypeError("input must be a dict or a str (file path)")
    return graph.invoke({"input_data": input})


## Application Streamlit ##
def main():
    st.title("Interface de détection d'anomalies et de recommandations")

    st.subheader("Entrez les données d'infrastructure au format JSON")
    input_json = st.text_input("Données de rapport JSON")

    if st.button("Lancer l'analyse"):
        try:
            data = json.loads(input_json)
            if not isinstance(data, dict):
                st.error("Veuillez entrer un dict JSON.")
            else:
                with st.spinner("Analyse en cours..."):
                    result = invoke_graph(data)
                st.header("Résultats de l'analyse :")

                if result.get("anomalies"):
                    st.subheader("Anomalies détectées:")
                    for anomaly in result["anomalies"]:
                        st.write(f"\t**Métrique :** :red[{anomaly['metric']}]")
                        st.write(f"\t**Valeur :** :orange[{anomaly['value']}]")
                        st.write(f"\t**Problème :** :orange[{anomaly['issue']}]")
                        st.divider()
                else:
                    st.info("Aucune anomalie détectée.")

                if result.get("recommendations"):
                    st.subheader("Recommandations:")
                    for recommendation_list in result["recommendations"]:
                        st.write(f"**Anomalie:** {recommendation_list['anomalie']}")
                        st.write(f"  **Suggestions:**")
                        for i, suggestion in enumerate(recommendation_list['suggestion'].strip().split('\n')):
                            st.write(f"\t{suggestion.strip()}")
                        st.divider()
                else:
                    st.info("Aucune recommandation générée.")

                if result.get("error"):
                    st.error(f"Une erreur s'est produite: {result['error']}")

        except json.JSONDecodeError:
            st.error("Erreur: Le format JSON entré n'est pas valide.")
        except Exception as e:
            st.error(f"Une erreur inattendue s'est produite: {e}")



### Exécution
if __name__ == "__main__":
    main()