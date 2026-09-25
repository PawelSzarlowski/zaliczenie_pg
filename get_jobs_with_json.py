from __future__ import annotations

import json
from typing import Any, Optional, TypedDict
import io

import requests
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import (ChatPromptTemplate)
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader

from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_openai import OpenAIEmbeddings
import tempfile
from functools import partial
from langchain_huggingface import HuggingFaceEmbeddings

API_URL_TEMPLATE = "https://solid.jobs/public-api/offers/{division}?campaign=moja-apka-na-zaliczenie"

VALID_DIVISIONS = {
    "IT",
    "Engineering",
    "Marketing",
    "Sales",
    "HR",
    "Logistics",
    "Finances",
    "Other",
}

# Configuration
BASE_URL = "http://localhost:11434"
MODEL_NAME = "gpt-oss:120b-cloud"
# llama 3.2
# qwen3:30b
# gpt-oss:120b-cloud
# SpeakLeash/bielik-11b-v2.3-instruct:Q6_K
# gemma3, deepseek-r1 -< they don't have tools support yet

llm = ChatOllama(model=MODEL_NAME, base_url=BASE_URL)

# ---------------------------------------------------------------------------
# 1. TOOL: pobieranie danych JSON z API dla wskazanego działu (division)
# ---------------------------------------------------------------------------
@tool
def fetch_job_offers(division: str) -> str:
    """Pobiera listę ofert pracy dla wskazanego działu (division) w formacie JSON.

    Args:
        division: Nazwa działu, jedna z:
            "IT", "Engineering", "Marketing", "Sales", "HR", "Logistics",
            "Finances", "Other".

    Returns:
        Surowy tekst JSON zwrócony przez API (jako string).
    """
    if division not in VALID_DIVISIONS:
        raise ValueError(
            f"Nieprawidłowy dział '{division}'. Dozwolone wartości: {sorted(VALID_DIVISIONS)}"
        )
    url = API_URL_TEMPLATE.format(division=division)

    print("url ->", url)

    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return response.text

# ---------------------------------------------------------------------------
# 2. Definicja stanu grafu (LangGraph State)
# ---------------------------------------------------------------------------

class GraphState(TypedDict, total=False):
    division: str
    raw_jobs: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# 3. Węzeł: human-in-the-loop — wybór działu (division)
# ---------------------------------------------------------------------------
def collect_division_node(state: GraphState) -> GraphState:
    """Zatrzymuje graf i prosi użytkownika o wybór działu (division)."""

    prompt_payload = {
        "message": "Wybierz dział, dla którego mają zostać pobrane oferty pracy.",
        "valid_divisions": sorted(VALID_DIVISIONS),
    }

    user_input: Any = interrupt(prompt_payload)

    division = str(user_input).strip()
    # Dopasowanie bez uwzględniania wielkości liter, ale zachowanie
    # kanonicznej pisowni z VALID_DIVISIONS.
    canonical = {d.lower(): d for d in VALID_DIVISIONS}
    if division.lower() not in canonical:
        # Fallback na "Other", jeśli podano coś spoza dozwolonego zbioru.
        print(
            f"[collect_division_node] Nieznany dział '{division}', "
            f"przyjęto domyślnie 'Other'."
        )
        division = "Other"
    else:
        division = canonical[division.lower()]

    print(f"[collect_division_node] Wybrany dział: {division}")
    return {"division": division}

# ---------------------------------------------------------------------------
# 4. Węzeł: pobranie ofert (wywołanie narzędzia) dla wybranego działu
# ---------------------------------------------------------------------------
def fetch_offers_node(state: GraphState) -> GraphState:
    division = state.get("division", "IT")
    raw_json = fetch_job_offers.invoke({"division": division})
    data = json.loads(raw_json)
    jobs = data.get("jobs", [])
    print(f"[fetch_offers_node] Pobrano {len(jobs)} ofert pracy dla działu '{division}'.")
    return {"raw_jobs": jobs}


# ---------------------------------------------------------------------------
# 7. Węzeł: prezentacja wyników
# ---------------------------------------------------------------------------
def present_results_node(state: GraphState) -> GraphState:
    jobs = state.get("raw_jobs", [])
    print("TO JEST present_results_node")
    i = 0
    jobs_context="["
    length_job = len(jobs)
    for job in jobs:

        salary = job.get("salary", {})
        jobs_context +=f"""
            {{ 
            "jobOfferKey": "{job.get('jobOfferKey')}",   
            "title": "{str(job.get('title')).replace('\n', ' ')}",
            "category": "{job.get('category')} {job.get('subCategory')}", 
            "company": "{job.get('company')}",
            "salary": "{salary.get('from')}-{salary.get('to')}", 
            "currency": "{salary.get('currency')}", 
            "locations": "{', '.join(job.get('locations', []))}",
            "remote": "{job.get('isRemote')}",
            "hybrid": "{job.get('isHybrid')}", 
            "benefits": "{", ".join(job.get("benefits", [])).replace('"', '\\"')}",
            "URL": "{job.get('url')}"
            }}
        """
        if i < length_job-1: jobs_context+=","
        i+=1

    jobs_context +="]"

    # print(jobs_context)

    # system_template='''
    #         You are an AI assistant responsible for extracting and filtering job offers from the provided RAG context.
    #
    #         ## Task
    #         Analyze only the retrieved context. Do NOT use external knowledge or generate fictional job offers.
    #
    #         Filter job offers according to the user's search criteria:
    #         - title
    #         - category
    #         - location
    #
    #         Only include jobs that satisfy all provided filters. If a filter is empty or not provided, ignore it.
    #
    #         ## Input
    #         User filters:
    #         {{
    #             "title": "",
    #             "category": "",
    #             "location": ""
    #         }}
    #
    #         Retrieved context:
    #         {{context}}
    #
    #         ## Output Requirements
    #
    #         Return ONLY a valid JSON array.
    #
    #         Each object must follow this schema exactly:
    #
    #         ```json
    #         {{
    #           "jobOfferKey": "",
    #           "title": "",
    #           "category": "",
    #           "company": "",
    #           "locations": "",
    #           "remote": true,
    #           "hybrid": false,
    #           "benefits": "",
    #           "url": ""
    #         }}
    #         ```
    #
    #         ## Extraction Rules
    #
    #         - jobOfferKey
    #           - Use the unique identifier from the source.
    #           - If unavailable, use an empty string.
    #
    #         - title
    #           - Extract the exact job title.
    #
    #         - category
    #           - Extract the job category from the source.
    #           - If unavailable, infer it only when it is clearly evident from the job title.
    #           - Otherwise return an empty string.
    #
    #         - company
    #           - Extract the company name.
    #
    #         - locations
    #           - Return the location exactly as provided in the source.
    #           - If multiple locations exist, return them as a comma-separated string.
    #
    #         - remote
    #           - true only if the offer explicitly states Remote or Fully Remote.
    #           - Otherwise false.
    #
    #         - hybrid
    #           - true only if the offer explicitly states Hybrid.
    #           - Otherwise false.
    #
    #         - benefits
    #           - Return a concise comma-separated summary of benefits.
    #           - If none are mentioned, return an empty string.
    #
    #         - url
    #           - Return the original job posting URL.
    #           - If unavailable, return an empty string.
    #
    #         ## Important Rules
    #
    #         - Use ONLY information from the retrieved context.
    #         - Do NOT invent or infer missing values unless explicitly allowed above.
    #         - Do NOT include explanations, markdown, or additional text.
    #         - Return only the JSON array.
    #         - If no matching jobs are found, return:
    #
    #         ```json
    #         []
    #         ```
    # '''

    system_template = '''
        You are job offer analytic. Find every proper job offer (focus on json fileds like: title,category,locations) by using below json data format:

        [JSON-DATA]
          {{
            'jobOfferKey': '49966611-3db1-4781-82d3-7bb8d745e964',
            'title': 'Calypso Developer',
            'category': 'Developer, Java',
            'company': 'Name of Company',
            'locations': 'Warszawa',
            'remote': false,
            'hybrid': true,
            'benefits': 'some benefits',
            'url': ''
          }}
        [/JSON-DATA]

        and data context repository:

        [JSON-DATA]

        {context}

        [/JSON-DATA]

        Return data like below JSON format:
        {{
            'jobOfferKey': '',
            'title': '',
            'category': '',
            'company': '',
            'locations': '',
            'remote': true/false,
            'hybrid': true/false,
            'benefits': 'some benefits',
            'url': ''
        }}
    '''

    user_template = '''
        query: {query}
    '''

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", user_template)]
    )

    print("Describe what kind of job are you looking for?: ")
    query=input()

    results = getVectorStore(jobs_context).similarity_search_with_score(query, k=500)

    # print('results',results)

    PROG_ODCIECIA = 0.8

    # 2. Przefiltruj wyniki - dopuść tylko te o małej odległości (wysokim podobieństwie)
    valid_docs = []
    for doc, score in results:
        # print('doc->',doc)
        if score <= PROG_ODCIECIA:
            valid_docs.append(doc)
        else:
            pass
            # print(f"Odrzucono dokument (zbyt słabe dopasowanie). Score: {score}")

    # 3. Przekazanie do LLM lub obsługa braku wiedzy
    if not valid_docs:
        # Zamiast pytać LLM i ryzykować halucynację, od razu zwracasz bezpieczną odpowiedź:
        context_for_llm = ""
        # print("Brak pasujących danych w bazie JSON dla tego zapytania.")
    else:
        # Budujesz kontekst z bezpiecznych, zweryfikowanych dokumentów
        context_for_llm = "\n\n".join([d.page_content for d in valid_docs])

    retrieved_docs = context_for_llm

    print("------------retrieved_docs----------------------\n ",retrieved_docs)

    llm_chain = (
            prompt
            | llm
            | StrOutputParser()
    )

    result = llm_chain.invoke(
        {
            "context": retrieved_docs,
            "query": query,
        }
    )

    print(result)

    return {}

def getVectorStore(json_text):

    # Tworzenie tymczasowego pliku
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json", encoding="utf-8") as temp_file:
        temp_file.write(json_text)
        temp_file.flush()
        temp_path = temp_file.name

    json.loads = partial(json.loads, strict=False)

    loader = JSONLoader(
        file_path=temp_path,
        # jq_schema='.[] | "\\(.title) , \\(.category) , \\(.locations)"',
        jq_schema='.[]',
        # content_key="category",
        text_content=False
    )
    documents = loader.load()

    docs = RecursiveCharacterTextSplitter(chunk_size=500).split_documents(documents)

    # print("------------getVectorStore docs----------------------", docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store


# ---------------------------------------------------------------------------
# 8. Budowa grafu LangGraph
# ---------------------------------------------------------------------------
def build_graph():
    graph_builder = StateGraph(GraphState)

    graph_builder.add_node("collect_division", collect_division_node)
    graph_builder.add_node("fetch_offers", fetch_offers_node)
    graph_builder.add_node("present_results", present_results_node)


    graph_builder.add_edge(START, "collect_division")
    graph_builder.add_edge("collect_division", "fetch_offers")
    graph_builder.add_edge("fetch_offers", "present_results")
    graph_builder.add_edge("present_results", END)

    # Checkpointer jest wymagany, aby mechanizm interrupt()/Command(resume=...)
    # mógł zapamiętać stan grafu pomiędzy zatrzymaniem a wznowieniem.
    checkpointer = MemorySaver()
    return graph_builder.compile(checkpointer=checkpointer)

# ---------------------------------------------------------------------------
# 9. Uruchomienie: pętla obsługująca human-in-the-loop w konsoli
# ---------------------------------------------------------------------------
def run_cli():
    graph = build_graph()
    config = {"configurable": {"thread_id": "job-offers-session-1"}}

    # Pierwsze wywołanie - graf zatrzyma się na pierwszym interrupt()
    # (wybór działu w collect_division_node).
    result = graph.invoke({}, config=config)

    # print("to jest RESULT")
    # print (result)

    while "__interrupt__" in result:
        interrupt_obj = result["__interrupt__"][0]
        payload = interrupt_obj.value

        if "valid_divisions" in payload:
            # --- Zatrzymanie: wybór działu (division) ---
            print("\n" + payload["message"])
            print(f"Dozwolone działy: {payload['valid_divisions']}\n")

            division_input = input("Podaj dział: ").strip()
            result = graph.invoke(Command(resume=division_input), config=config)

if __name__ == "__main__":
    run_cli()