import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import (ChatPromptTemplate)
import requests
from langchain_core.output_parsers import StrOutputParser
from models import SearchParams

class Llm:
    API_URL_TEMPLATE = "https://solid.jobs/public-api/offers/{division}?campaign=moja-apka-na-zaliczenie{searchParams}"

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
    MODEL_NAME = "qwen3:30b"
    # llama 3.2
    # qwen3:30b
    # gpt-oss:120b-cloud
    # SpeakLeash/bielik-11b-v2.3-instruct:Q6_K
    # gemma3, deepseek-r1 -< they don't have tools support yet

    llm = ChatOllama(model=MODEL_NAME, base_url=BASE_URL)

    def __init__(self):
        pass

    def fetch_job_offers(self, division: str, sp: SearchParams):

        """Pobiera listę ofert pracy dla wskazanego działu (division) w formacie JSON.

        Args:
            division: Nazwa działu, jedna z:
                "IT", "Engineering", "Marketing", "Sales", "HR", "Logistics",
                "Finances", "Other".

        Returns:
            Surowy tekst JSON zwrócony przez API (jako string).
        """
        if division not in self.VALID_DIVISIONS:
            raise ValueError(
                f"Nieprawidłowy dział '{division}'. Dozwolone wartości: {sorted(self.VALID_DIVISIONS)}"
            )

        params =""
        params +=f"&search.cities={sp.location}" if sp.location else ""
        params +=f"&search.categories={sp.category}" if sp.category else ""
        params +=f"&search.subCategories={sp.subCategory}" if sp.subCategory else ""
        params += f"&search.experiences={sp.experience}" if sp.experience else ""
        if sp.minimumSalary > 0:
            params += f"&search.minimumSalary={sp.minimumSalary}"

        url = self.API_URL_TEMPLATE.format(division=division,searchParams=params)

        print("url ->", url)

        response = requests.get(url, timeout=15)
        response.raise_for_status()
        res = json.loads(response.text)
        return res

    def json_details_of_offer(self, query):
        system_template = '''
            Jesteś asystentem do spraw analizowania tekstów oraz generowania schematów JSON. 
        '''
        user_template='''
            ##Analizując poniższy opis preferencji dotyczących stanowiska pracy:
            [OPIS-STANOWISKA-PRACY]
            {query}
            [/OPIS-STANOWISKA-PRACY]
            
            ##stwórz kod JSON, który będzie wynikiem analizy powyższego opisu stanowiska pracy.
            ###Kod JSON powinien być stworzony według poniższego schematu danych:
            {{
                "location": nazwa miasta, typ string,
                "category": nazwa stanowiska (Dostępne kategorie: ['Administrator', 'Analyst', 'Architect', 'DataScience', 'DevOps', 'Developer', 'ItManager', 'OtherIT', 'Security', 'Support', 'Tester', 'UXUIDesigner']), typ string, 
                "subCategory": pod-kategoria (Dostępne pod-kategorie: ['Analyst', 'Android', 'Angular', 'Architect', 'CCPlusPlus', 'CloudAdministrator', 'DataScience', 'DatabaseAdministrator', 'DevOps', 'DotNet', 'ERP', 'Golang', 'IOS', 'Java', 'JavaScript', 'ManualTester', 'NetworkAdministrator', 'NodeJs', 'OtherAdministrator', 'OtherDeveloper', 'OtherIT', 'OtherItManager', 'OtherTester', 'PHP', 'ProductManager', 'ProductOwner', 'ProjectManager', 'Python', 'React', 'Ruby', 'ScrumMaster', 'Security', 'Support', 'SystemsAdministrator', 'TestAutomationEngineer', 'UXUIDesigner']), typ string,
                "experience": doświadczenie zawodowe (Dostępne doświadczenie zawodowe: ['Junior', 'Regular', 'Senior']), typ string,
                "minimumSalary": oczekiwania finansowe, typ integer,
            }}
            
            ##Jeżeli w opisie stanowiska pracy będzie brakować danych odnośnie: 
            +nazwa miasta,
            +nazwa stanowiska,
            +pod-kategoria,
            +doświadczenie,
            +oczekiwania finansowe [tutaj ustaw "minimumSalary": 0 jeżeli będzie brak w opisie]
            to poprostu te pola ustaw jako pusty string (np. "location":"")
            
            ##wygeneruj tylko kod JSON, bez opisu, bez wyjaśnień
        '''

        prompt = ChatPromptTemplate.from_messages(
            [("system", system_template), ("user", user_template)]
        )

        llm_chain = (
                prompt
                | self.llm
                | StrOutputParser()
        )

        result = llm_chain.invoke(
            {
                "query": query
            }
        )

        print("----------PARAMS----------",result)

        return result.replace("```json","").replace("```","")