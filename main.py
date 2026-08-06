from fastapi import FastAPI, Response
import json
from llm import Llm
from models import Userquery, SearchParams

app = FastAPI()

@app.get("/")
async def helloworld():
    return {"message": "Hello JobOffers"}


@app.post("/joboffersjson")
async def joboffersjson(userquery: Userquery):
    llm = Llm()
    searchParams = llm.json_details_of_offer(userquery.job_desc)
    data = json.loads(searchParams)
    searchParams_data = SearchParams(**data)
    return llm.fetch_job_offers("IT", searchParams_data)