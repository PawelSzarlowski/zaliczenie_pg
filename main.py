"""The HTTP interface, for a web front end later on.

    python -m uvicorn main:app --host 127.0.0.1 --port 8000

Interactive documentation at http://127.0.0.1:8000/docs.

None of the logic lives here: POST /oferty and POST /przygotuj call what the
command line calls. GET /modele lists the models a role may be given, GET
/gotowosc checks that they answer.

The slow handlers are plain "def": FastAPI runs them in a thread, while an
"async def" blocking for minutes would freeze the server. POST /przygotuj
takes minutes and will need a job queue in front of it; POST /oferty answers
in seconds.
"""

import config
import offers as offers_module
import pipeline
import candidate as candidate_module
from fastapi import FastAPI, HTTPException
from llm import Llm, LlmOutputError
from models import OffersRequest, PrepareRequest
from candidate import profile_summary

app = FastAPI(title="Agent CV")

# Read once and kept: the data file does not change between requests.
_cache = {}


def divisions():
    if "divisions" not in _cache:
        try:
            _cache["divisions"] = offers_module.load_divisions()
        except FileNotFoundError:
            raise HTTPException(503, "data/division_categories.json is missing.")
    return _cache["divisions"]


def settings_from(request):
    """Whatever the caller set wins; the rest comes from .env and the defaults."""
    given = request.model_dump(exclude_none=True)
    given["minimum_salary"] = given.pop("minimumSalary")
    return config.Settings(**{name: value for name, value in given.items()
                              if name in config.Settings.model_fields})


@app.get("/")
async def alive():
    return {"message": "Agent CV"}


@app.get("/modele")
def models():
    """The catalogue and which model plays which role. Calls no model."""
    try:
        in_use = Llm().models_in_use()
    except LlmOutputError as error:
        raise HTTPException(500, f"The models are misconfigured: {error}")

    return {
        "roles": list(config.ROLES),
        "catalogue": {
            name: {
                "model_id": model["model_id"],
                "provider": model["provider"],
                "family": model["family"],
                "supports_tools": bool(model.get("supports_tools")),
            }
            for name, model in config.MODELS.items()
        },
        "in_use": in_use,
    }


@app.get("/gotowosc")
def readiness():
    """The full pre-flight, including one live call to writer and reviewer.

    Not part of POST /przygotuj: two model calls per request add up. Ask
    once after starting the server. 503, not 500: unreachable models are a
    state that passes, unlike a misconfiguration.
    """
    llm = Llm()
    try:
        llm.preflight()
    except LlmOutputError as error:
        raise HTTPException(503, str(error))
    return {
        "ok": True,
        "models": llm.models_in_use(),
        "retries": llm.retries,
    }


@app.post("/oferty")
def find_offers(request: OffersRequest):
    """Search only - no CV is written, so this answers in seconds."""
    try:
        llm = Llm(role_models=request.models)
    except LlmOutputError as error:
        raise HTTPException(400, str(error))

    try:
        candidate = candidate_module.build_profile(
            request.profile_text, llm, location=request.location,
            minimum_salary=request.minimumSalary)
        settings = config.Settings(location=request.location,
                                   minimum_salary=request.minimumSalary)
        query, why = offers_module.build_search_query(
            candidate, llm, divisions(), settings)
        found, level = offers_module.search_with_relaxation(
            query, min_offers=settings.min_offers)
    except LlmOutputError as error:
        raise HTTPException(502, f"The model returned nothing usable: {error}")
    except Exception as error:
        raise HTTPException(503, f"Could not reach a service: {error}")

    return {
        "profile": profile_summary(candidate),
        "query": dict(query.model_dump(), rationales=why),
        "relaxation_level": level,
        "offers_found": len(found),
        # The offers as the search returned them, trimmed to what a screen
        # needs. This is an order, not a verdict: the judgement of fit is a
        # model call and happens in POST /przygotuj.
        "offers": [{"jobOfferKey": one.get("jobOfferKey", ""),
                    "title": one.get("title", ""),
                    "url": one.get("url", "")} for one in found],
        "message": None if found else
                   "No offers found. Check the spelling of the city - the "
                   "API knows only the Polish one: Gdańsk, not Gdansk.",
    }


@app.post("/przygotuj")
def prepare(request: PrepareRequest):
    """The whole flow. Minutes, not seconds."""
    # 400: the request named a role or model the catalogue does not have.
    # A misconfiguration preflight_configuration finds below is a 500.
    try:
        llm = Llm(role_models=request.models)
    except LlmOutputError as error:
        raise HTTPException(400, str(error))

    settings = settings_from(request)

    # The configuration half of pre-flight, on every request: no model
    # call, and it stops a request that would otherwise hand back a CV
    # graded by its own author. The live half is GET /gotowosc.
    try:
        llm.preflight_configuration()
    except LlmOutputError as error:
        raise HTTPException(500, f"The models are misconfigured: {error}")

    try:
        candidate = candidate_module.build_profile(
            request.profile_text, llm, location=request.location,
            minimum_salary=request.minimumSalary)
        # `or None`: an empty list would mean "offers given, none", which
        # stops the run; a field left out means "search".
        outcome = pipeline.run(candidate, llm, settings,
                               offers=request.offers or None,
                               offers_from="żądanie HTTP" if request.offers else "",
                               divisions=divisions())
    except LlmOutputError as error:
        raise HTTPException(502, f"The model returned nothing usable: {error}")
    except Exception as error:
        raise HTTPException(503, f"Could not reach a service: {error}")

    # Written to disk as well, like a command-line run.
    written = pipeline.write_output(outcome)

    return {
        "outcome": outcome.model_dump(),
        "report_markdown": pipeline.report_text(outcome),
        "files": written,
        # A crashed run comes back as 200 with what it produced, so the
        # failure is said here and not left inside outcome.
        "crashed": outcome.crashed,
    }
