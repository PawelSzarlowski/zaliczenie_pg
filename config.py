"""Everything that can be configured, in one place.

Three layers, and a later one always wins:

    value written in this file  ->  .env  ->  the command line / the request

Nothing else in the application reads os.environ, and no other module keeps a
threshold, a path or an address of its own. When you want to know what
a run was configured with, this is the only file to look at - and the report a
run writes prints these numbers back, so a finished CV can say what produced
it.
"""

import os

from pydantic import BaseModel, ConfigDict

# Every path in the application is built from this one.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(PROJECT_DIR, ".env")


# --------------------------------------------------------------------------
# Reading .env
# --------------------------------------------------------------------------

def load_env_file(path=ENV_FILE):
    """Put the values of a .env file into the environment. Returns the names.

    A variable already set in the real environment is left alone. Only the
    names are returned, never the values: one of them is an API key.
    """
    names = []
    if not os.path.exists(path):
        return names

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            name = name.strip()
            value = value.strip().strip('"').strip("'")
            if name and name not in os.environ:
                os.environ[name] = value
            names.append(name)
    return names


load_env_file()


def _text(name, fallback):
    return os.environ.get(name, fallback)


def _number(name, fallback, convert):
    """A malformed value in .env falls back to the default, with a warning."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return fallback
    try:
        return convert(raw)
    except ValueError:
        print(f"warning: {name}={raw!r} in .env is not a number, "
              f"using {fallback}")
        return fallback


# --------------------------------------------------------------------------
# Where the models live
# --------------------------------------------------------------------------

OLLAMA_BASE_URL = _text("OLLAMA_BASE_URL", "http://192.168.178.54:11434")

# How long to wait for the offers API. Right after inference the machine
# is still busy, and fifteen seconds is too short.
API_TIMEOUT_SECONDS = 60
# Writing a CV is not a quick call.
CLOUD_TIMEOUT_SECONDS = 300
# The same, for Ollama. ChatOllama has no timeout of its own, so this is
# passed in explicitly.
OLLAMA_TIMEOUT_SECONDS = 300
# How long to wait before each new attempt: about a minute and a half of a
# busy service in total, which a free tier answering 429 needs.
RETRY_DELAYS = (2, 8, 30, 60)


# --------------------------------------------------------------------------
# The three layers of "which model"
# --------------------------------------------------------------------------
#
# A role says what a model is FOR. The catalogue says which model that is.
# The provider says how to reach it. Nothing below this file names a model:
# the code asks for role="cv_writer" and gets whatever these three tables
# point at.
#
#     ROLE_MODELS   "cv_writer"    ->  "minimax-m3-openrouter"
#     MODELS        "minimax-m3-openrouter"   ->  minimax/minimax-m3, at openrouter
#     PROVIDERS     "openrouter"   ->  an address and the name of a key
#
# To change which model writes CVs, change one line of ROLE_MODELS - or set
# ROLE_CV_WRITER in .env, which wins over it.

# Where models can be reached from. "kind" picks the class in providers.py:
# "ollama" for the LangChain Ollama client, "openai" for anything speaking
# the OpenAI chat API - OpenRouter and ExperientialLabs both do.
#
# No key is written down here, only the NAME of the variable it is read from.
PROVIDERS = {
    "ollama": {
        "kind": "ollama",
        "base_url": OLLAMA_BASE_URL,
        "key_env": "",
        "timeout": OLLAMA_TIMEOUT_SECONDS,
    },
    "openrouter": {
        "kind": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "timeout": CLOUD_TIMEOUT_SECONDS,
    },
    "experientiallabs": {
        "kind": "openai",
        "base_url": "https://api.experientiallabs.ai/v1",
        "key_env": "EXPERIENTIALLABS_API_KEY",
        "timeout": CLOUD_TIMEOUT_SECONDS,
    },
    "huggingface": {
        "kind": "openai",
        "base_url": "https://router.huggingface.co/v1",
        "key_env": "HUGGINGFACE_API_KEY",
        "timeout": CLOUD_TIMEOUT_SECONDS,
    },
    "nvidia": {
        "kind": "openai",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "key_env": "NVIDIA_API_KEY",
        "timeout": CLOUD_TIMEOUT_SECONDS,
    },
}

# Every model this application knows about, under a short name of our own.
# What the code needs to know about a model is DECLARED here, never read
# out of its name.
#
#   family          the model that writes a CV may not share it with the one
#                   that reviews it - a model grading its own work grades it
#                   kindly, and that grade drives the improvement loop.
#   supports_tools  whether the agent can run on it (bind_tools).
#   clean_prompt    optional: take the padding characters out of what
#                   this model is sent - see llm.without_padding.
#   clean_answer    optional: take them out of what it answers, before
#                   the rest of the code sees it. One model padded a CV
#                   with 126 424 em spaces. Both are OFF unless written
#                   down here: by default an answer is kept as it came,
#                   rubbish included, and the file on disk shows it.
#
# Adding a model is one entry here plus one line in ROLE_MODELS.
MODELS = {
    "gemma4:e2b-local": {
        # Ollama Cloud - billed against the account's subscription, not per
        # token. From our side the call is free.
        "provider": "ollama",
        "model_id": "gemma4:e2b-it-qat",
        "family": "gemma",
        "supports_tools": True,
    },
    "nemotron-3-ultra-ollama": {
        # Ollama Cloud - billed against the account's subscription, not per
        # token. From our side the call is free.
        # As reviewer, measured on one CV x3: no false finding, marks within
        # 4 points - but 37 s per call, five times gpt-oss:120b.
        "provider": "ollama",
        "model_id": "nemotron-3-ultra:cloud",
        "family": "nemotron",
        "supports_tools": True,
    },
    "nemotron-3-super-ollama": {
        # Ollama Cloud - billed against the account's subscription, not per
        # token. From our side the call is free.
        # As reviewer, measured on one CV x3: the same CV marked 70.8, 41.0
        # and 43.5. Not a reviewer.
        "provider": "ollama",
        "model_id": "nemotron-3-super:cloud",
        "family": "nemotron",
        "supports_tools": True,
    },
    "nemotron-3-nano-ollama": {
        # Ollama Cloud - billed against the account's subscription, not per
        # token. From our side the call is free.
        "provider": "ollama",
        "model_id": "nemotron-3-nano:30b-cloud",
        "family": "nemotron",
        "supports_tools": True,
    },
    "gemma4:31b-ollama": {
        # Ollama Cloud - billed against the account's subscription, not per
        # token. From our side the call is free.
        # As reviewer, measured on one CV x3: no false finding, the same
        # 15-16 inferences every time, 6 s per call. Same family as the
        # writer it usually pairs with, so pre-flight refuses that pair.
        "provider": "ollama",
        "model_id": "gemma4:31b-cloud",
        "family": "gemma",
        "supports_tools": True,
    },
    "gpt-oss:120b-ollama": {
        # Ollama Cloud - billed against the account's subscription, not per
        # token. From our side the call is free.
        # As reviewer, measured on one CV x3: 8 s per call, but 2 of 3 fact
        # checks reported skills the base holds as missing - it rejects a
        # correct CV now and then.
        # As writer, measured once: after a 1163-character CV it added
        # 126 424 em spaces (U+2003), which blew the reviewer past its
        # context window and cost the offer its CV.
        "provider": "ollama",
        "model_id": "gpt-oss:120b-cloud",
        "family": "gpt",
        "supports_tools": True,
        "clean_answer": True,
    },
    "gpt-oss:20b-ollama": {
        # Ollama Cloud - billed against the account's subscription, not per
        # token. From our side the call is free.
        "provider": "ollama",
        "model_id": "gpt-oss:20b-cloud",
        "family": "gpt",
        "supports_tools": True,
    },
    "glm-5.3-flash-ollama": {
        # Ollama Cloud - billed against the account's subscription, not per
        # token. From our side the call is not free.
        # $0.15 / $0.50 USD/1M
        "provider": "ollama",
        "model_id": "glm-5.3-flash:cloud",
        "family": "glm",
        "supports_tools": True,
    },
    "gpt-6-astra": {
        # 10 / 50 USD per million tokens (input / output), 1 for cached
        # input.
        #
        # THE DEAREST entry in the catalogue: twice the input price of
        # gpt-5.5 and eighty times that of minimax-m3. A review is two calls
        # per CV version, so one run of three CVs at five versions each comes
        # to about 1.60 USD - the same run on gpt-oss at Groq costs about a
        # cent. The measured quality is good (the steadiest marks in the
        # whole study, a spread of 0.8 points over three repetitions), but
        # not that many times better.
        "provider": "experientiallabs",
        "model_id": "gpt-6-astra",
        "family": "gpt",
        "supports_tools": True,
    },
    "gpt-5.4-pro": {
        # The provider's API carries no prices, and on this account the
        # model refuses anyway: HTTP 400 'requires your own provider API
        # key' (BYOK).
        "provider": "experientiallabs",
        "model_id": "gpt-5.4-pro",
        "family": "gpt",
        "supports_tools": True,
    },
    "minimax-m3-openrouter": {
        # 0.30 / 1.20 USD per million tokens (input / output).
        "provider": "openrouter",
        "model_id": "minimax/minimax-m3",
        "family": "minimax",
        "supports_tools": True,
    },
    "nemotron-3-super-openrouter": {
        # FREE (:free) - nothing per token. It is paid for in another coin:
        # a queue, request limits, and 69-87 s per call measured under load.
        "provider": "openrouter",
        "model_id": "nvidia/nemotron-3-super-120b-a12b:free",
        "family": "nemotron",
        "supports_tools": True,
    },
    "nemotron-3-ultra-openrouter": {
        # FREE (:free) - nothing per token. It is paid for in another coin:
        # a queue, request limits, and 69-87 s per call measured under load.
        "provider": "openrouter",
        "model_id": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "family": "nemotron",
        "supports_tools": True,
    },
    "nemotron-3.5-lightning-openrouter": {
        # FREE (:free) - nothing per token. It is paid for in another coin:
        # a queue, request limits, and 69-87 s per call measured under load.
        "provider": "openrouter",
        "model_id": "nvidia/nemotron-3.5-lightning:free",
        "family": "nemotron",
        "supports_tools": True,
    },
    "gpt-oss:120b-openrouter": {
        # 0.04 / 0.17 USD per million - the rate of the CHEAPEST machine,
        # because without a pin OpenRouter picks for itself. Measured:
        # 28.7 s on one prompt, so the routing takes the cheap slow one.
        "provider": "openrouter",
        "model_id": "openai/gpt-oss-120b",
        "family": "gpt",
        "supports_tools": True,
    },
    # The same model id, pinned to one machine. OpenRouter is a broker: asked
    # for openai/gpt-oss-120b it chooses among nineteen providers, and the
    # choice decides the speed. Measured on one 8.5 kB prompt: Cerebras 1.5 s,
    # Groq 6.3 s, DeepInfra 61.6 s, and OpenRouter's own automatic routing
    # 28.7 s. allow_fallbacks is off on purpose - with it on, a busy machine
    # is quietly swapped for another and the number above stops meaning
    # anything.
    "gpt-oss:120b-openrouter-cerebras": {
        # 0.35 / 0.75 USD per million - Cerebras's price, not the model's
        # headline one. Nine times dearer and forty times faster: 1.5 s
        # against 61.6 s at DeepInfra on the same prompt.
        "provider": "openrouter",
        "model_id": "openai/gpt-oss-120b",
        "family": "gpt",
        "supports_tools": True,
        "extra_body": {"provider": {"only": ["Cerebras"],
                                    "allow_fallbacks": False}},
    },
    "gpt-oss:120b-openrouter-groq": {
        # 0.15 / 0.60 USD per million - Groq's price. Measured 6.3 s.
        "provider": "openrouter",
        "model_id": "openai/gpt-oss-120b",
        "family": "gpt",
        "supports_tools": True,
        "extra_body": {"provider": {"only": ["Groq"],
                                    "allow_fallbacks": False}},
    },
    "gpt-oss:20b-openrouter": {
        # 0.03 / 0.13 USD per million (the cheapest machine's rate).
        "provider": "openrouter",
        "model_id": "openai/gpt-oss-20b",
        "family": "gpt",
        "supports_tools": True,
    },
    "gpt-oss:20b-openrouter-groq": {
        # 0.07 / 0.30 USD per million - Groq's price. Measured 7.5 s.
        "provider": "openrouter",
        "model_id": "openai/gpt-oss-20b",
        "family": "gpt",
        "supports_tools": True,
        "extra_body": {"provider": {"only": ["Groq"],
                                    "allow_fallbacks": False}},
    },
    # Three large general-purpose models, all with about a million tokens of
    # context and tool support confirmed in OpenRouter's API. Their families
    # differ, so any two of them can stand as writer and reviewer - except
    # that gpt-5.5 shares the "gpt" family with gpt-oss and gpt-5.4-pro, so
    # not with those.
    #
    # Per million tokens, input/output: qwen 0.32/1.28, gemini 0.75/3.75,
    # gpt-5.5 5.00/30.00. The last is forty times dearer on output than
    # gpt-oss at Groq, and the output is the CV.
    "gemini-3.7-flash-openrouter": {
        # 0.75 / 3.75 USD per million.
        #
        # 8.5 s on the extraction prompt, 60 reads without a single miss.
        # But on the trap document it credits the candidate with ALL FOUR
        # skills she does not have - Kubernetes, Go and Rust from a "what I
        # would like to learn" section, AWS from a course - and it does so
        # every single time. gemma4:31b falls for none of them, and is
        # free.
        "provider": "openrouter",
        "model_id": "google/gemini-3.7-flash",
        "family": "gemini",
        "supports_tools": True,
    },
    "mimo-v2.6-pro-openrouter": {
        # 0.435 / 0.87 USD per million.
        #
        "provider": "openrouter",
        "model_id": "xiaomi/mimo-v2.6-pro",
        "family": "xiaomi",
        "supports_tools": True,
    },
    "gpt-5.5-openrouter": {
        # 5.00 / 30.00 USD per million - THE DEAREST in the catalogue. The
        # output is forty times dearer than gpt-oss at Groq, and the output
        # is the CV: one writing run comes to about 0.47 USD.
        "provider": "openrouter",
        "model_id": "openai/gpt-5.5",
        "family": "gpt",
        "supports_tools": True,
    },
    "granite-4.2-8b-openrouter": {
        # 0.06 / 0.25 USD per million - the cheapest paid entry in the
        # catalogue. Two machines serve it: DeepInfra and CoreWeave.
        # DeepInfra measured as the slowest of everything studied (61.6 s on
        # a prompt Cerebras did in 1.5 s), so if this model were to take a
        # role that is called many times, pin CoreWeave and measure first.
        "provider": "openrouter",
        "model_id": "ibm-granite/granite-4.2-8b",
        "family": "granite",
        "supports_tools": True,
    },
    "qwen3.7-plus-openrouter": {
        # 0.32 / 1.28 USD per million.
        #
        # SLOW: about 48 s on the extraction prompt (8.5 kB of it), measured
        # three times in a row. One spike to 220 s turned up as well. A
        # single machine serves it (Alibaba), so there is no slack and a
        # spike has nowhere to spread. For comparison, gemma4:31b-ollama
        # does the same prompt in 2.1 s.
        "provider": "openrouter",
        "model_id": "qwen/qwen3.7-plus",
        "family": "qwen",
        "supports_tools": True,
    },
    "nemotron-3.5-lightning-nvidia": {
        # This provider's API carries no prices (/models returns nothing but
        # an id) - check on its own site before using it in production.
        "provider": "nvidia",
        "model_id": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "family": "nemotron",
        "supports_tools": True,
    },
    "granite-4.2-30b-huggingface": {
        # 0.16 / 0.65 USD per million - the price of the machine serving
        # it, which is DeepInfra. HuggingFace reports it under
        # providers[].pricing.
        #
        # DO NOT USE for reading the knowledge base or reviewing a CV.
        # Measured: a short prompt goes through in 2.3-2.5 s, but the full
        # extraction prompt (8.5 kB) ends in HTTP 504 from the HuggingFace
        # gateway - DeepInfra does not answer in time. The retry ladder then
        # tries three more times and the run stands still for a quarter of
        # an hour.
        #
        # Its supports_structured_output=false is pessimistic, by the way:
        # response_format json_object goes through without trouble and the
        # model returns valid JSON.
        "provider": "huggingface",
        "model_id": "ibm-granite/granite-4.2-30b",
        "family": "granite",
        "supports_tools": True,
    },
    "llama-3.3-70B-huggingface": {
        # This provider's API carries no prices (/models returns nothing but
        # an id) - check on its own site before using it in production.
        "provider": "huggingface",
        "model_id": "meta-llama/Llama-3.3-70B-Instruct",
        "family": "llama",
        "supports_tools": False,
    },
}

# The five jobs a model is asked to do. Named after the job, never after the
# model or the place it runs in.
ROLES = ("criteria_searcher", "offer_matcher", "cv_writer", "cv_reviewer",
         "cv_agent")

# Which model does which job when .env says nothing. A dict of its own so
# tests can read it without depending on this machine's .env. These five
# must be a configuration that passes pre-flight: cv_writer and cv_reviewer
# from different families, cv_agent with supports_tools.
ROLE_DEFAULT_MODELS = {
    # Steps 1 and 2: read the facts and then the search criteria out of
    # the knowledge base. Simple work, and the cheapest model in the
    # catalogue does it.
    "criteria_searcher": "nemotron-3-super-openrouter",
    # Step 4: how well does this offer fit this candidate.
    "offer_matcher": "nemotron-3-ultra-openrouter",
    # Step 6: writes the CV, and rewrites it.
    "cv_writer": "gpt-oss:120b-openrouter",
    # Step 6: checks the facts and marks the quality. Must be a different
    # family from cv_writer - pre-flight refuses to start otherwise.
    "cv_reviewer": "nemotron-3-ultra-openrouter",
    # Step 6: the agent that owns one offer. Needs supports_tools.
    "cv_agent": "nemotron-3-ultra-openrouter",
}

# The same five, after .env has had its say. ROLE_CRITERIA_SEARCHER,
# ROLE_CV_WRITER and so on - the variable name is the role in capitals.
ROLE_MODELS = {
    role: _text("ROLE_" + role.upper(), default_model)
    for role, default_model in ROLE_DEFAULT_MODELS.items()
}



# --------------------------------------------------------------------------
# The offers API, and the files it is read against
# --------------------------------------------------------------------------

# solid.jobs' public search. "campaign" is our own identifier towards them,
# and it comes back in the url of every offer they return.
OFFERS_API_URL = ("https://solid.jobs/public-api/offers/{division}"
                  "?campaign=cv-generator{params}")

# The two files under data/ are somebody else's vocabulary, not our code:
# what categories solid.jobs uses, and which Polish and English skill names
# mean the same thing.
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DIVISIONS_PATH = os.path.join(DATA_DIR, "division_categories.json")

# The Kaggle resume set profiles.py browses. Git-ignored: 17 MB, and it is
# somebody's resumes.
DATASET_CSV = os.path.join(PROJECT_DIR, "dataset", "resume_data.csv")

# Where a run leaves the CVs and the report.
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results")

# The file inside a run's directory that holds every prompt sent and every
# answer that came back, one JSON line per attempt - see Llm.record_call. A
# name and not a path, because the directory is the run's own and is only
# known once the run starts.
TRANSCRIPT_FILE = "llm_chat_log.jsonl"


# --------------------------------------------------------------------------
# The thresholds and limits of the flow
# --------------------------------------------------------------------------

# Step 3. How many offers are enough to stop relaxing the search filters.
MIN_OFFERS = _number("MIN_OFFERS", 10, int)

# Step 4. How many offers may be put in front of the matching model. A
# ceiling, not a filter: every offer found is scored until it bites, and
# reaching it means the search was too broad, which the report says.
MAX_OFFERS_TO_ASSESS = _number("MAX_OFFERS_TO_ASSESS", 50, int)

# Step 5. The mark out of 100 the matching model must give before a CV is
# written for that offer.
MATCH_SCORE_THRESHOLD = _number("MATCH_SCORE_THRESHOLD", 70, int)

# Step 5. How many offers may reach CV writing, best first. Ten offers at
# five versions each would be two hundred model calls for documents of which
# a person sends three.
MAX_OFFERS_FOR_CV = _number("MAX_OFFERS_FOR_CV", 3, int)

# Step 6. The weighted mark out of 100 at which a CV is good enough to stop.
#
# NOT MEASURED YET: 70 is a starting point chosen to match
# MATCH_SCORE_THRESHOLD, so the two marks a person reads side by side mean
# the same thing. Measured marks so far land between 40 and 67, so in
# practice the loop ends on a wall, not here. Not a sharp line either way:
# the reviewing model is not repeatable, and one document reviewed three
# times can spread 14 points.
CV_QUALITY_THRESHOLD = _number("CV_QUALITY_THRESHOLD", 70, float)

# Step 6. How many documents may exist for one offer, the first one included.
MAX_CV_VERSIONS = _number("MAX_CV_VERSIONS", 5, int)

# How many turns the agent may take before the loop is cut off. Write,
# review, fix, review, fix, review, finish is well inside this; the limit is
# for a model that keeps calling the same tool.
MAX_AGENT_STEPS = _number("MAX_AGENT_STEPS", 16, int)


# --------------------------------------------------------------------------
# How a CV is judged
# --------------------------------------------------------------------------

# The weights of the five criteria the reviewing model marks, 0-100 each.
CRITERION_WEIGHTS = {"A": 0.30, "B": 0.25, "C": 0.20, "D": 0.15, "E": 0.10}

# Where an invention sinks the WHOLE document. Everywhere else it is still
# reported and still stops the loop calling the CV ready. An invented
# employer is not the same mistake as an over-warm word among the skills -
# and an invented skill RAISES the quality mark, because criterion A pays
# for requirements covered, which is why the skills stay outside this gate
# and inside passes_quality.
#
# Stems, not whole names: the section name is free text from a model
# ("edukacja", "WYKSZTAŁCENIE", "EDUCATION"). An unrecognised name does not
# reject, because the everyday unrecognised one is a skill category
# ("narzędzia", "bazy danych"), not a hidden employer.
#
# Checked against the 26 section names seen so far - every skill category,
# every heading spelling, the frame names ("klauzula", "stopka", "dane
# kontaktowe", "nagłówek"): none contains a stem, all 14 spellings of the
# critical sections do. Re-run that check both ways before adding a stem.
# "projekt" is deliberately absent: "projektowanie systemów" is a skill
# category.
CRITICAL_SECTION_STEMS = (
    "doświadcz", "doswiadcz", "experience",
    "wykształ", "wyksztal", "eduka", "education",
    "certyfik", "certif",
    "zatrudni", "employ", "karier",
)

# Two marks this close say nothing about which document is better; inside
# this band agent.pick_best decides on something that is not noise. A
# fifteenth of the scale, a starting point rather than a measurement.
SCORE_NOISE = 7

# How many revisions in a row may fail to improve anything before tool_write_cv
# REFUSES to write another. The prompt asks the agent to stop there; this is
# the wall. A wall refuses and says why; the agent ends through tool_finish_cv.
# Versions written past the point of progress are, measured, worse, the
# same, or carry an invention back in.
MAX_VERSIONS_WITHOUT_PROGRESS = 2


# --------------------------------------------------------------------------
# Reading the candidate, and the language of a CV
# --------------------------------------------------------------------------

# How many technologies one search may name at once. The API takes the
# parameter repeated and answers with the union (the comma form answers with
# something else). Naming all fourteen would equal naming none, while
# reading like narrowing.
MAX_SUBCATEGORIES = 4

# Seniority buckets, left-closed. An assumption, written in one place.
# "Intern" is a level the API has: search.experiences=Intern answers with
# 17 IT offers, beside 49 for Junior.
INTERN_BELOW_YEARS = 0.5
JUNIOR_BELOW_YEARS = 2
REGULAR_BELOW_YEARS = 5

# The share of words carrying a Polish letter that makes a text Polish - the
# tie-break in cv.detect_language. Polish prose runs at a quarter to a half;
# a short English paragraph naming Gdańsk, Kraków and Wrocław reaches 0.12.
POLISH_WORD_SHARE = 0.15


class Settings(BaseModel):
    """The numbers one run works with, built once per run and passed down.

    The defaults are the .env-resolved values above. The report prints
    model_dump() back, so a CV can say what settings produced it.
    """

    model_config = ConfigDict(extra="forbid")

    location: str = ""
    minimum_salary: int = 0
    min_offers: int = MIN_OFFERS
    max_offers_to_assess: int = MAX_OFFERS_TO_ASSESS
    match_score_threshold: int = MATCH_SCORE_THRESHOLD
    max_offers_for_cv: int = MAX_OFFERS_FOR_CV
    cv_quality_threshold: float = CV_QUALITY_THRESHOLD
    max_cv_versions: int = MAX_CV_VERSIONS
    max_agent_steps: int = MAX_AGENT_STEPS
    # Offer keys to write a CV for whatever the assessment says. Naming
    # even one REPLACES the automatic choice: only these are assessed
    # and only these get a CV.
    force_offers: list[str] = []
