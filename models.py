"""The shapes that travel between the modules. One candidate at a time."""

from typing import Literal

from pydantic import BaseModel

# The eight divisions solid.jobs splits its offers into. Their names come from
# the API, so they stay in English and spelled its way.
Division = Literal[
    "IT", "Engineering", "Marketing", "Sales", "HR", "Logistics",
    "Finances", "Other",
]

# The four levels the API knows, plus "no preference" as an empty string.
# Closed: a level the API does not know would silently match nothing.
Experience = Literal["", "Intern", "Junior", "Regular", "Senior"]

# Closed, because this value picks the text of a legal clause.
Language = Literal["pl", "en"]


class CandidateProfile(BaseModel):
    """Everything known about the one candidate this run is about.

    raw_text is the knowledge base as it arrived and the only source of
    facts a CV may be built from. The parsed fields are for searching and
    matching; a model extracted them, a human wrote raw_text.
    """

    positions: list[str] = []
    skills: list[str] = []
    companies: list[str] = []
    education: list[str] = []
    years_of_experience: float | None = None   # None = could not be worked out
    experience_level: Experience = ""
    raw_text: str

    # Preferences. They may come from the knowledge base, but a value given on
    # the command line or in the request always wins over what a model read.
    preferred_location: str = ""
    preferred_minimum_salary: int = 0


class SearchQuery(BaseModel):
    """What goes to the offers API.

    category and subCategories are plain strings, checked against
    data/division_categories.json in code: every division has its own list.
    subCategories is a list because the API takes the parameter repeated and
    answers with the union.
    """

    division: Division
    location: str = ""
    category: str = ""
    subCategories: list[str] = []
    experience: Experience = ""
    minimumSalary: int = 0


class MatchAssessment(BaseModel):
    """What the matching model said about one offer (step 4)."""

    jobOfferKey: str
    score: int                      # 0-100
    rationale: str = ""


class CvReview(BaseModel):
    """What the reviewing model said about one CV (step 6).

    unsupported is the hard part: an invented fact may reject the document
    whatever it scores. The marks are the soft part and drive the loop.
    """

    unsupported: list[str] = []     # sentences nothing in the base supports
    # Beyond the letter of the base but following from it - "lists Kafka"
    # becoming "built services with Kafka". Worth softening, never a
    # reason to reject.
    inferred: list[str] = []
    hard_fail: bool = False
    # Wrong with the document itself: a mangled consent clause, a filled-in
    # contact block. Found by code.
    document_defects: list[str] = []
    scores: dict[str, int] = {}     # "A".."E", each 0-100
    analysis: dict[str, str] = {}
    total: float | None = None      # weighted, out of 100
    top_fix: str = ""


class CvVersion(BaseModel):
    """One document, with the review it earned. Version 1 is the first draft."""

    iteration: int
    markdown: str
    word_count: int = 0
    agent_notes: str = ""           # what the agent asked for before writing it
    review: CvReview | None = None


class OfferOutcome(BaseModel):
    """Everything that happened for one offer that reached CV writing."""

    jobOfferKey: str
    title: str
    url: str = ""
    match_score: int = 0
    match_rationale: str = ""
    # Decided once per offer, before the first word is written.
    language: Language = "pl"
    versions: list[CvVersion] = []
    # Which of `versions` is the one to send; 0 when none is (no versions,
    # or none reviewed). Every reader has to allow for 0.
    final_iteration: int = 1
    # "The fact check threw the chosen document out" - and nothing wider.
    # Not the same as review.passes_quality, which also wants no unsupported
    # sentence and the mark at the threshold. Every live path sets this;
    # the default is only for a run.json without the field, where True
    # would falsely stamp every CV as rejected.
    final_rejected_by_fact_check: bool = False
    # Asked for by name, so the fit threshold was not applied: a poor match
    # score reads differently.
    forced: bool = False
    agent_steps: int = 0
    agent_stop_reason: str = ""


class RunOutcome(BaseModel):
    """What one run produced. This is what the report is written from."""

    settings: dict = {}
    # Empty when the offers were searched for; otherwise where they came
    # from - a file, or the body of an HTTP request.
    offers_from: str = ""
    search_query: dict = {}
    relaxation_level: int = 0
    offers_found: int = 0
    assessed: list[MatchAssessment] = []   # every offer the model scored
    # The offers behind `assessed`, as the API sent them, so a run can be
    # repeated for one offer without searching again. Only the assessed
    # ones: a whole division is 500 offers and 2.5 MB, and this goes into
    # run.json and every POST /przygotuj answer.
    assessed_offers: list[dict] = []
    outcomes: list[OfferOutcome] = []      # those that got a CV
    stopped_reason: str | None = None
    # True when the run ended by raising. What it produced is still here and
    # still written to disk; the report and the exit code say it is partial.
    crashed: bool = False
    seconds: dict[str, float] = {}
    # Which model played which role, and what it cost - the roles can be
    # set per run.
    models: dict = {}
    usage: dict = {}
    # Things a person reading the report has to know: no offer at the
    # candidate's level, the ladder's bottom reached, an offer not scored.
    warnings: list[str] = []


# --------------------------------------------------------------------------
# The HTTP request bodies
# --------------------------------------------------------------------------

class PrepareRequest(BaseModel):
    """POST /przygotuj - the knowledge base and the preferences.

    Every threshold is optional; a caller that sets none gets .env's values.
    """

    profile_text: str
    location: str = ""
    minimumSalary: int = 0
    # Offers to work on instead of searching, in the API's shape. A path is
    # not accepted: no file is read on the server.
    offers: list[dict] = []
    # Offer keys to write a CV for whatever the assessment says; naming one
    # replaces the automatic choice. Whole or a prefix.
    force_offers: list[str] = []
    # Role -> name from config.MODELS, for the roles to change. An unknown
    # name is refused before the run starts.
    models: dict[str, str] = {}
    max_offers_to_assess: int | None = None
    match_score_threshold: int | None = None
    max_offers_for_cv: int | None = None
    cv_quality_threshold: float | None = None
    max_cv_versions: int | None = None


class OffersRequest(BaseModel):
    """POST /oferty - search and rank only, no CV writing."""

    profile_text: str
    location: str = ""
    minimumSalary: int = 0
    models: dict[str, str] = {}
