"""Finding offers: search criteria, the solid.jobs API, the relaxation ladder.

Steps 2 and 3. The only model calls pick a division, a category and the
sub-categories; the search and the ladder are plain code.
"""

import html
import json
import os
import re

import requests

import config
from models import SearchQuery
from candidate import profile_summary

# Looks like a space, is not one. It travels out of the API inside &nbsp; and
# would end up in a prompt and in a CV. Written as an escape because the
# character itself is invisible in an editor.
NON_BREAKING_SPACE = " "


def load_divisions(path=config.DIVISIONS_PATH):
    """The categories each division uses. The API does not publish them."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)["divisions"]


def offer_skill_names(offer) -> list[str]:
    """The skill names an offer asks for. The API sends {"level", "name"}."""
    names = []
    for skill in offer.get("skills") or []:
        name = skill.get("name") if isinstance(skill, dict) else str(skill)
        if name and name.strip():
            names.append(name.strip())
    return names


def clean_html(text) -> str:
    """The HTML of an offer description as plain text, before any prompt."""
    if not text:
        return ""

    result = re.sub(r"<\s*li[^>]*>", "\n- ", text, flags=re.IGNORECASE)
    result = re.sub(r"<\s*/?\s*(br|p|div|ul|ol|tr|h[1-6])[^>]*>", "\n",
                    result, flags=re.IGNORECASE)
    result = re.sub(r"<[^>]+>", " ", result)
    result = html.unescape(result).replace(NON_BREAKING_SPACE, " ")

    result = re.sub(r"[ \t]+", " ", result)
    tidy = []
    for line in (line.strip() for line in result.split("\n")):
        if line or (tidy and tidy[-1]):
            tidy.append(line)
    return "\n".join(tidy).strip()


def offer_text(offer):
    """One offer as the prompts see it: plain text, no HTML, no JSON."""
    return (
        f"Tytuł: {offer.get('title', '')}\n"
        f"Firma: {offer.get('company', '')}\n"
        f"Poziom: {offer.get('experienceLevel', '')}\n"
        f"Wymagane umiejętności: {', '.join(offer_skill_names(offer))}\n\n"
        f"{clean_html(offer.get('description'))}"
    )


# --------------------------------------------------------------------------
# An offer that did not come from the search
# --------------------------------------------------------------------------

# The labels a text file may begin with, and the API field each fills. The
# first line that is not one of these starts the description.
OFFER_FILE_LABELS = {
    "tytuł": "title", "tytul": "title", "title": "title",
    "firma": "company", "company": "company",
    "poziom": "experienceLevel", "level": "experienceLevel",
    "umiejętności": "skills", "umiejetnosci": "skills", "skills": "skills",
    "url": "url", "link": "url",
}


def offer_from_text(text, key, default_title=""):
    """One offer, in the shape the API sends, out of a pasted advertisement."""
    lines = (text or "").split("\n")
    fields = {}
    start = 0
    for number, line in enumerate(lines):
        label, colon, value = line.partition(":")
        name = OFFER_FILE_LABELS.get(label.strip().lower())
        if not colon or not name or not value.strip():
            break
        fields[name] = value.strip()
        start = number + 1

    skills = [name.strip() for name in fields.get("skills", "").split(",")
              if name.strip()]
    return {
        "jobOfferKey": key,
        "title": fields.get("title") or default_title,
        "company": fields.get("company", ""),
        "experienceLevel": fields.get("experienceLevel", ""),
        "url": fields.get("url", ""),
        # The same shape the API sends, objects and not strings, so that
        # offer_skill_names reads them without knowing where they came from.
        "skills": [{"name": name} for name in skills],
        "description": "\n".join(lines[start:]).strip(),
    }


def offers_from_file(path):
    """The offers to work on, read from a file instead of searched for.

      a text file      one advertisement, pasted
      one JSON object  one offer in the API's own shape
      a JSON list      several of them
      a run.json       every offer a previous run was asked about
    """
    with open(path, encoding="utf-8") as handle:
        text = handle.read()

    stem = os.path.splitext(os.path.basename(path))[0]
    if not path.lower().endswith(".json"):
        return [offer_from_text(text, key=file_key(stem), default_title=stem)]

    data = json.loads(text)
    if isinstance(data, dict):
        # A whole run, or a whole answer from the API, or one bare offer.
        found = data.get("assessed_offers") or data.get("jobs") or [data]
    else:
        found = data
    return with_keys(found, file_key(stem))


def file_key(stem):
    """An offer key made from a file name: readable and typeable."""
    key = re.sub(r"[^\w-]+", "-", (stem or "").strip().lower(), flags=re.UNICODE)
    return key.strip("-") or "plik"


def with_keys(offers, stem):
    """Every offer needs a key of its own: the flow indexes by key.

    A missing key is filled in from the position; a duplicate stops the run.
    """
    result = []
    seen = set()
    for number, offer in enumerate(offers, start=1):
        one = dict(offer)
        key = str(one.get("jobOfferKey") or "").strip()
        if key and key in seen:
            raise ValueError(
                f"Dwie oferty w pliku mają ten sam klucz {key!r} - "
                f"popraw plik, bo jedna z nich zostałaby pominięta")
        one["jobOfferKey"] = key or f"{stem}-{number}"
        seen.add(one["jobOfferKey"])
        result.append(one)
    return result


# --------------------------------------------------------------------------
# From a candidate to search parameters (step 2)
# --------------------------------------------------------------------------

CHOOSE_SYSTEM = """
Jesteś asystentem, który dopasowuje profil kandydata do sztywnych list
kategorii używanych przez serwis z ofertami pracy.
"""

CHOOSE_DIVISION_USER = """
##Profil kandydata:
[PROFIL]
{summary}
[/PROFIL]

##Wybierz JEDEN dział, w którym ten kandydat powinien szukać pracy.
Dozwolone wartości, i żadne inne - każda z kategoriami stanowisk, które
obejmuje:
{divisions}

##Kieruj się kategoriami, nie brzmieniem nazwy działu. Odpowiedz nazwą działu
##sprzed dwukropka, nie kategorią.

##Zwróć JSON:
{{"division": "wybrana wartość", "rationale": "jedno zdanie uzasadnienia"}}
"""

CHOOSE_CATEGORY_USER = """
##Profil kandydata:
[PROFIL]
{summary}
[/PROFIL]

##Kandydat szuka pracy w dziale: {division}
##Wybierz JEDNĄ kategorię z tej listy, i żadnej innej:
{categories}

##Jeżeli żadna nie pasuje wystarczająco dobrze, zwróć pusty string.
##Zwróć JSON:
{{"category": "wybrana wartość albo pusty string", "rationale": "jedno zdanie"}}
"""


CHOOSE_SUBCATEGORIES_USER = """
##Profil kandydata:
[PROFIL]
{summary}
[/PROFIL]

##Kandydat szuka pracy w dziale: {division}
##Wybierz od jednej do {limit} pozycji z tej listy, i żadnych innych:
{categories}

##Zasady:
+Wybieraj tylko to, co kandydat naprawdę zna z opisu.
+Nie wybieraj na wszelki wypadek. Wskazanie całej listy znaczy dokładnie
 tyle samo, co brak filtra.
+Jeżeli nic nie pasuje wystarczająco dobrze, zwróć pustą listę.

##Zwróć JSON:
{{"subCategories": ["wybrane wartości"], "rationale": "jedno zdanie"}}
"""


def _pick(answer, key, allowed):
    """The model's choice when it is on the list, else "" - not retried."""
    value = str(answer.get(key) or "").strip()
    return value if value in allowed else ""


def _ask(llm, template, values, key, pick):
    """Ask one closed-list question. Returns (what was picked, why).

    An empty answer is retried by ask_json. A value OUTSIDE the list is a
    judgement, not a failure: it is dropped, and the rationale says so, so
    the report never prints a reason for a choice nobody used.
    """
    answer = llm.ask_json(CHOOSE_SYSTEM, template, values,
                          role="criteria_searcher")
    chosen = pick(answer)
    rationale = str(answer.get("rationale") or "")
    raw = answer.get(key)
    if not chosen and raw not in (None, "", []):
        rationale = (f"model wskazał {raw!r}, czego nie ma na liście "
                     f"dozwolonych wartości - odpowiedź pominięta, szukano "
                     f"bez tego filtra. Uzasadnienie modelu: {rationale}")
    return chosen, rationale


def _pick_many(answer, key, allowed, limit):
    """Like _pick, for several values.

    Off-list values and repeats are dropped, order kept, at most limit.
    """
    chosen = []
    for value in answer.get(key) or []:
        value = str(value or "").strip()
        if value in allowed and value not in chosen:
            chosen.append(value)
    return chosen[:limit]


def build_search_query(profile, llm, divisions, settings):
    """The candidate as the parameters of the first search.

    City and salary come from the settings, never from a model here.
    """
    summary = profile_summary(profile)
    names = sorted(divisions)

    # Each division with its categories: the names alone mislead. A software
    # CTO reads as "Engineering" until the model sees that Engineering means
    # Mechatronics and ProductionEngineering while IT means Developer,
    # Architect and ItManager.
    listing = "\n".join(
        f"- {name}: {', '.join(divisions[name]['categories'])}"
        for name in names)
    names_set = set(names)
    division, reason = _ask(
        llm, CHOOSE_DIVISION_USER,
        {"summary": summary, "divisions": listing},
        "division", lambda answer: _pick(answer, "division", names_set))

    why = {"division": reason}
    if not division:
        # "Other" is the absence of a choice, and the report says so.
        division = "Other"
        why["division"] = ("model nie wskazał żadnego z dozwolonych działów - "
                           "użyto Other, czyli braku zawężenia")
        why["division_defaulted"] = True

    categories = divisions[division]["categories"]
    category, reason = _ask(
        llm, CHOOSE_CATEGORY_USER,
        {"summary": summary, "division": division,
         "categories": ", ".join(categories)},
        "category", lambda answer: _pick(answer, "category", set(categories)))
    why["category"] = reason

    # Outside IT subCategory equals category. Inside IT the second level
    # exists and may name several; the API returns the union.
    sub_categories = [category] if category else []
    if division == "IT" and category:
        sub_names = divisions[division]["subCategories"].get(category, [])
        if sub_names:
            sub_categories, reason = _ask(
                llm, CHOOSE_SUBCATEGORIES_USER,
                {"summary": summary, "division": division,
                 "categories": ", ".join(sub_names),
                 "limit": config.MAX_SUBCATEGORIES},
                "subCategories",
                lambda answer: _pick_many(answer, "subCategories",
                                          set(sub_names),
                                          config.MAX_SUBCATEGORIES))
            why["subCategories"] = reason
        else:
            sub_categories = []

    query = SearchQuery(
        division=division,
        location=settings.location or profile.preferred_location,
        category=category,
        subCategories=sub_categories,
        experience=profile.experience_level,
        minimumSalary=int(settings.minimum_salary
                          or profile.preferred_minimum_salary or 0),
    )
    return query, why


# --------------------------------------------------------------------------
# Asking the API, and the ladder that stops it returning nothing (step 3)
# --------------------------------------------------------------------------

def fetch_offers(query):
    """One call to solid.jobs. Returns the parsed answer.

    The city must be spelled in Polish: "Gdansk" gets HTTP 200 and an empty
    list. requests handles the URL encoding.
    """
    params = ""
    if query.location:
        params += f"&search.cities={query.location}"
    if query.category:
        params += f"&search.categories={query.category}"
    # Repeated, not comma-joined: the repeated parameter gives the union,
    # "Python,Java" answers with something else and no complaint.
    for sub_category in query.subCategories:
        params += f"&search.subCategories={sub_category}"
    if query.experience:
        params += f"&search.experiences={query.experience}"
    if query.minimumSalary > 0:
        params += f"&search.minimumSalary={query.minimumSalary}"

    url = config.OFFERS_API_URL.format(division=query.division, params=params)
    answer = requests.get(url, timeout=config.API_TIMEOUT_SECONDS)
    answer.raise_for_status()
    return answer.json()


def relaxed_query(query, level):
    """The search parameters at one rung of the ladder.

    Rung 0 is the full query. Each rung drops one filter: the sub-category,
    then the category, and at the bottom the seniority level with the
    salary. Technology goes before seniority because a Python job read by a
    Java developer is a near miss, while a senior job read by an intern is a
    wasted model call. Salary goes last because a person asked for it.
    """
    if level == 0:
        return query
    if level == 1:
        return query.model_copy(update={"subCategories": []})
    if level == 2:
        return query.model_copy(update={"subCategories": [], "category": ""})
    return SearchQuery(division=query.division, location=query.location)


def search_with_relaxation(query, min_offers=None):
    """Walk down the ladder until there are enough offers. Returns (offers, rung).

    Empty at rung 3 is an answer, not an error.
    """
    if min_offers is None:
        min_offers = config.MIN_OFFERS

    offers = []
    level = 0
    for level in range(4):
        answer = fetch_offers(relaxed_query(query, level))
        offers = answer.get("jobs") or []
        if len(offers) >= min_offers:
            break

    return offers, level
