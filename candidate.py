"""The candidate: one knowledge base, read from a text file into a profile.

Step 1. One model call copies the facts out of the base; the length of the
career is counted here in code, from the dates the model copied.

raw_text is the knowledge base as it arrived - the only source of facts for a
CV and the only thing the fact check compares it against. The base is
expected to hold no personal data; the CV keeps placeholders for those.
"""

import re

import config
from models import CandidateProfile

_YEAR = re.compile(r"(19|20)\d{2}")

# Months as they really appear in the data: "Jan 2019", "July 2021",
# "September 2018" and the numeric "06/2014". Longer spellings come first so
# that "March" is not read as "Mar" with a stray "ch" left over, and the word
# boundary keeps "Marketing" from being read as March.
_MONTH_NAME = re.compile(
    r"\b(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|"
    r"august|aug|september|sept|sep|october|oct|november|nov|december|dec)\b",
    re.IGNORECASE)
_MONTH_NUMBER = re.compile(r"\b(\d{1,2})\s*[/.-]\s*(?:19|20)\d{2}\b")
_MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun",
                "jul", "aug", "sep", "oct", "nov", "dec"]


# --------------------------------------------------------------------------
# How long somebody has worked
# --------------------------------------------------------------------------

def years_from_periods(starts, ends, this_year=None):
    """Career length in years, with overlapping jobs counted once.

    Counted in months, so two one-month internships and eleven months of
    work differ - the range where Intern and Junior part company. A date
    without a month counts as January. Returns None, not zero, when no
    period could be read: "no dates" and "no time worked" must not look
    alike. Ragged lists are fine: an unreadable start is skipped, a missing
    end means "still working there".
    """
    if this_year is None:
        from datetime import date
        this_year = date.today().year

    periods = []
    for index in range(max(len(starts), len(ends))):
        start = _months_in(starts[index] if index < len(starts) else "")
        if start is None:
            continue
        end = _months_in(ends[index] if index < len(ends) else "")
        if end is None:
            # "Till Date", "Current", "Present" or simply missing.
            end = this_year * 12
        if end < start:
            start, end = end, start
        periods.append((start, end))

    if not periods:
        return None

    periods.sort()
    merged = [list(periods[0])]
    for start, end in periods[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return sum(end - start for start, end in merged) / 12.0


def _months_in(text):
    """A date as a count of months. None without a year; January without a month."""
    text = str(text or "")
    year = _YEAR.search(text)
    if year is None:
        return None

    month = 1
    numeric = _MONTH_NUMBER.search(text)
    named = _MONTH_NAME.search(text)
    if numeric:
        month = min(max(int(numeric.group(1)), 1), 12)
    elif named:
        month = _MONTH_ORDER.index(named.group(1).lower()[:3]) + 1

    return int(year.group(0)) * 12 + month - 1


def experience_level(years):
    """The bucket the offers API understands.

    Empty only for None: zero years is "Intern", not unknown.
    """
    if years is None:
        return ""
    if years < config.INTERN_BELOW_YEARS:
        return "Intern"
    if years < config.JUNIOR_BELOW_YEARS:
        return "Junior"
    if years < config.REGULAR_BELOW_YEARS:
        return "Regular"
    return "Senior"


# --------------------------------------------------------------------------
# Reading the knowledge base
# --------------------------------------------------------------------------

def read_text_file(path):
    """The knowledge base as somebody wrote it. UTF-8, always."""
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


# --------------------------------------------------------------------------
# Turning the knowledge base into a profile
# --------------------------------------------------------------------------

EXTRACT_SYSTEM = """
Jesteś asystentem, który czyta opis doświadczenia zawodowego kandydata
i wypisuje z niego fakty. Niczego nie dopowiadasz i niczego nie zgadujesz.
"""

EXTRACT_USER = """
##Przeczytaj poniższy opis kandydata:
[OPIS-KANDYDATA]
{profile_text}
[/OPIS-KANDYDATA]

##Zwróć JSON dokładnie w tym schemacie:
{{
  "positions": lista stanowisk, typ lista stringów,
  "skills": lista umiejętności i technologii, typ lista stringów,
  "companies": lista firm, typ lista stringów,
  "education": lista uczelni i kierunków, typ lista stringów,
  "years_of_experience": liczba lat doświadczenia albo null, typ liczba,
  "employment_starts": daty rozpoczęcia kolejnych posad, typ lista stringów,
  "employment_ends": daty zakończenia tych samych posad, w tej samej
    kolejności, typ lista stringów,
  "preferred_location": miasto, w którym kandydat chce pracować, typ string,
  "preferred_minimum_salary": oczekiwana stawka miesięczna, typ liczba całkowita
}}

##Zasady:
+Wypisuj WYŁĄCZNIE to, co jest w opisie. Nie dopisuj technologii, których tam
 nie ma, nawet jeżeli "pasują" do stanowiska.
+"years_of_experience" podaj tylko wtedy, gdy opis mówi to wprost albo da się
 to policzyć z podanych dat. W przeciwnym razie wpisz null. NIE zgaduj.
+Zero jest wartością, a nie brakiem wartości. Jeżeli opis mówi o zerowym stażu,
 albo wymienia wyłącznie staże, praktyki i projekty studenckie, wpisz 0.
 null zostaw wtedy, gdy w opisie nie ma o stażu ani słowa i nie ma żadnych dat.
+"employment_starts" i "employment_ends" PRZEPISZ dosłownie z opisu, w tej
 samej kolejności, po jednej pozycji na każdą posadę - także wtedy, gdy była to
 praktyka albo staż. Niczego tu nie licz i niczego nie skracaj: "July 2021" ma
 zostać "July 2021". Jeżeli praca nadal trwa, w "employment_ends" wpisz
 "obecnie". Jeżeli w opisie nie ma żadnych dat, zostaw obie listy puste.
+"preferred_location" i "preferred_minimum_salary" wypełnij tylko wtedy, gdy
 opis o nich mówi. Inaczej wpisz "" oraz 0.
+Zwróć sam JSON, bez komentarza i bez wyjaśnień.
"""


def build_profile(raw_text, llm, location="", minimum_salary=0):
    """Read a knowledge base into a CandidateProfile.

    location and minimum_salary from the caller win over what the model read.

    Career length: computed here from the dates the model copied out of the
    text; only if there are none, the number the model worked out itself. A
    model copies "July 2021 - Aug 2021" reliably and turns two one-month
    internships into null.
    """
    answer = llm.ask_json(EXTRACT_SYSTEM, EXTRACT_USER,
                          {"profile_text": raw_text}, role="criteria_searcher")

    years = years_from_periods(_clean(answer.get("employment_starts")),
                               _clean(answer.get("employment_ends")))
    if years is None:
        years = answer.get("years_of_experience")
        if years is not None:
            try:
                years = float(years)
            except (TypeError, ValueError):
                years = None

    return CandidateProfile(
        positions=_clean(answer.get("positions")),
        skills=_clean(answer.get("skills")),
        companies=_clean(answer.get("companies")),
        education=_clean(answer.get("education")),
        years_of_experience=years,
        experience_level=experience_level(years),
        raw_text=raw_text,
        preferred_location=location or str(answer.get("preferred_location") or ""),
        preferred_minimum_salary=int(
            minimum_salary or answer.get("preferred_minimum_salary") or 0),
    )


def _clean(items):
    return [str(item).strip() for item in (items or []) if str(item).strip()]


def profile_summary(profile):
    """The short form the searching and matching prompts get.

    Only the CV writer and the fact check see raw_text.
    """
    parts = []
    if profile.positions:
        parts.append("Stanowiska: " + ", ".join(profile.positions[:8]))
    if profile.skills:
        parts.append("Umiejętności: " + ", ".join(profile.skills[:30]))
    if profile.education:
        parts.append("Wykształcenie: " + ", ".join(profile.education[:4]))
    if profile.years_of_experience is not None:
        parts.append(f"Staż: {profile.years_of_experience:.1f} roku")
    if profile.experience_level:
        parts.append("Poziom: " + profile.experience_level)
    return "\n".join(parts)
