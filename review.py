"""Two questions asked of models: is this offer worth it, and is this CV good.

Step 4 scores how well one offer fits the candidate, before a CV is written.

Step 6 reviews a finished CV with two independent calls:

  - does it state anything the knowledge base does not support? The hard
    question: an invented fact may reject the document, whatever it scores.
  - how good an application is it? The soft one; its mark drives the loop.

The fact check tells an inference from an invention. "Built event-driven
services with Kafka" where the base lists Kafka among the technologies is
ordinary CV writing; an employer nobody worked for is a lie. Only the second
rejects.
"""

import config
import cv as cv_module
from models import CvReview, MatchAssessment
from providers import LlmOutputError


# Every mark in this file runs on one scale. The prompts spell it out; the
# range checks, the error messages and the label on the writer's remarks
# read it from here.
SCORE_RANGE = (0, 100)
SCORE_SCALE = f"{SCORE_RANGE[0]}-{SCORE_RANGE[1]}"


# --------------------------------------------------------------------------
# Step 4: is this offer worth writing for
# --------------------------------------------------------------------------

ASSESS_PROMPT = """Jesteś doświadczonym rekruterem. Oceniasz, na ile profil
kandydata pasuje do konkretnego ogłoszenia o pracę.

##Profil kandydata:
[PROFIL]
{summary}
[/PROFIL]

##Ogłoszenie:
[OFERTA]
{offer_text}
[/OFERTA]

##Oceń dopasowanie w skali 0-100, gdzie:
+0-40: kandydat nie spełnia wymagań kluczowych,
+41-69: spełnia część wymagań, ale brakuje czegoś istotnego,
+70-100: spełnia wymagania kluczowe i realnie może dostać tę pracę.

##Bierz pod uwagę: wymagane technologie, wymagany staż i poziom stanowiska.
##Braku wykształcenia nie traktuj jako dyskwalifikacji, jeżeli doświadczenie
##je nadrabia.

##Zwróć JSON:
{{"score": liczba całkowita 0-100, "rationale": "2-3 zdania uzasadnienia"}}
"""


def assess_offer(profile_summary_text, offer_key, offer_text, llm):
    """How well one offer fits. Returns a MatchAssessment.

    Raises LlmOutputError when the mark is not a number in range, like
    score_cv: a failed call must not read as a poor fit. The caller decides
    what an unassessable offer means for the run.
    """
    # A number written as text is still an answer; a missing score is not.
    answer = llm.ask_json("", ASSESS_PROMPT,
                          {"summary": profile_summary_text,
                           "offer_text": offer_text},
                          role="offer_matcher",
                          needs={"score": (int, float, str)})
    raw = answer.get("score")
    try:
        score = int(raw)
    except (TypeError, ValueError):
        raise LlmOutputError(
            f"offer_matcher: ocena dopasowania bez liczbowej wartości ({raw!r})")
    if not SCORE_RANGE[0] <= score <= SCORE_RANGE[1]:
        raise LlmOutputError(
            f"offer_matcher: ocena dopasowania poza skalą {SCORE_SCALE} "
            f"({score})")
    return MatchAssessment(
        jobOfferKey=offer_key,
        score=score,
        rationale=str(answer.get("rationale") or ""),
    )


# --------------------------------------------------------------------------
# Step 6a: does the CV say anything the knowledge base does not support
# --------------------------------------------------------------------------

FACTS_PROMPT = """Jesteś audytorem zgodności danych. Sprawdzasz, czy CV mówi
coś, czego nie ma w bazie wiedzy o kandydacie.

##Baza wiedzy - jedyne źródło prawdy:
[BAZA-WIEDZY]
{profile_text}
[/BAZA-WIEDZY]

##CV do sprawdzenia:
[CV]
{cv_text}
[/CV]

##Wypisz zdania z CV, które wykraczają poza bazę wiedzy, i przypisz każdemu
##jedną z dwóch etykiet:

+"WNIOSEK" - zdanie rozwija to, co w bazie wiedzy jest, ale nie zmyśla.
 Przykład: baza wymienia Kafkę wśród technologii roli, a CV pisze "budował
 rozwiązania oparte o Kafkę". To jest normalna redakcja CV. Oznacz, żeby
 dało się złagodzić, ale to NIE jest zmyślenie.

+"BRAK_POKRYCIA" - w bazie wiedzy nie ma niczego, z czego to wynika, albo
 zdanie jest z nią sprzeczne. Przykład: liczba, procent, nazwa firmy,
 technologia, certyfikat albo stanowisko, których w bazie po prostu nie ma.

##Czego NIE wypisywać w ogóle:
+zdań, które są przeformułowaniem czegoś, co w bazie wiedzy jest wprost,
+zdań ogólnych, które nie podają żadnego faktu ("komunikatywny zespołowo"),
+pól w nawiasach kwadratowych - to miejsca do uzupełnienia przez człowieka,
 a nie twierdzenia o kandydacie,
+klauzuli zgody na przetwarzanie danych osobowych (RODO) na końcu
 dokumentu - to cytat z przepisu, a nie zdanie o kandydacie. Jej zgodność
 z ustawą sprawdza osobny mechanizm,
+sekcji pustych,
+NAGŁÓWKÓW SEKCJI ani niczego, co dotyczy formy dokumentu: nazw sekcji
 ("KLUCZOWE UMIEJĘTNOŚCI", "DOŚWIADCZENIE ZAWODOWE"), kolejności, układu,
 pogrupowania umiejętności w kategorie, sposobu zapisu dat. CV wolno mieć
 inną strukturę niż baza wiedzy - to jest inny rodzaj dokumentu. Sprawdzasz
 FAKTY O KANDYDACIE, a nie to, czy dokument jest przepisany jeden do jednego.

##Oceniasz wyłącznie treść. Jeżeli twoje uzasadnienie brzmiałoby "baza wiedzy
##nie stosuje takiego podziału" albo "różnica w formacie" - nie wypisuj tego.

##Sekcję nazywaj tak, jak nazywa się w CV: doświadczenie, umiejętności,
##wykształcenie, certyfikaty, języki, podsumowanie, inne.

##Zwróć JSON:
{{"findings": [{{"sentence": "cytat z CV", "label": "WNIOSEK albo BRAK_POKRYCIA",
                "section": "nazwa sekcji", "why": "czego brakuje w bazie wiedzy"}}]}}

##Jeżeli wszystko ma pokrycie, zwróć pustą listę.
"""


# The reviewer sees the contact block and the consent clause, and neither is
# a claim about the candidate, so findings quoting them are dropped. Short
# enough that a fragment of the clause is still recognisable, long enough
# that an ordinary sentence cannot be one by accident.
SHORTEST_CLAUSE_FRAGMENT = 20


def about_the_frame(sentence):
    """Is this finding about the contact block or the consent clause?

    Both halves ask "IS the quote one of those", never "does it look like
    one": a filter that drops real findings is worse than none, because the
    fact check is all that stands between an invention and a sent CV. So no
    marker words ("rodo" is inside "srodowisku") and no "contains a bracket"
    ("Lokalizacja: [Miejscowosc], Polska" claims Polska).

    Containment is checked both ways - the quote may be one sentence of the
    clause or the clause plus its separator - through cv.clause_fragment,
    shared with cv.clause_is_intact so the two cannot disagree.
    """
    if cv_module.only_placeholders(sentence):
        return True

    flat = cv_module.clause_fragment(sentence)
    if len(flat) < SHORTEST_CLAUSE_FRAGMENT:
        return False
    for clause in (cv_module.GDPR_CLAUSE_PL, cv_module.GDPR_CLAUSE_EN):
        whole = cv_module.clause_fragment(clause)
        if flat in whole:
            return True
        # Bounded: room for the separator, not for a smuggled sentence.
        if whole in flat and len(flat) - len(whole) <= SHORTEST_CLAUSE_FRAGMENT:
            return True
    return False


def rejects_the_document(section):
    """Is a sentence in this section serious enough to sink the whole CV?

    Matched on stems, because the name is free text from a model:
    "edukacja", "WYKSZTAŁCENIE" and "EDUCATION" are one section.

    An unrecognised or missing name does NOT reject - the opposite of an
    unrecognised label. A label is one of two words we chose, so anything
    else is a malfunction; a section name is whatever the model wrote, and
    the everyday unknown one is a skill category ("narzędzia"), not a hidden
    employer. The finding is still reported and still blocks "ready".
    """
    section = (section or "").lower()
    return any(stem in section for stem in config.CRITICAL_SECTION_STEMS)


def check_facts(profile_text, cv_markdown, llm):
    """Returns (unsupported, inferred, hard_fail).

    The reviewer sees the document exactly as it will be sent, placeholders
    and clause included; findings about those are filtered out afterwards
    (about_the_frame) rather than the input trimmed.
    """
    # "findings": [] is a clean CV; no findings key is an audit that did
    # not happen. The two must never look alike.
    answer = llm.ask_json("", FACTS_PROMPT,
                          {"profile_text": profile_text,
                           "cv_text": cv_markdown},
                          role="cv_reviewer",
                          needs={"findings": list})

    unsupported = []
    inferred = []
    hard_fail = False
    for item in answer.get("findings") or []:
        if not isinstance(item, dict):
            continue
        sentence = str(item.get("sentence") or "").strip()
        if not sentence or about_the_frame(sentence):
            continue
        # "nieznana" is for the report; rejects_the_document says no to it.
        section = str(item.get("section") or "").strip().lower() or "nieznana"
        why = str(item.get("why") or "").strip()
        line = f"[{section}] {sentence}" + (f" - {why}" if why else "")

        # An unknown label is the serious one: an odd answer must not wave
        # a CV through.
        label = str(item.get("label") or "BRAK_POKRYCIA").strip().upper()
        if label == "WNIOSEK":
            inferred.append(line)
            continue

        unsupported.append(line)
        if rejects_the_document(section):
            hard_fail = True

    return unsupported, inferred, hard_fail


# --------------------------------------------------------------------------
# Step 6b: how good an application is it
# --------------------------------------------------------------------------

SCORE_PROMPT = """Jesteś doświadczonym rekruterem technicznym oceniającym
jakość CV przygotowanego pod konkretne ogłoszenie.

UWAGA: zgodność CV ze stanem faktycznym sprawdza osobny proces. NIE oceniasz
prawdziwości - oceniasz wyłącznie jakość dokumentu jako narzędzia aplikacyjnego
pod TO ogłoszenie. Pola w nawiasach kwadratowych to miejsca do uzupełnienia
przez kandydata; nie obniżaj za nie oceny. Tak samo klauzula zgody na
przetwarzanie danych na końcu dokumentu - jest wymagana i nie podlega
ocenie jakości.

##Ogłoszenie:
[OFERTA]
{offer_text}
[/OFERTA]

##CV:
[CV]
{cv_text}
[/CV]

##Oceń w pięciu kryteriach, każde w skali 0-100. Najpierw napisz 2-3 zdania
##analizy z konkretnymi przykładami z CV, dopiero potem przypisz ocenę.

##Opisy poniżej to KOTWICE, a nie jedyne dozwolone wartości: wpisz dowolną
##liczbę całkowitą od 0 do 100 i użyj kotwic, żeby wiedzieć, gdzie jesteś.

A. DOPASOWANIE DO OGŁOSZENIA
 0-25: CV generyczne.  26-50: adresuje mniejszość wymagań kluczowych.
 51-75: większość wymagań kluczowych.  76-100: wszystkie kluczowe wyeksponowane.
B. PRIORYTETYZACJA TREŚCI
 0-25: nieistotne dominują.  26-50: kolejność przypadkowa.
 51-75: istotne na początku sekcji.  76-100: najmocniejsze widać w 5 sekund.
C. KONKRETNOŚĆ OSIĄGNIĘĆ
 0-25: same ogólniki.  26-50: sporadyczne konkrety.
 51-75: większość pozycji z rezultatem lub skalą.
 76-100: działanie + skala + rezultat.
D. STRUKTURA I CZYTELNOŚĆ
 0-25: chaotyczna.  26-50: nierówne formatowanie.
 51-75: czytelna i spójna.  76-100: wzorowa spójność, adekwatna długość.
E. JĘZYK I PROFESJONALIZM
 0-25: błędy lub zły ton.  26-50: poprawnie, ale szablonowo.
 51-75: poprawnie i naturalnie.  76-100: bezbłędnie, ton dopasowany do branży.

##Zwróć JSON:
{{"criteria": {{"A": {{"analysis": "...", "score": 70}}, "B": {{...}},
               "C": {{...}}, "D": {{...}}, "E": {{...}}}},
  "top_fix": "jedna najważniejsza rzecz do poprawienia"}}
"""


def weighted_total(scores):
    """The single mark of a CV out of 100. The weights sum to 1.0."""
    total = sum(scores.get(name, 0) * weight
                for name, weight in config.CRITERION_WEIGHTS.items())
    return round(total, 3)


def score_cv(offer_text, cv_markdown, llm):
    """Returns (scores, analysis, total, top_fix).

    Raises LlmOutputError when a criterion has no number in range: a failed
    call must not read as a terrible CV. The agent's tool reports the
    failure to the agent.
    """
    answer = llm.ask_json("", SCORE_PROMPT,
                          {"offer_text": offer_text, "cv_text": cv_markdown},
                          role="cv_reviewer",
                          needs={"criteria": dict})

    criteria = answer.get("criteria") or {}
    scores = {}
    analysis = {}
    for name in config.CRITERION_WEIGHTS:
        entry = criteria.get(name) or {}
        mark = (entry or {}).get("score") if isinstance(entry, dict) else None
        try:
            mark = int(mark)
        except (TypeError, ValueError):
            raise LlmOutputError(
                f"cv_reviewer: kryterium {name} bez liczbowej oceny "
                f"({mark!r})")
        if not SCORE_RANGE[0] <= mark <= SCORE_RANGE[1]:
            raise LlmOutputError(
                f"cv_reviewer: ocena kryterium {name} poza skalą "
                f"{SCORE_SCALE} ({mark})")
        scores[name] = mark
        analysis[name] = str(entry.get("analysis") or "")

    return scores, analysis, weighted_total(scores), str(answer.get("top_fix") or "")


def review_cv(profile_text, offer_text, cv_markdown, llm, language="pl",
              progress=None):
    """Both halves of step 6: two model calls plus the checks made by code.

    progress is told before each of the two calls: together they are the
    longest silence in a run.
    """
    say = progress or (lambda message: None)
    say("checking the facts...")
    unsupported, inferred, hard_fail = check_facts(
        profile_text, cv_markdown, llm)
    say("marking the quality...")
    scores, analysis, total, top_fix = score_cv(offer_text, cv_markdown, llm)
    return CvReview(
        unsupported=unsupported,
        inferred=inferred,
        hard_fail=hard_fail,
        document_defects=cv_module.document_defects(cv_markdown, language),
        scores=scores,
        analysis=analysis,
        total=total,
        top_fix=top_fix,
    )


def passes_quality(review, threshold):
    """Is this version good enough to stop?

    Four conditions: not rejected by the fact check, no unsupported sentence
    at all, no document defect, and the mark at the threshold. "No
    unsupported sentence at all" is what lets the loop remove an invention
    the scorer would otherwise reward. Defects are found by code and
    repaired by the model on the next attempt, never silently.

    This does not reject a document: a version failing here is still a
    candidate for pick_best.
    """
    if review is None or review.hard_fail:
        return False
    if review.unsupported:
        return False
    if review.document_defects:
        return False
    return review.total is not None and review.total >= threshold


def feedback_text(review):
    """The remarks handed to the writer, verbatim from the review."""
    if review is None:
        return "Brak uwag."

    lines = []
    if review.scores:
        marks = " ".join(f"{name}={review.scores.get(name, 0)}" for name in "ABCDE")
        lines.append(f"Oceny (skala {SCORE_SCALE}): {marks}")
        if review.total is not None:
            lines.append(f"Wynik ważony: {review.total}/100")
        lines.append("")
        for name in "ABCDE":
            if review.analysis.get(name):
                lines.append(f"{name}: {review.analysis[name]}")

    if review.top_fix:
        lines.append("")
        lines.append("Najważniejsza poprawka: " + review.top_fix)

    if review.document_defects:
        lines.append("")
        lines.append("Usterki samego dokumentu - popraw je bezwzględnie. "
                     "Dopóki choć jedna zostaje, CV nie może zostać uznane "
                     "za gotowe:")
        for defect in review.document_defects:
            lines.append("- " + defect)

    if review.unsupported:
        lines.append("")
        lines.append("Zdania bez pokrycia w bazie wiedzy - USUŃ je. Dopóki "
                     "choć jedno zostaje, CV nie może zostać uznane za "
                     "gotowe, a w doświadczeniu, wykształceniu i "
                     "certyfikatach powoduje odrzucenie całego dokumentu:")
        for sentence in review.unsupported:
            lines.append("- " + sentence)

    if review.inferred:
        lines.append("")
        lines.append("Zdania, które wykraczają poza literę bazy wiedzy - "
                     "złagodź je do tego, co baza mówi wprost. Nie usuwaj "
                     "ich, to nie są zmyślenia:")
        for sentence in review.inferred:
            lines.append("- " + sentence)

    return "\n".join(lines).strip() or "Brak uwag."
