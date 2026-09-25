"""The whole flow, in one function that both interfaces call.

cli.py and main.py are two doors into run(); neither holds any logic.
"""

import json
import os
import time
import traceback
import uuid
from datetime import datetime

import agent as agent_module
import config
import cv as cv_module
import offers as offers_module
from models import RunOutcome
from candidate import profile_summary
from providers import LlmOutputError
from review import assess_offer

def search_warnings(query, found, level, settings):
    """Things about the search a reader of the report has to know."""
    warnings = []

    if query.experience:
        at_level = any(one.get("experienceLevel") == query.experience
                       for one in found)
        if not at_level:
            warnings.append(
                f"Ani jedna z {len(found)} znalezionych ofert nie jest na "
                f"poziomie {query.experience}. Oceny dopasowania będą niskie "
                f"niezależnie od tego, co kandydat umie.")
    else:
        # No level means the search was never narrowed by seniority.
        warnings.append(
            f"Nie udało się ustalić stażu kandydata, więc {len(found)} ofert "
            f"znaleziono bez zawężania do jakiegokolwiek poziomu. Mogą wśród "
            f"nich być stanowiska wymagające wielu lat doświadczenia, a to "
            f"samo w sobie obniży oceny dopasowania. Sprawdź, czy baza wiedzy "
            f"podaje daty zatrudnienia albo liczbę lat.")

    # The model was shown the division's categories and picked none. It
    # usually says so in its rationale; the run cannot act on that, but the
    # reader can.
    if query.division != "Other" and not query.category:
        warnings.append(
            f"Model nie wskazał żadnej kategorii w dziale {query.division} - "
            f"to zwykle znaczy, że sam dział jest zły. Sprawdź uzasadnienie "
            f"w 'Czego szukano' i w razie czego powtórz przebieg.")

    # At the bottom rung only division and city are left. Too few offers
    # there says the division - one unchecked model call - is probably wrong.
    if level >= 3 and len(found) < settings.min_offers:
        warnings.append(
            f"Po zdjęciu wszystkich filtrów zostało tylko {len(found)} ofert "
            f"(oczekiwano {settings.min_offers}). Sprawdź, czy dział "
            f"{query.division} jest właściwy - wybiera go model i nikt go nie "
            f"kontroluje.")

    return warnings


def forced_offers(wanted, found):
    """The offers named on purpose, in the order named.

    Matched on a prefix, like a git hash: the report prints eight characters
    of a 36-character key. An ambiguous prefix stops the run rather than
    guessing - a guess would spend an agent loop on the wrong offer.
    """
    chosen = []
    unknown = []
    for prefix in wanted:
        prefix = str(prefix or "").strip()
        if not prefix:
            continue
        hits = [one for one in found
                if str(one.get("jobOfferKey", "")).startswith(prefix)]
        exact = [one for one in hits if one.get("jobOfferKey", "") == prefix]
        if not hits:
            unknown.append(prefix)
            continue
        if not exact and len(hits) > 1:
            raise ValueError(
                f"Klucz {prefix} pasuje do {len(hits)} ofert: "
                + ", ".join(short_key(one.get("jobOfferKey", ""))
                              for one in hits[:4])
                + " - podaj więcej znaków")
        one = exact[0] if exact else hits[0]
        # Naming one offer twice is not asking for two CVs for it.
        already = [other.get("jobOfferKey", "") for other in chosen]
        if one.get("jobOfferKey", "") not in already:
            chosen.append(one)
    return chosen, unknown


def finished(outcome, llm, started):
    """What every ending of a run writes down."""
    outcome.seconds["razem"] = round(time.time() - started, 1)
    # Deep copy: the per-role counters must not stay shared with the Llm.
    outcome.usage = {role: dict(counters)
                     for role, counters in llm.usage.items()}
    return outcome


def nothing(message):
    """The default progress reporter: says nothing. cli.py passes print."""


def run(profile, llm, settings, offers=None, offers_from="",
        divisions=None, progress=nothing):
    """Steps 2 to 7 for one candidate. Always returns a RunOutcome.

    No offers found, or every offer rejected, is a result, not an error.

    offers, given, replaces the search (steps 2 and 3); offers_from is only
    a label for the report. An EMPTY list is not None: the caller had offers
    to give and there were none, so the run stops.

    Everything raised by steps() is caught here: the run is marked crashed,
    the reason goes into the report, the screen and the exit code, and the
    partial outcome is still written out.
    """
    started = time.time()
    outcome = RunOutcome(settings=settings.model_dump(),
                         models=llm.models_in_use())

    try:
        steps(outcome, profile, llm, settings, offers, offers_from,
              divisions, progress)
    except Exception as error:
        outcome.crashed = True
        outcome.stopped_reason = (
            f"Przebieg przerwany błędem: {type(error).__name__}: {error}")
        progress("   FAILED: " + outcome.stopped_reason)
        # The line above says what; the traceback says where.
        traceback.print_exc()

    return finished(outcome, llm, started)


def steps(outcome, profile, llm, settings, offers, offers_from,
          divisions, progress):
    """Everything a run does, filling in `outcome` as it goes.

    Stopping early is a plain return; run() writes the closing lines.
    """
    divisions = divisions if divisions is not None else offers_module.load_divisions()

    # ---- steps 2 and 3: what to search for, and searching ------------
    stage = time.time()
    if offers is None:
        progress("searching for offers...")
        query, why = offers_module.build_search_query(profile, llm, divisions,
                                                      settings)
        found, level = offers_module.search_with_relaxation(
            query, min_offers=settings.min_offers)
        outcome.search_query = dict(query.model_dump(), rationales=why)
        outcome.relaxation_level = level
        progress(f"   division {query.division}, offers: {len(found)} "
                 f"(relaxation rung {level})")
    else:
        outcome.offers_from = offers_from or "podane przez wywołującego"
        # The flow indexes offers by key: keyless or duplicate keys would
        # silently collapse two offers into one.
        try:
            found = offers_module.with_keys(offers, "oferta")
        except ValueError as error:
            outcome.stopped_reason = str(error)
            return
        progress(f"offers given, not searched for: {len(found)}")

    outcome.offers_found = len(found)
    outcome.seconds["szukanie"] = round(time.time() - stage, 1)

    if not found:
        outcome.stopped_reason = ("Nie znaleziono żadnej oferty"
                                  if offers is None else
                                  "Nie podano ani jednej oferty")
        return

    if offers is None:
        # Search warnings mean nothing for offers somebody handed in.
        outcome.warnings = search_warnings(query, found, level, settings)
        if why.get("division_defaulted"):
            outcome.warnings.insert(0,
                "Model nie wskazał działu, więc szukano w dziale Other. "
                "Oceny dopasowania będą niskie. "
                "Zalecane powtórzenie przebiegu.")
        for warning in outcome.warnings:
            progress("   WARNING: " + warning)

    # ---- step 4: every offer found goes to the matching model --------
    stage = time.time()
    summary = profile_summary(profile)

    if settings.force_offers:
        # Only the named offers are assessed.
        try:
            shortlist, unknown = forced_offers(settings.force_offers,
                                               found)
        except ValueError as error:
            outcome.stopped_reason = str(error)
            outcome.seconds["dopasowanie"] = round(time.time() - stage, 1)
            return

        for prefix in unknown:
            # Forcing replaces the automatic choice (so a typo means no CV).
            warning = (f"Nie ma oferty o kluczu zaczynającym się od "
                       f"{prefix} - literówka albo oferta z innego przebiegu")
            outcome.warnings.append(warning)
            progress("   WARNING: " + warning)

        if not shortlist:
            outcome.stopped_reason = (
                "Żaden z podanych kluczy nie pasuje do ofert tego przebiegu. "
                "Nie ma żadnej oferty, dla której można by napisać CV. ")
            outcome.seconds["dopasowanie"] = round(time.time() - stage, 1)
            return

        progress(f"forcing {len(shortlist)} offers, thresholds not applied")
    elif settings.max_offers_for_cv == 0:
        # The pre-screen: search and report, no offer put to a model.
        outcome.stopped_reason = (
            "max_offers_for_cv = 0: żadna oferta nie była oceniana przez "
            "model i żadne CV nie powstało")
        outcome.seconds["dopasowanie"] = round(time.time() - stage, 1)
        return
    else:
        # Every offer found goes to the model, in the order the API sent them. 
        shortlist = found[:settings.max_offers_to_assess]
        if len(found) > len(shortlist):
            outcome.warnings.append(
                f"Znaleziono {len(found)} ofert, a modelowi pokazano pierwszych "
                f"{len(shortlist)} - wyszukiwanie było zbyt szerokie. "
                f"Zawęź innymi kryteriami, np. miastem lub stawką (albo "
                f"podnieś max_offers_to_assess)")

    progress(f"scoring the fit of {len(shortlist)} offers...")
    for number, offer in enumerate(shortlist, start=1):
        key = offer.get("jobOfferKey", "")
        text = offers_module.offer_text(offer)
        try:
            assessment = assess_offer(summary, key, text, llm)
        except LlmOutputError as error:
            # An answer that is not a mark leaves the offer unscored - not 0.
            warning = (f"Oferta {short_key(key)} ({offer.get('title', '')}) "
                       f"nie została oceniona: {error}")
            outcome.warnings.append(warning)
            progress(f"   {number}/{len(shortlist)}  brak oceny: {error}")
            continue
        outcome.assessed.append(assessment)
        # The offer itself, so a later run can work on it without searching
        # again - the search is live and may not return it.
        outcome.assessed_offers.append(offer)
        progress(f"   {number}/{len(shortlist)}  {assessment.score}/100  "
                 f"{offer.get('title', '')}")
    outcome.seconds["dopasowanie"] = round(time.time() - stage, 1)

    # ---- step 5: which offers actually get a CV ----------------------
    if settings.force_offers:
        # All of them, in the order named, no cap. 
        # The assessment reaches the report only. The writer is not informed 
        # about the result of the offer-candidate assessment.
        # It gets only the pure candidate data and the pure offer (for the 1st version).
        # For other versions it gets also extras: the generated cv, its review and agent notes.
        chosen = list(outcome.assessed)
    else:
        good = [item for item in outcome.assessed
                if item.score >= settings.match_score_threshold]
        good.sort(key=lambda item: item.score, reverse=True)
        chosen = good[:settings.max_offers_for_cv]

    if not chosen:
        outcome.stopped_reason = (
            f"Żadna oferta nie przekroczyła progu dopasowania "
            f"{settings.match_score_threshold}")
        return

    # ---- steps 6 and 7: loop of write-review, then the best version --
    stage = time.time()
    by_key = {one.get("jobOfferKey", ""): one
              for one in outcome.assessed_offers}
    for number, assessment in enumerate(chosen, start=1):
        offer = by_key[assessment.jobOfferKey]
        text = offers_module.offer_text(offer)
        progress(f"writing CV {number}/{len(chosen)}: "
                 f"{offer.get('title', '')} "
                 f"(up to {settings.max_cv_versions} versions)...")
        # Decided once, from the offer, so every version of one CV shares it.
        language = cv_module.detect_language(
            offers_module.clean_html(offer.get("description")) or text)
        one = agent_module.prepare_cv_for_offer(
            profile, offer, text, assessment, llm, settings, language,
            progress=progress)
        # A fact about the run, not about the writing: a poor match score
        # reads differently when the gate was not applied.
        one.forced = bool(settings.force_offers)
        outcome.outcomes.append(one)
        if not one.versions:
            progress(f"   no CV written: {one.agent_stop_reason}")
        elif not one.final_iteration:
            progress(f"   {len(one.versions)} versions, none reviewed, "
                     f"nothing to send: {one.agent_stop_reason}")
        else:
            progress(f"   {len(one.versions)} versions, sending "
                     f"{one.final_iteration}: {one.agent_stop_reason}")
    outcome.seconds["cv"] = round(time.time() - stage, 1)


# --------------------------------------------------------------------------
# Writing what a person actually reads
# --------------------------------------------------------------------------

def short_key(key):
    """The first eight characters of a 36-character key.

    The same eight the CV file names use, which ties a report row to a file.
    Unique enough: at 500 offers a collision is about one in 17 000.
    """
    return (key or "")[:8] or "?"


def version_notes(one, version):
    """One version with its review, as one readable file. --dump-run only."""
    lines = [
        "[//]: # (Oferta: %s)" % one.title,
        "[//]: # (Wersja %s z %s)" % (version.iteration, len(one.versions)),
        "[//]: # (Prompt i odpowiedź: %s, linie z offer=%s i version=%s)"
        % (config.TRANSCRIPT_FILE, one.jobOfferKey, version.iteration),
        "",
        version.markdown,
        "",
        "---",
        "",
        "## Recenzja wersji %s" % version.iteration,
        "",
    ]

    if version.agent_notes:
        lines.append("**O co prosił agent przed napisaniem tej wersji:** "
                     + version.agent_notes)
        lines.append("")

    review = version.review
    if review is None:
        lines.append("Ta wersja nie została oceniona.")
        return "\n".join(lines) + "\n"

    marks = " ".join("%s=%s" % (name, review.scores.get(name, 0))
                     for name in "ABCDE")
    lines.append("Oceny (0-100): %s, wynik ważony **%s/100**."
                 % (marks, review.total))
    lines.append("")
    for name in "ABCDE":
        if review.analysis.get(name):
            lines.append("- **%s**: %s" % (name, review.analysis[name]))
    lines.append("")
    if review.top_fix:
        lines.append("**Najważniejsza poprawka:** " + review.top_fix)
        lines.append("")
    if review.unsupported:
        lines.append("**Zdania bez pokrycia w bazie wiedzy%s:**"
                     % (" - ODRZUCAJĄ DOKUMENT" if review.hard_fail else ""))
        for sentence in review.unsupported:
            lines.append("- " + sentence)
        lines.append("")
    if review.inferred:
        lines.append("**Zdania wykraczające poza literę bazy wiedzy:**")
        for sentence in review.inferred:
            lines.append("- " + sentence)
        lines.append("")

    return "\n".join(lines) + "\n"


def write_dump(outcome, out_dir):
    """Everything a run produced, for looking at afterwards.

    run.json is the whole RunOutcome as POST /przygotuj returns it; the
    cv-*-v*.md files are the same, readable without a JSON viewer.
    """
    written = []

    path = os.path.join(out_dir, "run.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(outcome.model_dump(), handle, ensure_ascii=False, indent=2)
    written.append(path)

    for one in outcome.outcomes:
        for version in one.versions:
            name = "cv-%s-v%s.md" % (one.jobOfferKey[:8] or "oferta",
                                     version.iteration)
            path = os.path.join(out_dir, name)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(version_notes(one, version))
            written.append(path)

    return written


def default_out_dir():
    """A directory of its own for every run.

    Timestamp plus eight random hex characters: concurrent HTTP requests can
    finish in the same second, and four characters collide within fifty
    draws.
    """
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    return os.path.join(config.OUTPUT_DIR, f"{stamp}-{uuid.uuid4().hex[:8]}")


def write_output(outcome, out_dir=None, dump=False):
    """Write the CVs and the report. Returns the list of paths written.

    With dump, every version and its review are written as well - see
    write_dump.
    """
    out_dir = out_dir or default_out_dir()
    os.makedirs(out_dir, exist_ok=True)
    written = []

    for one in outcome.outcomes:
        version = agent_module.final_version(one)
        if version is None:
            continue
        name = "cv-%s.md" % (one.jobOfferKey[:8] or "oferta")
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8") as handle:
            # Markdown comments: invisible once rendered.
            handle.write("[//]: # (Oferta: %s)\n" % one.title)
            handle.write("[//]: # (Ogłoszenie: %s)\n" % one.url)
            handle.write("[//]: # (Wersja %s z %s, dopasowanie %s/100)\n\n"
                         % (one.final_iteration, len(one.versions),
                            one.match_score))
            # Exactly the bytes the model returned: nothing is added and
            # nothing is trimmed, so rubbish in an answer stays visible.
            handle.write(version.markdown)
        written.append(path)

    path = os.path.join(out_dir, "raport.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(report_text(outcome))
    written.append(path)

    if dump:
        written.extend(write_dump(outcome, out_dir))
    return written


def report_text(outcome):
    """The report, in Polish, for the one person who will read it.

    It exists so that somebody holding a CV can check why they got that one:
    which offers were found, how each scored, what was rejected and where, and
    how many attempts the final document took.
    """
    lines = ["# Raport z przygotowania CV", ""]

    if outcome.offers_from:
        # No search happened, so the search section would mislead.
        lines.append("## Skąd wzięły się oferty")
        lines.append("")
        lines.append(f"Nie szukano w serwisie. Oferty pochodzą z: "
                     f"**{outcome.offers_from}**.")
        lines.append("")
        lines.append(f"Ofert wczytanych: **{outcome.offers_found}**.")
        lines.append("")
    else:
        query = outcome.search_query or {}
        lines.append("## Czego szukano")
        lines.append("")
        lines.append(f"- dział: **{query.get('division', '?')}**")
        for field, label in (("location", "miasto"), ("category", "kategoria"),
                             ("subCategories", "technologie"),
                             ("experience", "poziom"),
                             ("minimumSalary", "minimalna stawka")):
            value = query.get(field)
            if isinstance(value, list):
                value = ", ".join(value)
            if value:
                lines.append(f"- {label}: {value}")
        rationales = query.get("rationales") or {}
        for key, value in rationales.items():
            # Only the sentences; the dict also carries a boolean flag.
            if value and isinstance(value, str):
                lines.append(f"- dlaczego {key}: {value}")
        lines.append("")
        lines.append(f"Znaleziono **{outcome.offers_found}** ofert "
                     f"na szczeblu rozluźniania **{outcome.relaxation_level}** "
                     f"(0 = pełne zapytanie, 3 = tylko dział i miasto).")
        lines.append("")

    if outcome.warnings:
        lines.append("### Na co zwrócić uwagę")
        lines.append("")
        for warning in outcome.warnings:
            lines.append("- **" + warning + "**")
        lines.append("")

    if outcome.assessed:
        lines.append("## Ocena dopasowania")
        lines.append("")
        by_key = {one.get("jobOfferKey", ""): one
                  for one in outcome.assessed_offers}
        for item in sorted(outcome.assessed, key=lambda one: -one.score):
            title = by_key.get(item.jobOfferKey, {}).get("title", "")
            lines.append(f"**{item.score}/100 — {title}**")
            lines.append("")
            lines.append(f"- klucz: `{short_key(item.jobOfferKey)}`")
            lines.append("")
            lines.append(item.rationale)
            lines.append("")

    if outcome.stopped_reason:
        # A crashed run may still have produced a CV, so the heading differs.
        lines.append("## Przebieg przerwany błędem" if outcome.crashed
                     else "## Dlaczego nie powstało CV")
        lines.append("")
        lines.append(outcome.stopped_reason)
        if outcome.crashed:
            lines.append("")
            lines.append("Poniżej jest tylko to, co przebieg zdążył zrobić "
                         "przed błędem. Nie jest to komplet.")
        lines.append("")

    for one in outcome.outcomes:
        lines.append(f"## CV: {one.title}")
        lines.append("")
        lines.append(f"- ogłoszenie: {one.url}")
        lines.append(f"- dopasowanie: **{one.match_score}/100**")
        if one.forced:
            lines.append("- **CV powstało na żądanie: próg dopasowania nie "
                         "był stosowany do tej oferty.**")
        if not one.versions:
            # No version, no verdict, no table - only the reason, made to
            # stand out under a heading that says "CV".
            lines.append("- **NIE POWSTAŁO ŻADNE CV. Agent zakończył pracę, "
                         "nie prosząc o napisanie ani jednej wersji, a kod "
                         "nie pisze CV za niego.**")
            lines.append(f"- agent: {one.agent_steps} kroków, "
                         f"zatrzymanie: {one.agent_stop_reason}")
            lines.append("")
            continue
        if not one.final_iteration:
            # Written, never reviewed, never a candidate. The table below
            # shows them with "—" for the review.
            lines.append(f"- wersji napisanych: **{len(one.versions)}**, "
                         f"**ŻADNA NIE ZOSTAŁA WYSŁANA**: agent nie poprosił "
                         f"o ocenę żadnej z nich, a do wysłania wybiera się "
                         f"wyłącznie spośród ocenionych. Kod nie ocenia CV "
                         f"za agenta.")
        else:
            lines.append(f"- wersji napisanych: **{len(one.versions)}**, "
                         f"wysłana jest wersja **{one.final_iteration}**")
        if one.final_rejected_by_fact_check:
            lines.append("- **UWAGA: kontrola faktów odrzuciła każdą "
                         "wersję. Przeczytaj CV zanim je wyślesz.**")
        lines.append(f"- agent: {one.agent_steps} kroków, "
                     f"zatrzymanie: {one.agent_stop_reason}")
        lines.append("")
        lines.append("| wersja | ocena | kontrola faktów | słów |")
        lines.append("|---|---|---|---|")
        for version in one.versions:
            review = version.review
            mark = review.total if review else "—"
            gate = "—"
            if review:
                parts = []
                if review.unsupported:
                    parts.append(f"{len(review.unsupported)} bez pokrycia")
                if review.inferred:
                    parts.append(f"{len(review.inferred)} wniosków")
                gate = ", ".join(parts) or "czysto"
                if review.hard_fail:
                    gate += " (odrzucone)"
            star = " ←" if version.iteration == one.final_iteration else ""
            lines.append(f"| {version.iteration}{star} | {mark} | {gate} "
                         f"| {version.word_count} |")
        lines.append("")

        version = agent_module.final_version(one)
        review = version.review if version else None
        if review and review.document_defects:
            lines.append("**UWAGA - usterki samego dokumentu.** Pętla ich "
                         "nie zdążyła poprawić, więc popraw je ręcznie "
                         "przed wysłaniem:")
            lines.append("")
            for defect in review.document_defects:
                lines.append("- " + defect)
            lines.append("")
        if review and review.unsupported:
            # Names the same rejecting sections as feedback_text does.
            lines.append("**Zdania bez pokrycia w bazie wiedzy.** "
                         "W doświadczeniu, wykształceniu i certyfikatach "
                         "powodują odrzucenie dokumentu; w pozostałych "
                         "sekcjach nie odrzucają go, ale nie pozwalają "
                         "uznać CV za gotowe. Tak czy inaczej sprawdź je "
                         "przed wysłaniem:")
            lines.append("")
            for sentence in review.unsupported:
                lines.append("- " + sentence)
            lines.append("")
        if review and review.inferred:
            lines.append("Zdania wykraczające poza literę bazy wiedzy. To nie "
                         "są zmyślenia, ale warto na nie spojrzeć:")
            lines.append("")
            for sentence in review.inferred:
                lines.append("- " + sentence)
            lines.append("")

    lines.append("## Modele i zużycie")
    lines.append("")
    if outcome.models:
        lines.append("| rola | model | dostawca | wywołań | tokeny wej. "
                     "| tokeny wyj. |")
        lines.append("|---|---|---|---|---|---|")
        for role, what in outcome.models.items():
            used = (outcome.usage or {}).get(role) or {}
            calls = used.get("calls", 0)
            if not calls:
                went_in = came_out = "—"
            elif not used.get("reported"):
                went_in = came_out = "nie podano"
            else:
                went_in = str(used.get("input_tokens", 0))
                came_out = str(used.get("output_tokens", 0))
                if used["reported"] < calls:
                    partial = f" (z {used['reported']}/{calls} wywołań)"
                    went_in += partial
                    came_out += partial
            lines.append(f"| {role} | {what.get('model_id', '?')} "
                         f"| {what.get('provider', '?')} | {calls} "
                         f"| {went_in} | {came_out} |")

        cleaned = [(role, (outcome.usage or {}).get(role, {})
                    .get("cleaned_chars", 0)) for role in outcome.models]
        cleaned = [(role, count) for role, count in cleaned if count]
        if cleaned:
            lines.append("")
            lines.append("Katalog modeli każe czyścić śmieci dla tych "
                         "ról (clean_prompt / clean_answer): "
                         + ", ".join(f"{role} - usunięto {count} znaków"
                                     for role, count in cleaned)
                         + ". Transkrypt zapisuje odpowiedź sprzed "
                           "czyszczenia.")
        lines.append("")
        lines.append("Wywołania liczymy sami, więc ta kolumna jest dokładna. "
                     "Tokeny pokazujemy tylko tam, gdzie podał je dostawca - "
                     "nie każdy to robi, a wpisane zero czytałoby się jak "
                     "pomiar.")
        lines.append("")
        lines.append("Wiersz `cv_agent` to same tury agenta. Praca, którą "
                     "zlecił swoimi narzędziami, liczy się w wierszach "
                     "`cv_writer` i `cv_reviewer` - to nie jest ta sama rzecz "
                     "policzona dwa razy.")
        lines.append("")

    lines.append("## Ustawienia, z jakimi uruchomiono przebieg")
    lines.append("")
    for name, value in (outcome.settings or {}).items():
        lines.append(f"- `{name}` = {value}")
    lines.append("")
    if outcome.seconds:
        lines.append("Czasy: " + ", ".join(
            f"{name} {value} s" for name, value in outcome.seconds.items()))
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"Ocena jakości CV nie jest powtarzalna co do punktu - ten sam "
                 f"dokument oceniony dwa razy dostaje różne wyniki. Różnicy "
                 f"mniejszej niż {config.SCORE_NOISE} nie traktuj jako sygnału.")
    return "\n".join(lines) + "\n"
