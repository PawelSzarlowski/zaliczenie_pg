"""The agent that owns one offer: write the CV, review it, fix it, end.

Steps 6 to 7 for a single offer. A LangChain tool-calling agent: @tool
functions bound with bind_tools, AIMessage.tool_calls in, ToolMessage back.
The executor loop is written out here rather than taken from
langchain.agents.create_agent, so the limits below are visible and testable.

THE DIVISION OF POWER

  The agent decides: what to ask for, whether another attempt is worth it,
  when it is done. Nothing is done or decided on its behalf. No CV without
  tool_write_cv, no sendable version without tool_review_cv, no ending without tool_finish_cv;
  a prose answer instead of a tool call ends the work. Each of those is
  written into the report and the transcript as what the agent did.

  The code keeps the budget, inside the tools: a tool is a wall, and a wall
  REFUSES - it never performs work and never ends the loop.

    - tool_write_cv refuses past max_cv_versions, and after
      MAX_VERSIONS_WITHOUT_PROGRESS revisions that improved nothing;
    - every refusal says what happened and what to call instead;
    - only max_agent_steps ends the loop without tool_finish_cv - budget, not
      judgement;
    - the version sent is chosen by pick_best, arithmetic over the reviewed
      versions. Never "the last one": marks are noisy and a later version
      can be worse.
"""

import time
import traceback

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

import config
import cv as cv_module
import review as review_module
from models import CvVersion, OfferOutcome

AGENT_SYSTEM = """
Jesteś redaktorem prowadzącym. Twoim zadaniem jest przygotować dobre CV
kandydata pod jedną konkretną ofertę, korzystając wyłącznie z narzędzi.

Jesteś JEDYNYM autorem tej pracy i nikt Cię nie zastąpi:
+Nikt nie napisze CV za Ciebie. Jeżeli nie wywołasz narzędzia tool_write_cv, kandydat nie
 dostanie żadnego CV dla tej oferty.
+Nikt nie oceni za Ciebie poszczególnych wersji CV. Wersja, dla której nie wywołałeś narzędzia tool_review_cv,
 NIE MOŻE zostać wysłana! Do wysyłki wybiera się wyłącznie spośród ocenionych wersji CV.
+Nikt nie zakończy pracy za Ciebie. Kończysz wyłącznie wywołaniem narzędzia tool_finish_cv, 
 z samodzielnie wybranym i uzasadnionym powodem.
+Każda Twoja odpowiedź MUSI być wywołaniem narzędzia z listy dostępnych. Odpowiedź tekstem, bez
 wywołania narzędzia, kończy pracę natychmiast, potencjalnie bez wygenerowania żadnego CV. 
 Taka sytuacja jest uważana za błąd i raportowana.

Jak pracujesz:
+Zacznij od narzędzia tool_write_cv, żeby powstała pierwsza wersja.
+Po KAŻDEJ wersji wywołaj narzędzie tool_review_cv i przeczytaj raport z tej oceny.
+Jeżeli raport mówi, że PRÓG JEST SPEŁNIONY - wywołaj narzędzie tool_finish_cv.
+Jeżeli nie, ponownie wywołaj narzędzie tool_write_cv . W argumencie "uwagi" napisz krótkie,
 konkretne polecenie: co poprawić w pierwszej kolejności i czego nie ruszać.
 Opieraj je na raporcie, a nie na własnych domysłach.
+Narzędzia tool_show_cv używaj tylko wtedy, gdy naprawdę musisz zobaczyć treść CV.
+Kiedy narzędzie tool_write_cv ODMAWIA - bo wyczerpał się limit wersji albo dwie kolejne
 poprawki niczego nie podniosły - nie ponawiaj wywołania. Wywołaj narzędzie tool_finish_cv
 i podaj powód. Kolejna wersja i tak nie powstanie.
+NIE kończ, dopóki ostatnia napisana wersja nie została oceniona. Wersja bez
 oceny to zmarnowane wywołanie i CV, którego nie da się wysłać.


Odpowiadaj wyłącznie wywołaniami narzędzi z listy dostępnych.
"""


def truth_rank(review):
    """0 rejected by the fact check, 1 not rejected but with an unsupported
    sentence, 2 without a single one. The order pick_best chooses by."""
    if review.hard_fail:
        return 0
    return 1 if review.unsupported else 2


def pick_best(versions):
    """Which version gets sent. Returns (iteration, cleared_the_hard_gate).

    Step 7. Arithmetic, no model, over the REVIEWED versions only. Among
    those that cleared the fact check, the ones without a single unsupported
    sentence go first, whatever their marks; only when there is none do the
    others compete. Within that pool the best mark wins; within
    config.SCORE_NOISE of it the marks are noise, so the version with fewer
    unsupported sentences wins, then fewer document defects, then the
    earlier one. If nothing cleared the gate, the best of the rejected is
    returned with False.

    (0, False) when no version was reviewed: nothing to send, nothing put to
    the gate.
    """
    reviewed = [version for version in versions if version.review is not None]
    if not reviewed:
        return 0, False

    # The best rank present decides who competes. An invented skill outside
    # the hard sections does not reject a version, but the reviewer rewards
    # it with a higher mark. 
    reached = max(truth_rank(version.review) for version in reviewed)
    pool = [version for version in reviewed
            if truth_rank(version.review) == reached]

    top = max((version.review.total or 0.0) for version in pool)
    contenders = [version for version in pool
                  if (version.review.total or 0.0) >= top - config.SCORE_NOISE]

    # Unsupported sentences before document defects: a mangled clause is a
    # paragraph a human can replace, an unsupported sentence is an untruth.
    best = min(contenders,
               key=lambda version: (len(version.review.unsupported),
                                    len(version.review.document_defects),
                                    version.iteration))
    return best.iteration, reached > 0


class OfferRun:
    """The state one agent works on. A plain class, never serialised."""

    def __init__(self, profile, offer, offer_text, llm, settings, language,
                 progress=None):
        self.profile = profile
        self.offer = offer
        self.offer_text = offer_text
        self.llm = llm
        self.settings = settings
        self.language = language
        # One line per model call: writing and reviewing take minutes,
        # and a screen that says nothing looks like a screen that hung.
        self.progress = progress or (lambda message: None)

        self.versions = []
        self.finished = False
        self.stop_reason = ""
        self.without_progress = 0

    # -- the work behind the tools -------------------------------------

    def write(self, notes):
        """Write the first draft, or the next version. Refuses past a wall.

        Two walls, both only refuse: the version limit, and revisions without
        progress. Neither ends the loop - the agent is told what to call
        instead and ends through tool_finish_cv.
        """
        limit = self.settings.max_cv_versions
        if len(self.versions) >= limit:
            self.progress(f"   tool_write_cv refused: {limit} versions "
                          f"already written")
            return (f"ODMOWA: limit {limit} wersji CV został wyczerpany. "
                    f"Kolejna nie powstanie. Wywołaj narzędzie tool_finish_cv i podaj powód.")

        if self.without_progress >= config.MAX_VERSIONS_WITHOUT_PROGRESS:
            self.progress(f"   tool_write_cv refused: "
                          f"{self.without_progress} revisions in a row "
                          f"improved nothing")
            return (f"ODMOWA: {self.without_progress} kolejne poprawki niczego "
                    f"nie podniosły, więc dalsze wersje nie powstaną. Wywołaj "
                    f"narzędzie tool_finish_cv i podaj powód.")

        # Tags the writer's transcript lines with the version being written.
        number = len(self.versions) + 1
        self.llm.context["version"] = number
        if not self.versions:
            self.progress(f"   v{number}/{limit}: writing the first "
                          f"version...")
            text, words = cv_module.write_cv(
                self.profile, self.offer_text, self.llm, self.language)
        else:
            source = self.last_reviewed()
            if source is None:
                return ("Bieżąca wersja nie została jeszcze oceniona. Wywołaj "
                        "najpierw narzędzie tool_review_cv - bez oceny nie ma czego poprawiać.")
            self.progress(f"   v{number}/{limit}: revising version "
                          f"{source.iteration}...")
            text, words = cv_module.revise_cv(
                self.profile, self.offer_text, source.markdown,
                review_module.feedback_text(source.review), notes, self.llm,
                self.language)

        version = CvVersion(
            iteration=len(self.versions) + 1,
            markdown=text,
            word_count=words,
            agent_notes=notes or "",
        )
        self.versions.append(version)
        self.progress(f"   v{number}: written, {words} words")
        return (f"Powstała wersja {version.iteration} ({words} słów). "
                f"Oceń ją narzędziem tool_review_cv.")

    def review(self):
        """Review the newest version, unless it already has one."""
        if not self.versions:
            return "Nie ma jeszcze żadnego CV. Wywołaj najpierw narzędzie tool_write_cv."

        version = self.versions[-1]
        if version.review is None:
            self.llm.context["version"] = version.iteration
            version.review = review_module.review_cv(
                self.profile.raw_text, self.offer_text, version.markdown,
                self.llm, self.language,
                progress=lambda message: self.progress(
                    f"   v{version.iteration}: {message}"))
            self.progress(self.marks_line(version))
            # Counted once per version: a repeated tool_review_cv costs not so much
            # and is not a revision that went nowhere.
            if self.made_progress(version):
                self.without_progress = 0
            else:
                self.without_progress += 1
        else:
            self.progress(f"   v{version.iteration}: reviewed already, "
                          f"the same report again")

        # Nothing is decided here: a good-enough version is reported as PRÓG
        # SPEŁNIONY and the agent ends through tool_finish_cv; no-progress is
        # counted here and enforced by tool_write_cv refusing.
        return self.report(version)

    def marks_line(self, version):
        """The marks of one version, as one line on the screen."""
        review = version.review
        marks = " ".join(f"{name}={review.scores.get(name, '?')}"
                         for name in "ABCDE")
        gate = ("REJECTED by the fact check" if review.hard_fail
                else "fact check passed")
        return (f"   v{version.iteration}: {review.total}/100 ({marks}), "
                f"{gate}, {len(review.unsupported)} unsupported, "
                f"{len(review.document_defects)} document defects")

    def made_progress(self, version):
        """Is this version better than everything reviewed before it?

        Better in the order pick_best chooses by, or the two would disagree
        on what "better" means. Reaching a rank no earlier version reached
        is progress - the first version the fact check does not reject, the
        first without a single unsupported sentence - because it changes
        what can be sent, even with the mark standing still. Within the best
        rank reached so far, a mark higher by more than config.SCORE_NOISE
        is progress. A higher mark in a lower rank is not: pick_best would
        never send that version. Fewer unsupported sentences alone is a
        tie-break for pick_best, not progress.
        """
        earlier = [one for one in self.versions[:-1] if one.review is not None]
        if not earlier:
            return True

        reached = max(truth_rank(one.review) for one in earlier)
        mine = truth_rank(version.review)
        if mine != reached:
            return mine > reached

        best = max((one.review.total or 0.0) for one in earlier
                   if truth_rank(one.review) == reached)
        return (version.review.total or 0.0) > best + config.SCORE_NOISE

    def last_reviewed(self):
        for version in reversed(self.versions):
            if version.review is not None:
                return version
        return None

    def report(self, version):
        """What the agent is told about one reviewed version."""
        review = version.review
        gate = "ODRZUCONE" if review.hard_fail else "przeszło"
        defects = ("\nUsterki dokumentu: %s." % len(review.document_defects)
                   if review.document_defects else "")
        met = ("SPEŁNIONY"
               if review_module.passes_quality(review,
                                               self.settings.cv_quality_threshold)
               else "NIESPEŁNIONY")
        return "\n".join([
            f"Wersja {version.iteration} z {self.settings.max_cv_versions}.",
            f"Kontrola faktów: {gate} "
            f"({len(review.unsupported)} zdań bez pokrycia).{defects}",
            "",
            review_module.feedback_text(review),
            "",
            f"PRÓG (wynik >= {self.settings.cv_quality_threshold}, brak "
            f"odrzucenia przez kontrolę faktów, ani jednego zdania bez "
            f"pokrycia i ani jednej usterki dokumentu): {met}",
        ] + self.what_next(met == "SPEŁNIONY"))

    def what_next(self, threshold_met):
        """What the agent should do now, told while it can still act on it.

        Advice, not enforcement: the walls are in tool_write_cv.
        """
        if threshold_met:
            return ["", "Próg jest spełniony. Wywołaj narzędzie tool_finish_cv."]
        left = config.MAX_VERSIONS_WITHOUT_PROGRESS - self.without_progress
        if left <= 0:
            return ["", (f"{self.without_progress} kolejne poprawki niczego "
                         f"nie podniosły. Narzędzie tool_write_cv odmówi kolejnej wersji - "
                         f"wywołaj narzędzie tool_finish_cv i podaj powód.")]
        if left == 1:
            return ["", ("UWAGA: ostatnia poprawka niczego nie podniosła. "
                         "Jeżeli następna też nie, narzędzie tool_write_cv odmówi kolejnej "
                         "wersji i zostanie Ci tylko narzędzie tool_finish_cv.")]
        return []


def build_tools(state):
    """The four tools, closed over one offer's state.

    Built per run so the CV need not travel as a tool argument.
    """

    @tool
    def tool_show_cv() -> str:
        """Zwraca pełną treść aktualnej wersji CV."""
        if not state.versions:
            return "Nie ma jeszcze żadnej wersji CV."
        return state.versions[-1].markdown

    @tool
    def tool_write_cv(uwagi: str = "") -> str:
        """Pisze CV. Przy pierwszym wywołaniu tworzy pierwszą wersję, przy
        kolejnych poprawia ostatnią ocenioną. W argumencie 'uwagi' podaj
        krótkie, konkretne polecenie: co poprawić i czego nie ruszać."""
        return state.write(uwagi)

    @tool
    def tool_review_cv() -> str:
        """Ocenia najnowszą wersję CV: sprawdza, czy nie mówi czegoś, czego nie
        ma w bazie wiedzy o kandydacie, i wystawia oceny w pięciu kryteriach.
        Zwraca raport z uwagami i informacją, czy CV spełnia próg."""
        return state.review()

    @tool
    def tool_finish_cv(powod: str) -> str:
        """Kończy pracę nad tym CV. Wywołaj, gdy CV spełnia próg albo gdy
        dalsze poprawki nie mają sensu. Podaj krótki powód."""
        state.finished = True
        state.stop_reason = "agent zakończył: " + (powod or "bez powodu")
        return "Zakończono."

    tools = [tool_show_cv, tool_write_cv, tool_review_cv, tool_finish_cv]
    return tools, {one.name: one for one in tools}


def message_text(message):
    """One message as text, tool calls included.

    A normal agent turn has empty content and one tool call; .content alone
    would record it as a blank line.
    """
    parts = [message.content or ""]
    for call in getattr(message, "tool_calls", None) or []:
        parts.append(f"[tool] {call['name']}({call['args']})")
    return "\n".join(part for part in parts if part)


def conversation_text(messages):
    """The whole conversation the agent was sent this turn, in full.

    Every turn re-sends everything - the offer, every answer and every tool
    result - so this is the largest prompt in the run. Not summarised: the
    transcript has to explain the answer that came back.
    """
    return "\n\n".join(f"[{type(one).__name__}] {message_text(one)}"
                           for one in messages)


def run_agent(state, chat_model):
    """The executor. Returns (steps taken, why it stopped)."""
    tools, by_name = build_tools(state)
    bound = chat_model.bind_tools(tools)

    opening = ("Oferta, pod którą powstaje CV:\n\n" + state.offer_text
               + "\n\nZacznij od napisania pierwszej wersji CV.")
    # This loop talks to the model without going through Llm.send, so
    # clean_prompt and clean_answer are applied here by hand.
    messages = [
        SystemMessage(AGENT_SYSTEM),
        HumanMessage(state.llm.cleaned("cv_agent", "prompt", opening)),
    ]

    steps = 0
    for steps in range(1, state.settings.max_agent_steps + 1):
        # An agent turn belongs to the offer, not to a version.
        state.llm.context.pop("version", None)
        started = time.time()
        answer = bound.invoke(messages)
        # This call bypasses Llm.send, so it is recorded and counted here.
        # Before the append: what is written is what was sent.
        state.llm.record_call("cv_agent", AGENT_SYSTEM,
                              conversation_text(messages[1:]),
                              message_text(answer), time.time() - started)
        messages.append(answer)
        state.llm.record_usage("cv_agent",
                               getattr(answer, "usage_metadata", None))

        if not answer.tool_calls:
            # Prose ends the work; no nudge. The prose is in the transcript
            # line above.
            return steps, "brak wywołania narzędzia"

        for call in answer.tool_calls:
            chosen = by_name.get(call["name"])
            if chosen is None:
                result = f"Nie ma narzędzia o nazwie {call['name']}."
            else:
                arguments = {
                    name: state.llm.cleaned("cv_agent", "answer", value)
                    if isinstance(value, str) else value
                    for name, value in (call["args"] or {}).items()}
                try:
                    result = chosen.invoke(arguments)
                except Exception as error:
                    # Back to the agent as an ordinary answer. A tool that
                    # fails is information it can act on; an exception here
                    # would cost the whole offer.
                    result = f"Narzędzie zawiodło: {type(error).__name__}: {error}"
            messages.append(
                ToolMessage(state.llm.cleaned("cv_agent", "prompt",
                                              str(result)),
                            tool_call_id=call["id"]))

        if state.finished:
            return steps, state.stop_reason

    return steps, "limit kroków agenta"


def prepare_cv_for_offer(profile, offer, offer_text, assessment, llm, settings,
                         language="pl", progress=None):
    """Everything for one offer. Always returns an OfferOutcome, never raises.

    Nothing is done on the agent's behalf: no draft, no review, no ending.
    What the agent did not do is written into agent_stop_reason.
    """
    state = OfferRun(profile, offer, offer_text, llm, settings, language,
                     progress)

    # Every transcript line from here to the end of the loop carries the
    # offer key; the tools add the version.
    llm.context = {"offer": offer.get("jobOfferKey", "")}
    try:
        steps, reason = run_agent(state, llm.chat_model_for_agent())
    except Exception as error:
        steps, reason = 0, f"błąd agenta: {type(error).__name__}: {error}"
        # The report gets the line above; the traceback says where. A crash
        # between calls is the one failure the transcript cannot show.
        traceback.print_exc()
    llm.context = {}

    final_iteration, cleared = pick_best(state.versions)
    # Rejected means the fact check looked and said no; with nothing to send
    # it never looked.
    rejected = bool(final_iteration) and not cleared

    if not state.versions:
        reason += " (agent nie napisał żadnej wersji CV)"
    elif not final_iteration:
        reason += (" (agent nie ocenił żadnej wersji, więc żadna nie mogła "
                   "zostać wybrana do wysłania)")
    elif state.versions[-1].review is None:
        reason += (f" (wersja {state.versions[-1].iteration} została napisana, "
                   f"ale nie oceniona - do wysłania wybrano spośród "
                   f"ocenionych)")

    return OfferOutcome(
        jobOfferKey=offer.get("jobOfferKey", ""),
        title=offer.get("title", ""),
        url=offer.get("url", ""),
        match_score=assessment.score if assessment else 0,
        match_rationale=assessment.rationale if assessment else "",
        language=language,
        versions=state.versions,
        final_iteration=final_iteration,
        final_rejected_by_fact_check=rejected,
        agent_steps=steps,
        agent_stop_reason=reason,
    )


def final_version(outcome):
    """The CvVersion pick_best chose, or None. No fallback to another one."""
    for version in outcome.versions:
        if version.iteration == outcome.final_iteration:
            return version
    return None
