"""Asking a model something and getting an answer back.

The middle layer: turns a ROLE into a model and a provider (config.py says
which), and adds what every call needs - retries, and an answer that parses.
The one rule that is not a preference: the model that writes a CV may not be
the model that reviews it, because a model grading its own work grades it
kindly and that grade drives the improvement loop. Pre-flight enforces it.
"""

import json
import os
import re
import time

from langchain_core.prompts import ChatPromptTemplate

import config
import providers
# Raised in providers.py, because a provider has to be able to raise it and
# imports only go one way. Passed on here so that the rest of the application
# can go on saying "from llm import LlmOutputError".
from providers import LlmOutputError


def extract_json_object(raw: str) -> str:
    """The outermost {...} of the answer, whatever the model wrapped it in."""
    if raw is None:
        raise LlmOutputError("the model returned nothing")

    text = raw.strip()
    text = re.sub(r"^```[a-zA-Z]*", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise LlmOutputError(f"no JSON object in the answer: {raw[:200]!r}")
    return text[start:end + 1]


# Whitespace no document of ours needs: the non-breaking and typographic
# spaces, and the zero-width characters. Ordinary spaces, tabs and newlines
# are deliberately absent - two spaces at the end of a Markdown line are a
# line break, not rubbish. Written as escapes: the characters
# themselves are invisible in an editor.
PADDING_SPACES = ("\u00a0\u2000\u2001\u2002\u2003\u2004\u2005\u2006"
                  "\u2007\u2008\u2009\u200a\u202f\u205f\u3000")
ZERO_WIDTH = "\u200b\u200c\u200d\ufeff"

_PADDING_RUN = re.compile("[" + PADDING_SPACES + "]+")
_ZERO_WIDTH = re.compile("[" + ZERO_WIDTH + "]")


def without_padding(text):
    """The text with the padding characters taken out. Three rules, no more.

    A run of those spaces becomes one ordinary space, zero-width characters
    go, and whitespace at the very end of the text goes. Everything else is
    left exactly as it arrived - the Markdown is not reflowed and nothing
    inside the text is collapsed.
    """
    if not text:
        return text
    clean = _PADDING_RUN.sub(" ", text)
    clean = _ZERO_WIDTH.sub("", clean)
    return clean.rstrip()


def resolve_roles(overrides=None):
    """config.ROLE_MODELS with overrides applied, every name checked.

    Overrides can come from an HTTP request, so an unknown name is refused
    here rather than twenty minutes into a run.
    """
    chosen = dict(config.ROLE_MODELS)

    for role, model in (overrides or {}).items():
        if role not in config.ROLES:
            raise LlmOutputError(
                f"There is no role called '{role}'. The roles are: "
                f"{', '.join(config.ROLES)}."
            )
        if model not in config.MODELS:
            raise LlmOutputError(
                f"There is no model called '{model}'. config.MODELS knows: "
                f"{', '.join(sorted(config.MODELS))}."
            )
        chosen[role] = model

    for role in config.ROLES:
        if chosen.get(role) not in config.MODELS:
            raise LlmOutputError(
                f"The '{role}' role is set to {chosen.get(role)!r}, which is "
                f"not in config.MODELS. Check ROLE_MODELS, or the ROLE_ "
                f"variable for it in .env."
            )

    return chosen


def worth_retrying(error):
    """Could a second attempt survive this, or will it fail the same way?

    Retried: a dropped socket, a timeout, 408, 429, a mangled answer - all
    about the moment. Not retried: any other 4xx - a wrong model name, a bad
    key, a refused body fail identically until a person changes something.

    Only providers of the "openai" kind attach a status; the "ollama" kind
    goes through LangChain, whose errors carry none, so they are always
    retried. No status is read out of a message by matching text.
    """
    status = getattr(getattr(error, "response", None), "status_code", None)
    if status is None:
        return True
    if status in (408, 429):
        return True
    return not 400 <= status < 500


class Llm:
    """One object per run. Holds no connection - only who plays which role."""

    def __init__(self, role_models=None, transcript_path=None):
        self.role_models = resolve_roles(role_models)
        self.retries = 0
        # Every prompt and answer goes here, or nowhere when None.
        self.transcript_path = transcript_path
        # Role -> calls and tokens, as far as the providers reported them.
        self.usage = {}
        # Tags written into every transcript line: the offer key while an
        # offer is being worked on, and the version a writer or reviewer
        # call is about. Set by agent.py; they are what matches a line of
        # the transcript to a cv-*-v*.md file of --dump-run.
        self.context = {}
        # Built on first use and kept.
        self._providers = {}

    # ------------------------------------------------------------------
    # From a role to a model to a provider
    # ------------------------------------------------------------------

    def model(self, role):
        """Everything config.py declares about the model playing one role."""
        return config.MODELS[self.role_models[role]]

    def provider(self, role):
        """The provider that role's model lives behind."""
        name = self.model(role)["provider"]
        if name not in self._providers:
            self._providers[name] = providers.build(name)
        return self._providers[name]

    def models_in_use(self):
        """Role -> what is behind it. Printed in the report and by GET /modele."""
        return {
            role: {
                "model": name,
                "model_id": config.MODELS[name]["model_id"],
                "provider": config.MODELS[name]["provider"],
            }
            for role, name in self.role_models.items()
        }

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    @staticmethod
    def render(system_template, user_template, values):
        """Fill a prompt in and give back the two finished texts.

        Through ChatPromptTemplate: a literal brace in a prompt is doubled,
        and values are not re-parsed.
        """
        messages = ChatPromptTemplate.from_messages(
            [("system", system_template), ("user", user_template)]
        ).format_messages(**values)
        return messages[0].content, messages[1].content

    def send(self, role, system_text, user_text, as_json):
        """One call, to whichever provider this role's model lives behind.

        Tokens are counted here; callers get only the text. The catalogue
        entry's "extra_body" (how a model is pinned to one machine behind a
        broker) is handed down from here.
        """
        model = self.model(role)
        # Only when the entry asks for it; the transcript then holds the
        # prompt as it really went out, and the answer as it really came.
        before = len(system_text or "") + len(user_text or "")
        system_text = self.cleaned(role, "prompt", system_text)
        user_text = self.cleaned(role, "prompt", user_text)
        cut_from_prompt = before - len(system_text or "") - len(user_text or "")

        started = time.time()
        try:
            text, usage = self.provider(role).send(
                model["model_id"], system_text, user_text, as_json,
                extra=model.get("extra_body"))
        except Exception as error:
            # Failures are recorded too. A bare raise keeps the HTTP status
            # on the original for worth_retrying.
            self.record_call(role, system_text, user_text, None,
                             time.time() - started, error=error,
                             removed={"prompt": cut_from_prompt})
            raise

        clean = self.cleaned(role, "answer", text)
        self.record_call(role, system_text, user_text, text,
                         time.time() - started,
                         removed={"prompt": cut_from_prompt,
                                  "answer": len(text or "") - len(clean or "")})
        self.record_usage(role, usage)
        return clean

    def record_call(self, role, system_text, user_text, answer, seconds,
                    error=None, removed=None):
        """One line of the transcript. Does nothing when no path was given.

        JSON Lines, one line per ATTEMPT: a retried call appears once per
        try, so the time it really cost is visible. The answer is recorded
        raw, before any parsing, so a rejected answer is still readable.
        Opened and closed per line: a killed run cannot lose lines to a
        buffer. No key can reach this file - send() never sees one.
        """
        if not self.transcript_path:
            return

        model = self.model(role)
        line = {
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "role": role,
            "offer": self.context.get("offer"),
            "version": self.context.get("version"),
            # The catalogue name, not only the id: several entries share an
            # id and differ in the machine, which decides the speed.
            "model": self.role_models[role],
            "model_id": model["model_id"],
            "provider": model["provider"],
            "seconds": round(seconds, 1),
            "system": system_text,
            "user": user_text,
            "answer": answer,
            "error": f"{type(error).__name__}: {error}" if error else None,
        }
        # Only when something was actually taken out, so a line looks the
        # same as it always did while no entry asks for cleaning.
        for name, count in (removed or {}).items():
            if count:
                line["removed_%s_chars" % name] = count

        folder = os.path.dirname(self.transcript_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(self.transcript_path, "a", encoding="utf-8") as handle:
            print(json.dumps(line, ensure_ascii=False), file=handle)

    def usage_for(self, role):
        """This role's row of the usage table, created on first use."""
        return self.usage.setdefault(role, {"calls": 0, "reported": 0,
                                            "input_tokens": 0,
                                            "output_tokens": 0})

    def cleans(self, role, what):
        """Does this role's entry ask for cleaning? what is prompt or answer."""
        return bool(self.model(role).get("clean_" + what))

    def cleaned(self, role, what, text):
        """The text as the catalogue entry wants it, with the cut counted.

        Nothing happens unless the entry says so - without the flag the text
        comes back exactly as it was. What was cut goes into usage, so the
        report can say it even when no transcript was written.
        """
        if not text or not self.cleans(role, what):
            return text
        clean = without_padding(text)
        removed = len(text) - len(clean)
        if removed:
            seen = self.usage_for(role)
            seen["cleaned_chars"] = seen.get("cleaned_chars", 0) + removed
        return clean

    def record_usage(self, role, usage):
        """Count one call, and its tokens when the provider reported any.

        Called from send and from the agent loop, which bypasses send.
        "calls" is exact; "reported" says how many calls carried token
        counts, so a partial sum can be shown as partial.
        """
        seen = self.usage_for(role)
        seen["calls"] += 1

        tokens = providers.tokens_from(usage)
        if tokens:
            seen["reported"] += 1
            seen["input_tokens"] += tokens["input_tokens"]
            seen["output_tokens"] += tokens["output_tokens"]

    def call_with_retries(self, action, description):
        """Try, sleep each delay in turn, then give up loudly.

        One attempt more than there are delays, so every delay is slept on.
        Every retry is counted. Failures that cannot survive a second attempt
        are not retried - see worth_retrying.
        """
        last = None
        for delay in list(config.RETRY_DELAYS) + [None]:
            try:
                return action()
            except Exception as error:
                last = error
                if delay is None or not worth_retrying(error):
                    break
                self.retries += 1
                print(f"retrying {description} after {delay} s: {error}")
                time.sleep(delay)
        raise last

    # ------------------------------------------------------------------
    # The three ways the rest of the code asks for something
    # ------------------------------------------------------------------

    def ask_json(self, system_template, user_template, values, role,
                 needs=None):
        """A prompt with placeholders, an answer that must parse as JSON.

        An empty object is a mangled answer and is retried: "{}" from the
        fact check would otherwise read as "the CV invents nothing".

        `needs` names the keys an answer is worthless without, with the type
        each must have: needs={"findings": list}. Checked inside the retry
        so a half-answer is asked again.
        """
        system_text, user_text = self.render(system_template, user_template, values)

        def attempt():
            raw = self.send(role, system_text, user_text, as_json=True)
            answer = json.loads(extract_json_object(raw))
            if not isinstance(answer, dict) or not answer:
                raise LlmOutputError(
                    f"{role}: the model answered with nothing at all")
            for key, kind in (needs or {}).items():
                if not isinstance(answer.get(key), kind):
                    raise LlmOutputError(
                        f"{role}: the answer carries no usable '{key}': "
                        f"{str(answer)[:200]}")
            return answer

        return self.call_with_retries(attempt, f"{role} (json)")

    def ask_text(self, system_template, user_template, values, role):
        """An answer that is prose - the CV, which is Markdown."""
        system_text, user_text = self.render(system_template, user_template, values)

        def attempt():
            return self.send(role, system_text, user_text, as_json=False)

        return self.call_with_retries(attempt, f"{role} (text)")

    # ------------------------------------------------------------------
    # The agent's model
    # ------------------------------------------------------------------

    def chat_model_for_agent(self, role="cv_agent"):
        """The model object the agent binds its tools to.

        Refused here, before the run, when the catalogue entry has no
        "supports_tools".
        """
        model = self.model(role)
        if not model.get("supports_tools"):
            raise LlmOutputError(
                f"The '{role}' role runs a tool-calling agent, but "
                f"{self.role_models[role]} is declared in config.MODELS "
                f"without tool support."
            )
        return self.provider(role).chat_model(model["model_id"])

    # ------------------------------------------------------------------
    # Pre-flight
    # ------------------------------------------------------------------

    def preflight_configuration(self):
        """The half of pre-flight with no model call.

        Two comparisons of the catalogue, and the agent's provider asked to
        confirm tool support when it can answer that (capabilities()).
        POST /przygotuj runs this on every request. Raises LlmOutputError
        with a readable message.
        """
        writer = self.role_models["cv_writer"]
        reviewer = self.role_models["cv_reviewer"]
        family = config.MODELS[writer]["family"]
        if family == config.MODELS[reviewer]["family"]:
            raise LlmOutputError(
                f"The CV writer ({writer}) and the reviewer ({reviewer}) are "
                f"both from the '{family}' family. A model grading its own "
                f"work grades it kindly, and that grade drives the whole "
                f"improvement loop. Point cv_writer and cv_reviewer at two "
                f"different families."
            )

        agent = self.role_models["cv_agent"]
        if not config.MODELS[agent].get("supports_tools"):
            raise LlmOutputError(
                f"The agent runs on {agent}, which config.MODELS declares "
                f"without tool support, so the improvement loop has no way to "
                f"work. Point cv_agent at a model that has it."
            )

        # The catalogue is written by hand; a provider that can confirm it
        # is asked to. Most cannot and answer [].
        model_id = config.MODELS[agent]["model_id"]
        capabilities = self.provider("cv_agent").capabilities(model_id)
        if capabilities and "tools" not in capabilities:
            raise LlmOutputError(
                f"The agent model {model_id} does not support tool calling "
                f"(its provider reports: {', '.join(capabilities)}), so the "
                f"improvement loop has no way to work. Pick a model with the "
                f"'tools' capability, and correct config.MODELS."
            )

    def preflight(self):
        """The configuration checks plus one live call to writer and reviewer.

        cli.py runs this on every run; the server offers it as GET /gotowosc.
        """
        self.preflight_configuration()

        for role in ("cv_writer", "cv_reviewer"):
            model = self.model(role)
            where = f"{model['model_id']} at {model['provider']}"
            try:
                # Through the retry policy, so a dropped socket does not kill
                # the run before it started. role is bound as a default so
                # the lambda does not capture the loop variable.
                answer = self.call_with_retries(
                    lambda role=role: self.send(role, "",
                                      'Odpowiedz dokładnie: {"ok": true}',
                                      as_json=True),
                    f"pre-flight for {role}",
                )
            except Exception as error:
                raise LlmOutputError(
                    f"Pre-flight failed for the '{role}' model {where}: "
                    f"{error}"
                )
            if not answer:
                raise LlmOutputError(
                    f"Pre-flight: the '{role}' model {where} answered nothing."
                )
