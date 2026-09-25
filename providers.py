"""One class per kind of provider, and one shape for all of them.

A provider is handed a model id, a system text and a user text, and gives an
answer back. Nothing here knows what a CV is. Provider is a plain class with
NotImplementedError, not abc.ABC: two kinds do not earn a metaclass.
"""

import os

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

import config


class LlmOutputError(ValueError):
    """The model answered, but not with anything this code can use.

    Defined here because a provider has to raise it; llm.py re-exports it.
    """


def tokens_from(usage):
    """Whatever a provider called them, as our two numbers.

    LangChain says input_tokens/output_tokens, the OpenAI body says
    prompt_tokens/completion_tokens, some providers say nothing. An empty
    dict means "not reported"; a made-up 0 would read as a measurement.
    """
    if not usage:
        return {}

    went_in = usage.get("input_tokens", usage.get("prompt_tokens"))
    came_out = usage.get("output_tokens", usage.get("completion_tokens"))
    if went_in is None and came_out is None:
        return {}
    return {"input_tokens": int(went_in or 0),
            "output_tokens": int(came_out or 0)}


class Provider:
    """What every provider has to be able to do.

    Address and key name come from config.PROVIDERS. The key is read from
    the environment when needed and never kept on the object.
    """

    def __init__(self, name, base_url, key_env="", timeout=300):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.key_env = key_env
        self.timeout = timeout

    def send(self, model_id, system_text, user_text, as_json, extra=None):
        """One call. Returns (text, usage as the provider reported it, or None).

        `extra` is what the catalogue entry adds to the request body - how a
        model is pinned to one machine behind a broker, where the machine
        decides the speed. Providers with nothing to route ignore it.
        """
        raise NotImplementedError

    def chat_model(self, model_id):
        """The model object an agent can bind tools to."""
        raise NotImplementedError

    def capabilities(self, model_id):
        """What the provider says a model can do. Empty when it cannot say."""
        return []

    def api_key(self):
        """The key for this provider, or "" when it needs none."""
        if not self.key_env:
            return ""
        key = os.environ.get(self.key_env)
        if not key:
            raise LlmOutputError(
                f"{self.key_env} is not set, and the provider {self.name} "
                f"needs it. Put it in .env."
            )
        return key


class OllamaProvider(Provider):
    """An Ollama, through the LangChain client."""

    def client(self, model_id, as_json=False):
        return ChatOllama(
            model=model_id,
            base_url=self.base_url,
            format="json" if as_json else None,
            temperature=0,
            # The thinking phase is most of the call and does not change
            # the answer on these tasks.
            reasoning=False,
            # ChatOllama has no timeout of its own; without this a silent
            # socket waits for ever and the retry policy never fires.
            client_kwargs={"timeout": self.timeout},
        )

    def send(self, model_id, system_text, user_text, as_json, extra=None):
        # extra is ignored: Ollama is the machine, there is nothing to
        # route. invoke rather than a StrOutputParser chain, which would
        # drop the usage_metadata.
        answer = self.client(model_id, as_json).invoke(
            [SystemMessage(system_text), HumanMessage(user_text)])
        return answer.content, getattr(answer, "usage_metadata", None)

    def chat_model(self, model_id):
        """No format="json": forced JSON and tool calling cannot both be on."""
        return self.client(model_id, as_json=False)

    def capabilities(self, model_id):
        """What Ollama says a model can do. Empty on any error.

        "Could not check" must not become "cannot do it".
        """
        try:
            answer = requests.post(
                self.base_url + "/api/show",
                json={"model": model_id},
                timeout=config.API_TIMEOUT_SECONDS,
            )
            return answer.json().get("capabilities") or []
        except Exception:
            return []


class OpenAiCompatibleProvider(Provider):
    """Anything that speaks the OpenAI chat API.

    POST <base_url>/chat/completions with a Bearer key. Another provider of
    this kind is an entry in config.PROVIDERS and no code.
    """

    def send(self, model_id, system_text, user_text, as_json, extra=None):
        body = {
            "model": model_id,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
        }
        if as_json:
            body["response_format"] = {"type": "json_object"}
        # Last, so the catalogue entry can override what is above it.
        body.update(extra or {})

        answer = requests.post(
            self.base_url + "/chat/completions",
            headers={"Authorization": "Bearer " + self.api_key(),
                     "Content-Type": "application/json"},
            json=body,
            timeout=self.timeout,
        )
        answer.raise_for_status()
        data = answer.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as error:
            raise LlmOutputError(
                f"{self.name} answered in an unexpected shape: {error}; "
                f"{str(data)[:200]}"
            )

        return content, data.get("usage")

    def chat_model(self, model_id):
        return ChatOpenAI(
            model=model_id,
            base_url=self.base_url,
            api_key=self.api_key(),
            temperature=0,
            timeout=self.timeout,
        )


# The "kind" of a provider in config.PROVIDERS picks one of these.
KINDS = {
    "ollama": OllamaProvider,
    "openai": OpenAiCompatibleProvider,
}


def build(name):
    """The provider called `name`, built from config.PROVIDERS."""
    settings = config.PROVIDERS.get(name)
    if settings is None:
        raise LlmOutputError(
            f"There is no provider called '{name}'. config.PROVIDERS knows: "
            f"{', '.join(sorted(config.PROVIDERS))}."
        )

    kind = KINDS.get(settings.get("kind"))
    if kind is None:
        raise LlmOutputError(
            f"The provider '{name}' is declared as kind "
            f"'{settings.get('kind')}', which providers.py does not "
            f"implement. Known kinds: {', '.join(sorted(KINDS))}."
        )

    return kind(name, settings["base_url"], settings.get("key_env", ""),
                settings.get("timeout", 300))
