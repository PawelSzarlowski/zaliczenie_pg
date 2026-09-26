# Agent CV

One candidate, a few offers, one CV worth sending.

Give it a knowledge base about a person and it finds job offers that suit
them, writes a CV for the best few, checks each CV for facts it invented, and
keeps rewriting until the document is good enough - or until it can say why it
will not get better.

This is a practical tool, not a study. There is no batch mode, no statistics
and nothing to calibrate: it serves one person at a time and its output is a
Markdown file and a report explaining it.

## What it does

```
a knowledge base (a text file; profiles.py makes one from the dataset)
  -> a model reads out the search criteria
  -> offers are downloaded, relaxing the filters until there are enough
     (the technology goes first, the seniority level last)
  -> a model scores the fit of every offer found; only the best few get a CV
  -> for each of those, an agent writes, reviews and rewrites the CV
  -> results/<date>-<id>/cv-*.md, raport.md and llm_chat_log.jsonl
```

Five roles, and `config.py` says which model plays each one. One rule is not
a preference: **the model that writes a CV may not be the model that reviews
it.** A model grading its own work grades it kindly, and that grade is what
drives the rewriting - so pre-flight refuses to start when the two share a
family.

## Before you start

You need Python, an Ollama reachable at the address in `.env`, and a key for
whichever cloud provider the roles point at - OpenRouter out of the box.

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
```

Then put your key in `.env`. Every threshold has a default there too, and a
command line option that overrides it.

> `requirements.txt` must stay UTF-8. Editors on Windows will save it as
> UTF-16 given half a chance, and pip's complaint - "Expected package name at
> the start of dependency specifier" - does not mention encoding at all.

## Running it

```
.venv\Scripts\python.exe cli.py --profile examples\przyklad-profilu.txt
.venv\Scripts\python.exe cli.py --profile moj-profil.txt --location Gdańsk --salary 15000
```

`examples/przyklad-profilu.txt` is a made-up knowledge base, there so that the
first run needs nothing but a key. It also shows the shape the tool expects:
plain text, sections, and no personal data in it.

The screen says what is happening while it happens. Writing a CV takes
minutes, so every call to the writer and the reviewer is announced before it
goes out, and the marks of each version appear as soon as its review is done:

```
   v1: 57.5/100 (A=45 B=60 C=30 D=90 E=95), fact check passed, 0 unsupported, 0 document defects
```

A refusal from one of the agent's tools is shown too. Every line is flushed at
once, so a run whose output goes to a file shows them as they come.

## Using the dataset

`dataset/resume_data.csv` is the Kaggle resume set: **9544 lines that are 344
people**, because everybody appears 4 to 28 times. Opening it by hand is
hopeless, so `profiles.py` browses it:

```
.venv\Scripts\python.exe profiles.py --list
.venv\Scripts\python.exe profiles.py --list --find java
.venv\Scripts\python.exe profiles.py --show 12
.venv\Scripts\python.exe profiles.py --extract 12
```

```
344 people in the dataset, 57 match 'java'

   nr  level      years  skills  positions
    0  Senior       6.2   21 skills  Big Data Analyst
    2  Senior       7.6   14 skills  Software Developer (Machine Learning Engineer)
   13  Intern       0.2   22 skills  Intern / Intern
```

`--extract 12` writes `profil-12.txt`, which is exactly what `--profile`
expects. That is the normal way round:

```
.venv\Scripts\python.exe profiles.py --extract 12
.venv\Scripts\python.exe cli.py --profile profil-12.txt --location Gdańsk
```

That is the only road from the dataset to a run: `cli.py` reads a text file
and nothing else, so you get to read what the agent will be told **before**
paying for a run - and you can edit it. `profiles.py` calls no model and
touches no network.

> **A number means a person, not a line.** `profiles.py` counts distinct
> people, identified by their skills, positions, companies and school. Lines
> 0 to 27 of the file can all be the same person, so a line number would be
> a coin flip between copies.

The dataset is git-ignored: 17 MB, and it is somebody's résumés. Download it
from Kaggle and drop `resume_data.csv` into `dataset/`, or point
`profiles.py --csv` wherever you keep it.

`cli.py --help` lists every option with its default. The ones worth knowing:

| option | default | what it changes |
|---|---|---|
| `--max-offers-for-cv` | 3 | how many offers get a CV at all |
| `--cv-quality-threshold` | 70 | when a CV is good enough to stop (0-100, like every other mark) |
| `--max-cv-versions` | 5 | how many attempts one offer may cost |
| `--match-score-threshold` | 70 | how well an offer must fit to be worth writing for |

A cheap first run: `--max-offers-for-cv 1 --max-cv-versions 2`.

**A recon run: `--max-offers-for-cv 0`.** It reads the knowledge base,
searches and writes the report without asking a model about a single offer
and without writing a CV - so you can see which offers a knowledge base
actually finds, and whether the division came out right, before spending
more. Worth doing on every new profile, with `--dump-run` so the offers are
kept in `run.json`.

The city has to be spelled in Polish. `Gdańsk` finds offers; `Gdansk` finds
none, and the API says nothing about why - it answers 200 with an empty list.

## Choosing the models

Three tables in `config.py`, and nothing below them ever names a model:

```
ROLE_MODELS   "cv_writer"               ->  "minimax-m3-openrouter"
MODELS        "minimax-m3-openrouter"   ->  minimax/minimax-m3, at openrouter
PROVIDERS     "openrouter"              ->  an address and the name of a key
```

To have something else write the CVs, change one line of `ROLE_MODELS` - or
set `ROLE_CV_WRITER` in `.env`, or pass `--model-cv-writer` for a single run.
A model the catalogue does not know yet is one entry in `MODELS` plus that
same line.

The five roles are `criteria_searcher`, `offer_matcher`, `cv_writer`,
`cv_reviewer` and `cv_agent`. Named after the job, never after the model or
the place it runs in, because which model does what is a decision that
changes - and `MODEL_CLOUD_B` is a poor name for the thing that reviews CVs.

Two things an entry **declares** rather than letting the code work out
from the name: `family`, because the writer and the reviewer may not share
one, and `supports_tools`, because the agent needs a client that can call
tools. A name like `nemotron-3-ultra:cloud` shows why a name is not a safe
source for either.

A third thing an entry may declare is `extra_body`, which is added to the
request. Today it does one job: **OpenRouter is a broker, not a provider.**
Asked for one `model_id` it picks the machine itself, out of a dozen or more.
Measured on one prompt, the same `openai/gpt-oss-120b` weights answered in
1.5 s on Cerebras, 6.3 s on Groq, 28.7 s when the broker chose, and 61.6 s on
DeepInfra - a forty-fold spread. A catalogue that cannot name the machine
cannot promise anything about speed, so a pinned entry has to say so in its
name and may not leave provider fallbacks on; both rules are tests. The price
of a pinned entry is **that machine's** price, not the model's headline one,
which is the cheapest endpoint's. An entry without the field adds nothing to
the request.

Two more fields are about rubbish: `clean_prompt` cleans the text a model is
sent, `clean_answer` cleans what it sends back, before the rest of the code
sees it. Both are off unless an entry says otherwise, and by default an
answer is kept exactly as it came. They exist because one model, measured,
followed a 1163-character CV with 126 424 em spaces (U+2003) - invisible on
the screen, and enough to push the reviewer past its context window, so no
version got reviewed. Cleaning does three things and no more: a run of
typographic spaces becomes one ordinary space, zero-width characters go, and
whitespace at the very end goes. Ordinary spaces and newlines inside the text
are left alone, because two spaces at the end of a Markdown line are a line
break and the contact block is built out of them. It applies to the agent's
turns as well, which bypass the ordinary call path: the offer it starts from
and every tool result on the way in, the text arguments of its tool calls on
the way out. The transcript keeps every answer as it came, and on the lines of
every role but the agent's it records how much was cut; the report says it for
all of them under the model table, so a run started over HTTP, which writes no
transcript, shows it too. `gpt-oss:120b-ollama` is the one entry with `clean_answer` on.

Providers come in two kinds: `ollama`, and anything speaking the OpenAI chat
API. OpenRouter, ExperientialLabs and the Hugging Face router are all the
second, so they share one class in `providers.py` and differ in nothing but an
address and the name of a key. A fourth of that kind is six lines of
`PROVIDERS` and no code. The kind is declared, not derived from the name, and
the catalogue test refuses a kind `providers.py` does not implement.

## An offer that did not come from the search

`--offers-from FILE` replaces steps 2 and 3. Nothing is searched for and no
query is built; everything after that - the fit assessment, the writing, the
reviewing, the revising - runs exactly as it otherwise would, because the
flow works on the offer dict and the prompts only ever see `offer_text` of
it.

    python cli.py --profile baza.txt --offers-from ogloszenie.txt

Four shapes of file are accepted. A text file is one advertisement pasted in:
select all on a job page, paste, save. It may begin with `Tytuł:`, `Firma:`,
`Poziom:`, `Umiejętności:`, `URL:` lines, and the first line that is not one
of those starts the description - so an advertisement beginning "Wymagania:"
is kept whole rather than half-parsed. With no header at all, the whole file
is the description and the title comes from the file name.

A `.json` file is one offer in the API's own shape, a list of them, a whole
answer from the API, or **the `run.json` of an earlier run**. That last one
is the useful shape: a run can be repeated for one offer without searching
again, and the search is live - an offer that was there an hour ago need not
be there now.

The thresholds still apply. An offer read from a file is not a licence to
write a CV for it: if the matching model scores it below
`--match-score-threshold`, no CV is written, exactly as for a searched offer.

`POST /przygotuj` takes the same thing as `offers` in its body - the offers
themselves, never a path. A path would be opened on the server, and choosing
which of its files to read is not something this endpoint has any business
offering.

## A CV for the offer you point at

`--force-offer KEY` writes a CV for that offer whatever the matching model
scored it. The key is the one `raport.md` prints - whole, or just its first
characters, the way git takes the start of a commit hash. A prefix that
matches two offers stops the run and names them both rather than picking one.

    python cli.py --profile baza.txt         --offers-from results/2026-09-08-143012-a3f1c2d4/run.json         --force-offer 1d5ecb2a

That form is the one that always works, and it is why runs keep their offers.
Forcing resolves the key against the offers of THIS run, so the bare
`--force-offer 1d5ecb2a` searches again first - and the search is live. If
the offer has gone, the run stops and says no key matched, which is loud and
correct but not what you wanted. Reading the offers out of the earlier
`run.json` skips the search altogether and cannot miss.

Naming even one offer **replaces** the automatic choice instead of adding to
it: only the named offers are assessed and only they get a CV. That is what
makes it cheap - four offers found and one named costs one `offer_matcher`
call, not four - and it is also why a key that matches nothing is a warning
you can see, and why no key matching anything stops the run. Forcing does not
quietly fall back to the run that would have happened anyway.

It reaches past the score threshold as well: an offer the model rates 30 out
of 100 never gets a CV on its own, and naming it is how you overrule that -
the mark is still taken and still printed in the report, so overruling it is
done in the open.

The two features compose into the thing they were built for - paste an
advertisement into a file and get a CV for it, with no search and with no
threshold in the way:

    python cli.py --profile baza.txt         --offers-from ogloszenie.txt --force-offer ogloszenie

The fit assessment still runs for a forced offer, and its mark and reasons
still go into the report - a poor score there means "the gate was not
applied", which is different from "the gate let this through", and the report
says which. What the assessment never does is reach the writer: nothing tells
the model that the fit was weak, so the document is written the same way it
would be for an offer that qualified on its own.

## What you get

`results/<date>-<id>/` holds one `cv-*.md` per offer the agent wrote for, plus
`raport.md`. An offer the agent gave up on without asking for a single version
gets no file and a line in the report saying so - nothing writes a CV on the
agent's behalf. The
eight characters on the end are what keeps two runs that finish in the same
second out of each other's directory - which the command line has to work
at, and a web front end would do by accident.

`llm_chat_log.jsonl` is written by the command line with no option at all (a run
started over HTTP writes none): every prompt sent and every
answer that came back, one line per **attempt**, so a retried call leaves two.
Failed calls get a line of their own with the error in it, and the file is
closed after each line - a run that is killed halfway still holds everything
up to the kill. No key can reach it: the `Authorization` header is built in
`providers.py` and `send()` never sees one.

`--dump-run` adds `run.json` and one `cv-*-v1.md`, `-v2.md` … per version
written. Without it only the version that gets sent is written: the ones the
loop rejected, the analysis behind every mark and what the agent asked for at
each step stay in memory and are gone. `run.json` is the same answer
`POST /przygotuj` returns, and the per-version files are it laid out for
reading, each with a pointer to the lines of `llm_chat_log.jsonl` that
produced it.

Both of them carry the offers a model was actually asked about, exactly as
the API sent them. That is what makes it possible to come back to one offer
later without searching again - the search is live, and an offer that was
there an hour ago need not be there now. Only the assessed ones: one division
without a city filter answers with 500 offers, which is 2.5 MB of JSON, and
all of it would travel in every HTTP answer.

`raport.md` prints the first eight characters of each offer key, beside
every mark. They are the same eight the CV file names use, so a row in the
report and a file on disk can be matched.

The CVs keep `[placeholders]` in the contact block. The knowledge base is
anonymised and has no name, phone or address in it, so there is nothing to put
there - fill them in before sending. The consent clause stands at the bottom,
in the language of the CV: the model writes it out word for word and the code
checks that it did.

That language is the language of the offer, and it is decided once, from the
advert, before a word is written - then told to the writer, so every version
of one CV shares it. What decides it is a count of function words (`w`,
`oraz`, `jest` against `the`, `and`, `with`) rather than the share of Polish
letters: an English advert naming Gdańsk, Kraków and Wrocław carries no Polish
function words at all. The share of Polish letters is only the tie-break, for
a text too short to hold a function word.

The report says which offers were found, how each scored, why the chosen ones
were chosen, how many versions each CV took and what the fact check objected
to. It exists so that somebody holding a CV can check why they got that one.

It also says which model played which role and what the run cost. The count of
calls is exact, because the code counts it itself; the tokens appear only where
the provider reported them, because Ollama does not always and a written-in
zero would read as a measurement. The `cv_agent` row is the agent's own turns,
which never go through the ordinary call path and would otherwise be invisible
- and they are the largest, because every turn re-sends the whole conversation.
The work the agent ordered through its tools is counted under `cv_writer` and
`cv_reviewer`, not a second time under it. When a catalogue entry has the
code clean a prompt or an answer, a sentence under the table says how many
characters were taken out in which role.

It opens with **"Na co zwrócić uwagę"** when the run noticed something the
numbers alone would hide: not one offer at the candidate's seniority (the
market simply may not have any); no seniority known at all, so nothing was
narrowed by it and a junior may be read against senior jobs; the model shown
a division's categories and picking none, which usually means the division is
wrong; every filter dropped and still too few offers; more offers found than
the assessment ceiling allows; an offer the model failed to score.

## The HTTP interface

```
.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Interactive documentation at `/docs`.

| endpoint | what it does |
|---|---|
| `GET /` | is it alive |
| `GET /modele` | the roles, the catalogue, and which model is in which role |
| `GET /gotowosc` | the full pre-flight, live calls included - about a minute |
| `POST /oferty` | search and rank only - seconds, no CV written |
| `POST /przygotuj` | the whole flow - minutes |

Kept for a web front end later on. `POST /oferty` and `POST /przygotuj` call
`pipeline.run`, exactly as the command line does, so a screen built on this
cannot drift away from what the command line does.

`GET /modele` is there for that screen too: it answers with the roles, what
may go in them and what is in them now - a drop-down per role, ready to
build. A request to `POST /przygotuj` may then carry
`{"models": {"cv_writer": "..."}}`. It goes through the same pre-flight, so
picking a reviewer from the writer's family in a browser is refused exactly
as it is on the command line; a name that is in no catalogue is a 400 before
anything runs.

Pre-flight is split in two, because its checks do not cost the same.
`POST /przygotuj` runs the configuration half on **every** request - the
writer and the reviewer may not come from one family, and the agent model has
to be able to call tools - and answers 500 in milliseconds when either is
wrong. Without that the request would spend minutes and money and hand back a
CV graded by its own author, which is the one rule this project calls
non-negotiable.

The other half - one small live call to the writer and one to the reviewer -
is `GET /gotowosc`, to be asked once after the server starts rather than on
every request; it takes as long as those two models take to answer. It
answers 503 when a model does not reply.

> Whoever builds that front end will need a job queue in front of
> `POST /przygotuj`: one run takes minutes and no browser waits that long.
> That is why the work is one function - a worker can call it without any of
> this being rewritten.

## The files

| file | role |
|---|---|
| `config.py` | **every threshold, limit, path and address**, and the `.env` reader |
| `candidate.py` | the knowledge base, from a text file, into a profile |
| `offers.py` | search criteria, the API, the relaxation ladder |
| `llm.py` | a role to a model to a provider; retries; pre-flight; refuses an empty answer |
| `pipeline.py` | the whole flow in one function, and the report; always returns an outcome, crash or no crash |
| `providers.py` | how to reach one provider. One class per kind, not per name |
| `cv.py` | writing and rewriting; the prompts |
| `review.py` | the fit score, the fact check and the quality marks |
| `agent.py` | the agent's tools, its loop, and the choice of final version |
| `models.py` | the shapes that travel between the modules |
| `cli.py`, `main.py` | the two doors into it |
| `profiles.py` | run by hand: browse the dataset, pull one person out of it |
| `dataset/` | the Kaggle résumé set; git-ignored, 17 MB and somebody's CVs |
| `data/` | a fact about the outside world, not code: the categories solid.jobs splits its offers into |
| `examples/` | a made-up knowledge base to run against |
| `tests/` | one file per module; **none needs the network or a model** |
| `results/` | what runs produce; git-ignored, because a CV is about a person |
| `tools/` | the documentation sources and the scripts that build them |

```
.venv\Scripts\python.exe -m unittest discover -s tests -t .
```

`-t .` matters: it puts the project root on the import path, so the tests
import `config`, `offers` and the rest by their plain names.

## What is not obvious

**The agent decides, the code only refuses.** The agent picks the path - what
to ask for, whether another attempt is worth it, when to stop - and nothing is
done on its behalf: an agent that ends without writing ends without a CV; a
version it never asked to have reviewed cannot be sent; an answer in prose
instead of a tool call is the end of its work. Each of those is written into
the report as what the agent did, and `llm_chat_log.jsonl` holds the turn
where it did it.

The limits live in the tools rather than in the prompt, because a prompt is a
request and a tool is a wall - and a wall refuses, it does not perform.
`tool_write_cv` says no past the version limit and after two revisions that
improved nothing; it says why, and what to call instead; and the ending is
the agent's own, through `tool_finish_cv`. The one thing that ends a loop the agent
has not ended is the step limit, which is money and not judgement.

The version that gets sent is chosen by arithmetic, never by the agent and
never "the last one". Only reviewed versions are candidates; those the fact
check did not reject come first, and among those the ones without a single
unsupported sentence, whatever their marks - an invented skill outside the
hard sections does not reject a version, but the reviewer rewards it with a
higher mark; within `SCORE_NOISE` (7 points) of the best
mark the marks say nothing, so the version with fewer unsupported sentences
wins, then fewer document defects, then the earlier one. A measured run went
48.5 → 49.75 → 50.65 with the third version rejected by the fact check: the
highest mark never took part in the choice, and the first version was sent.

**What gets reviewed is what gets sent, byte for byte.** The model writes the
whole document - contact placeholders, consent clause and all - and nothing is
appended to it afterwards. The reviewer therefore sees the `[brackets]` and
the clause, which a checker can read as claims about the candidate that
nothing supports. So the FINDINGS are filtered instead of the document:
anything quoting the contact block or the clause is dropped, one step later
and to the same effect.

Nothing is trimmed on the way to disk either: after three hidden header lines
the file holds the answer exactly as it came back, so rubbish in it is there
to be seen. A model that pads its answers is dealt with in its catalogue entry
(`clean_answer`), which cleans the text once, where it arrives - the file, the
reviewer and the next revision still see one and the same text.
`clean_prompt` on the reviewer's model would break this: the reviewer would
read a cleaned document and the file would keep the rubbish.

Two things about the document itself are checked by code and **not repaired**:
the consent clause has to be there word for word, and the contact block has to
still be placeholders. Both become remarks the writer is given on the next
attempt, and neither lets the loop call a document ready. Nothing is put right
behind the model's back - a clause silently corrected is a mistake nobody ever
learns about and a run that reports itself clean. The clause is checked here
rather than by the reviewing model for a plain reason: it is a quotation of a
statute, so the reviewer has nothing to compare it against, and handing it the
canonical text would be a string comparison carried out by a language model.

**Every mark is out of 100** - how well an offer fits, and how good the CV is.
One scale, so the two can be read side by side and the two thresholds mean
the same thing.

**Neither score is repeatable, and both are thresholds.** Measured on one
version of one CV, three reviews per model: `gpt-oss:120b` reported 4, 1 and
1 unsupported sentences and marked it 48.0, 49.25 and 49.75;
`nemotron-3-super` marked the same document 70.75, 41.0 and 43.5. Both
defaults are useful and both are sharp lines drawn on a blurred quantity. If
a run finds nothing, running it again is a reasonable thing to do; comparing
two runs to the point is not.

**The quality threshold is rarely reached.** Quality marks measured so far
land between 40 and 67, so in practice the agent seldom sees PRÓG SPEŁNIONY:
it writes until `tool_write_cv` refuses - the version limit, or two revisions
without progress - and then ends with `tool_finish_cv`. The report says which.

**Career length is computed, never asked for.** The model is asked to copy the
employment dates out of the knowledge base, and the arithmetic happens in code
- the same function `profiles.py` uses on the dataset's date columns. A model
asked for a number instead answers `null` for somebody whose whole record is
two one-month internships - to it that is "nothing worth reporting" rather
than "zero" - while copying "July 2021 - Aug 2021" gives it no trouble. The
dates are counted in months, because a job inside one calendar year is not no
time worked - and that is exactly the range where `Intern` and `Junior` part
company.

**The search may name several technologies at once.** The API takes
`search.subCategories` more than once and answers with the union (measured by
set identity, not by counting). The sub-category list is a list of
*languages*, so somebody doing machine learning has no row of their own in it,
and a candidate who knows five languages would otherwise be reduced to one.
Four at a time, because naming all fourteen is arithmetically the same as
naming none. The comma form `Python,Java` answers with something else and no
complaint, so it is not used.

**Every offer found goes to the matching model.** There is no cheap pre-score
in front of it: a comparison of skill names is not a ranking - an offer
asking for `Kontrola jakości` and `Test planning` scores zero against a
candidate listing `MCTP`, `PCIe` and `FPGA`, and the model may then rate it
85 out of 100 - so nothing orders the queue before the model sees it. The only
bound is a ceiling, `MAX_OFFERS_TO_ASSESS`, 50 by default. Find more than that
and the report says so outright - the search was too broad and nobody asked
about the rest. Nothing is cut quietly, because there is nothing to sort the
list by, and claiming the offers below the cut were the worse ones would be a
claim nothing supports.
