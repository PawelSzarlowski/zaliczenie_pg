"""The command line: from a knowledge base to a CV ready to send.

    python cli.py --profile moj-profil.txt
    python cli.py --profile moj-profil.txt --location Gdańsk --salary 15000

Every threshold has an option here and a default in config.py, so a run can be
made cheap (--max-offers-for-cv 1 --max-cv-versions 2) without editing
anything. What a run produces lands in results/<data>/ - one Markdown CV per
offer, plus a report saying why those offers and how many attempts it took,
and llm_chat_log.jsonl: every prompt sent and every answer that came back,
one line per attempt, written as they happen.
"""

import argparse
import os
import sys

import agent as agent_module
import config
import offers as offers_module
import pipeline
import candidate as candidate_module
from llm import Llm, LlmOutputError


def say(message):
    """One line of progress, out at once.

    Flushed, because the writing stage takes minutes and a buffered
    line reads as a screen that has hung - which is exactly what it
    looks like when the output is a file rather than a console.
    """
    print(message, flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--profile", required=True,
                        help="a text file with the knowledge base. "
                             "profiles.py --extract makes one from the dataset")

    wanted_job = parser.add_argument_group("what kind of work")
    wanted_job.add_argument("--location", default="",
                        help="the city to search in. Polish spelling matters: "
                             "'Gdańsk' finds offers, 'Gdansk' finds none and "
                             "says nothing")
    wanted_job.add_argument("--salary", type=int, default=0,
                        help="the lowest acceptable monthly salary")

    system_config_limits = parser.add_argument_group("thresholds and limits")
    system_config_limits.add_argument("--min-offers", type=int, default=config.MIN_OFFERS,
                        help="how many offers are enough to stop relaxing the "
                             "search filters (default %(default)s)")
    system_config_limits.add_argument("--max-offers-to-assess", type=int,
                        default=config.MAX_OFFERS_TO_ASSESS,
                        help="how many of the offers found may be put in front "
                             "of the matching model (default %(default)s). A "
                             "ceiling, not a filter - every offer is asked "
                             "about until it bites")
    system_config_limits.add_argument("--match-score-threshold", type=int,
                        default=config.MATCH_SCORE_THRESHOLD,
                        help="the mark out of 100 an offer must get before a CV "
                             "is written for it (default %(default)s)")
    system_config_limits.add_argument("--max-offers-for-cv", type=int,
                        default=config.MAX_OFFERS_FOR_CV,
                        help="how many offers get a CV, best first "
                             "(default %(default)s)")
    system_config_limits.add_argument("--cv-quality-threshold", type=float,
                        default=config.CV_QUALITY_THRESHOLD,
                        help="the mark out of 100 at which a CV is good enough "
                             "to stop revising it (default %(default)s). Not a "
                             "sharp line - the reviewer is not repeatable")
    system_config_limits.add_argument("--max-cv-versions", type=int,
                        default=config.MAX_CV_VERSIONS,
                        help="how many versions of a CV may exist for one offer, "
                             "the first one included (default %(default)s)")
    system_config_limits.add_argument("--max-agent-steps", type=int,
                        default=config.MAX_AGENT_STEPS,
                        help="how many turns the agent may take before the loop "
                             "is cut off (default %(default)s)")

    offers_source = parser.add_argument_group("instead of searching")
    offers_source.add_argument("--offers-from", default=None, metavar="FILE",
                         help="take the offers from this file rather than "
                              "searching solid.jobs. A text file is one "
                              "advertisement pasted in; a .json is one offer, "
                              "a list of them, or the run.json of an earlier "
                              "run, whose offers can then be worked on again "
                              "without searching. Everything after this - the "
                              "fit assessment, writing, reviewing, revising - "
                              "happens exactly as it otherwise would")

    offers_source.add_argument("--force-offer", action="append", default=None,
                         metavar="KEY", dest="force_offers",
                         help="write a CV for this offer whatever the fit "
                              "assessment says. Repeatable. The key is the "
                              "one raport.md prints, whole or just its first "
                              "characters, the way git takes the start of a "
                              "commit hash. Naming even one offer REPLACES "
                              "the automatic choice: only the named offers "
                              "are assessed and only they get a CV")

    which = parser.add_argument_group(
        "which model does what",
        "One option per role. Without them a run uses config.ROLE_MODELS, "
        "which .env can steer as well.")
    for role in config.ROLES:
        which.add_argument("--model-" + role.replace("_", "-"),
                           dest="model_" + role, default=None,
                           choices=sorted(config.MODELS),
                           help=f"the model for {role} "
                                f"(default {config.ROLE_MODELS[role]})")

    other = parser.add_argument_group("other")
    other.add_argument("--out-dir", default=None,
                       help="where to write the CVs and the report "
                            "(default: results/<date>)")
    other.add_argument("--skip-preflight", action="store_true",
                       help="do not check the models before starting")
    other.add_argument("--dump-run", action="store_true",
                       help="also write run.json and every version of every "
                            "CV with its review, for looking into the run "
                            "afterwards")
    other.add_argument("--display-profile-cv-online", action="store_true",
                       help="Display candidate profile details and final CV")

    return parser


def models_from(options):
    """The roles the command line asked to change, and only those.

    Everything left out keeps what config.ROLE_MODELS says. Run with
    no --model-* option, behaves exactly as config.ROLE_MODELS sets.
    """
    chosen = {}
    for role in config.ROLES:
        value = getattr(options, "model_" + role, None)
        if value:
            chosen[role] = value
    return chosen


def settings_from(options):
    return config.Settings(
        location=options.location,
        minimum_salary=options.salary,
        min_offers=options.min_offers,
        max_offers_to_assess=options.max_offers_to_assess,
        match_score_threshold=options.match_score_threshold,
        max_offers_for_cv=options.max_offers_for_cv,
        cv_quality_threshold=options.cv_quality_threshold,
        max_cv_versions=options.max_cv_versions,
        max_agent_steps=options.max_agent_steps,
        force_offers=options.force_offers or [],
    )


def main():
    options = build_parser().parse_args()

    if not os.path.exists(options.profile):
        print("no such file:", options.profile)
        return 2

    if os.path.exists(config.ENV_FILE):
        # Says the file was found, and nothing about what is in it.
        print("settings read from .env")

    # Settled before the first model call, so the transcript of a run that
    # dies in pre-flight still has a directory.
    out_dir = options.out_dir or pipeline.default_out_dir()

    try:
        llm = Llm(role_models=models_from(options),
                  transcript_path=os.path.join(out_dir, config.TRANSCRIPT_FILE))
    except LlmOutputError as error:
        print("the models are misconfigured:", error)
        return 1

    for role, what in llm.models_in_use().items():
        print(f"   {role}: {what['model_id']} ({what['provider']})")
    print("every prompt and every answer:", llm.transcript_path)

    if not options.skip_preflight:
        # Better to stop now with a readable message than twenty minutes in.
        print("pre-flight: checking the models...")
        try:
            llm.preflight()
        except LlmOutputError as error:
            print("pre-flight failed:", error)
            return 1
        print("pre-flight: ok")

    print("reading the knowledge base...")
    try:
        if options.display_profile_cv_online:
            candidate_text = candidate_module.read_text_file(options.profile)
            print (f"\n***************************************************************\n")
            print (f"Fragment profilu kandydata")
            print (f"\n***************************************************************")
            print (f" \n{candidate_text[:800]} ... \n characters read from {options.profile}")
            print (f"***************************************************************")

        candidate = candidate_module.build_profile(
            candidate_module.read_text_file(options.profile), llm,
            location=options.location, minimum_salary=options.salary)
    except Exception as error:
        print("could not read the profile:", error)
        return 1

    print(f"   {len(candidate.skills)} skills, "
          f"level {candidate.experience_level or '?'}")

    given = None
    if options.offers_from:
        try:
            given = offers_module.offers_from_file(options.offers_from)
        except (OSError, ValueError) as error:
            print("could not read the offers:", error)
            return 2
        print(f"offers read from {options.offers_from}: {len(given)}")

    outcome = pipeline.run(candidate, llm, settings_from(options),
                           offers=given, offers_from=options.offers_from or "",
                           progress=say)

    if outcome.stopped_reason:
        print("stopped:", outcome.stopped_reason)
    for one in outcome.outcomes:
        if not one.versions:
            print(f"   {one.match_score}/100  {one.title} -> NO CV: "
                  f"{one.agent_stop_reason}")
            continue
        if not one.final_iteration:
            print(f"   {one.match_score}/100  {one.title} -> "
                  f"{len(one.versions)} versions, NONE REVIEWED, nothing "
                  f"sent: {one.agent_stop_reason}")
            continue
        # "Not rejected" is not "clean"; the unsupported sentences are in
        # the report.
        gate = ("  [fact check: REJECTED]"
                if one.final_rejected_by_fact_check else "")
        print(f"   {one.match_score}/100  {one.title} -> version "
              f"{one.final_iteration} of {len(one.versions)}{gate}")

    for path in pipeline.write_output(outcome, out_dir,
                                      dump=options.dump_run):
        print("written:", path)

    if options.display_profile_cv_online:
        print (f"\n***************************************************************\n")
        print (f"CV Kandydata")
        print (f"\n***************************************************************")

        for one in outcome.outcomes:
            version = agent_module.final_version(one)
            if version:
                print(version.markdown.rstrip() + "\n")

        print (f"\n***************************************************************")
    calls = sum(one.get("calls", 0) for one in outcome.usage.values())
    tokens = sum(one.get("input_tokens", 0) + one.get("output_tokens", 0)
                 for one in outcome.usage.values())
    print(f"retries: {llm.retries}, model calls: {calls}, "
          f"tokens reported: {tokens}, "
          f"total {outcome.seconds.get('razem', 0)} s")
    # Files first, exit code second: a partial run is worth keeping, but a
    # script must not read it as finished.
    return 3 if outcome.crashed else 0


if __name__ == "__main__":
    sys.exit(main())
