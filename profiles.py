"""Browsing the Kaggle dataset and pulling one person out of it.

Everything that knows the dataset's shape lives here; candidate.py reads
plain text and nothing else.

The dataset is 9544 lines that are 344 people - everybody appears 4 to 28
times - so a row number is a coin flip. This script counts people:

    python profiles.py --list                      what is in there
    python profiles.py --list --find java          only the ones matching
    python profiles.py --show 12                   one knowledge base on screen
    python profiles.py --extract 12                save it as a text file

The saved file is what cli.py --profile expects, and can be read and edited
before a run. No model is called here and nothing goes over the network.
"""

import argparse
import ast
import csv
import os
import sys

import candidate as candidate_module
import config


def parse_list(raw):
    """A CSV cell holding something like "['Java', 'SQL']" as a real list."""
    if raw is None:
        return []
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null", "[]"}:
        return []
    try:
        value = ast.literal_eval(text)
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()]
    except (ValueError, SyntaxError):
        # Not a literal - treat it as a comma separated line.
        return [part.strip() for part in text.split(",") if part.strip()]


# The four columns that identify a person. The dataset repeats everybody 4
# to 28 times, differing only in "responsibilities", which is never read:
# 9544 lines are 344 people.
IDENTITY_COLUMNS = ["skills", "positions", "professional_company_names",
                    "educational_institution_name"]


def read_rows(path):
    """Every line of the dataset, as dictionaries."""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def distinct_people(path):
    """One entry per person, in the order they first appear.

    Dicts with "index" (0-based, the number --show and --extract take),
    "line" and "row".
    """
    people = []
    seen = set()
    for line, row in enumerate(read_rows(path)):
        identity = tuple((row.get(column) or "").strip()
                         for column in IDENTITY_COLUMNS)
        if identity in seen:
            continue
        seen.add(identity)
        people.append({"index": len(people), "line": line, "row": row})
    return people


def _row_as_text(row):
    parts = []

    objective = (row.get("career_objective") or "").strip()
    if objective:
        parts.append("Cel zawodowy:\n" + objective)

    positions = parse_list(row.get("positions"))
    companies = parse_list(row.get("professional_company_names"))
    starts = parse_list(row.get("start_dates"))
    ends = parse_list(row.get("end_dates"))

    lines = []
    for index, position in enumerate(positions):
        line = position
        if index < len(companies) and companies[index]:
            line += " w " + companies[index]
        start = starts[index] if index < len(starts) else ""
        end = ends[index] if index < len(ends) else ""
        if start or end:
            line += f" ({start} - {end})".replace("( - )", "")
        lines.append(line)
    parts.append(_section("Doświadczenie zawodowe", lines))

    parts.append(_section("Umiejętności", parse_list(row.get("skills"))))
    parts.append(_section("Wykształcenie",
                          parse_list(row.get("educational_institution_name"))))
    parts.append(_section("Kierunki i stopnie", parse_list(row.get("degree_names"))))
    parts.append(_section("Certyfikaty", parse_list(row.get("certification_skills"))))
    parts.append(_section("Języki", parse_list(row.get("languages"))))

    return "\n\n".join(part for part in parts if part)


def _section(title, items):
    if not items:
        return ""
    return title + ":\n" + "\n".join("- " + item for item in items)


def summary_line(person):
    """One person as one line of the listing."""
    row = person["row"]
    positions = parse_list(row.get("positions"))
    skills = parse_list(row.get("skills"))
    companies = parse_list(row.get("professional_company_names"))
    years = candidate_module.years_from_periods(
        parse_list(row.get("start_dates")),
        parse_list(row.get("end_dates")))
    level = candidate_module.experience_level(years)

    return "%4d  %-9s %5s  %3d skills  %s" % (
        person["index"],
        level or "?",
        ("%.1f" % years) if years is not None else "?",
        len(skills),
        " / ".join(positions[:2]) or " / ".join(companies[:1]) or "(no title)",
    )


def matches(person, needle):
    """Does the word appear in what they did, know, or where they worked?"""
    if not needle:
        return True
    row = person["row"]
    haystack = " ".join([
        str(row.get("positions") or ""),
        str(row.get("skills") or ""),
        str(row.get("professional_company_names") or ""),
        str(row.get("degree_names") or ""),
        str(row.get("major_field_of_studies") or ""),
    ]).lower()
    return needle.lower() in haystack


def knowledge_base(person):
    """The person as the text a CV may be written from.

    The career length is written into the text: the dataset ends an open
    job with "Till Date", which years_from_periods understands and a model
    told never to guess does not.
    """
    row = person["row"]
    text = _row_as_text(row)

    years = candidate_module.years_from_periods(
        parse_list(row.get("start_dates")),
        parse_list(row.get("end_dates")))
    if years is None:
        return text

    return (f"Łączny staż pracy: {years:.1f} roku "
            f"(policzony z dat zatrudnienia).\n\n" + text)


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--csv", default=config.DATASET_CSV,
                        help="the dataset (default: dataset/resume_data.csv)")

    parser.add_argument("--list", action="store_true",
                        help="list the people in the dataset")
    parser.add_argument("--find", default="",
                        help="only the people whose positions, skills, "
                             "companies or studies mention this word")
    parser.add_argument("--limit", type=int, default=40,
                        help="how many to list at once (default %(default)s; "
                             "0 for all of them)")

    parser.add_argument("--show", type=int, default=None, metavar="N",
                        help="print one person's knowledge base")
    parser.add_argument("--extract", type=int, default=None, metavar="N",
                        help="save one person's knowledge base as a text file")
    parser.add_argument("--out", default=None,
                        help="where to save it (default: profil-N.txt)")

    return parser


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    options = build_parser().parse_args()

    if not os.path.exists(options.csv):
        print("no dataset at", options.csv)
        print("copy resume_data.csv into dataset/, or pass --csv")
        return 2

    if options.show is None and options.extract is None and not options.list:
        build_parser().print_help()
        return 2

    people = distinct_people(options.csv)

    if options.list:
        chosen = [person for person in people if matches(person, options.find)]
        shown = chosen if options.limit <= 0 else chosen[:options.limit]

        print("%d people in the dataset" % len(people), end="")
        if options.find:
            print(", %d match %r" % (len(chosen), options.find), end="")
        print()
        print()
        print("   nr  level      years  skills  positions")
        for person in shown:
            print(summary_line(person))
        if len(shown) < len(chosen):
            print()
            print("... %d more; raise --limit or narrow it with --find"
                  % (len(chosen) - len(shown)))
        print()
        print("then: python profiles.py --extract <nr>")
        return 0

    wanted = options.show if options.show is not None else options.extract
    if not 0 <= wanted < len(people):
        print("there are %d people, numbered 0 to %d"
              % (len(people), len(people) - 1))
        return 2

    person = people[wanted]
    text = knowledge_base(person)

    if options.show is not None:
        print(text)
        return 0

    path = options.out or ("profil-%d.txt" % wanted)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")

    print("written:", path)
    print()
    print("read it, edit it if you like, then:")
    print('   python cli.py --profile %s --location "Gdańsk"' % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
