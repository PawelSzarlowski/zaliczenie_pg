"""Writing the CV, and rewriting it when the review asks for more.

Step 6. Both prompts share the same three blocks - content rules,
skeleton, form rules - so a rule cannot hold for one draft and not the other.

The contact block stays as placeholders: the knowledge base is anonymised, so
there is nothing to put there. Whoever sends the CV fills them in.
"""

import re

import config

# --------------------------------------------------------------------------
# The consent clause
# --------------------------------------------------------------------------

# A literal, never generated: a paraphrase of a statute is worth nothing.
# Here because both the prompt and the review need it.
GDPR_CLAUSE_PL = (
    "Wyrażam zgodę na przetwarzanie moich danych osobowych zawartych "
    "w niniejszym dokumencie do realizacji procesu rekrutacji zgodnie z ustawą "
    "z dnia 10 maja 2018 roku o ochronie danych osobowych (Dz. U. z 2018 r., "
    "poz. 1000) oraz zgodnie z Rozporządzeniem Parlamentu Europejskiego i Rady "
    "(UE) 2016/679 z dnia 27 kwietnia 2016 r. (RODO)."
)

GDPR_CLAUSE_EN = (
    "I agree to the processing of personal data provided in this document for "
    "realizing the recruitment process pursuant to the Personal Data "
    "Protection Act of 10 May 2018 (Journal of Laws 2018, item 1000) and in "
    "agreement with Regulation (EU) 2016/679 of the European Parliament and "
    "of the Council of 27 April 2016 (GDPR)."
)


def gdpr_clause(language):
    """The clause in the language of the CV."""
    return GDPR_CLAUSE_EN if language == "en" else GDPR_CLAUSE_PL


CV_SYSTEM = """
Jesteś doświadczonym doradcą zawodowym. Piszesz CV wyłącznie na podstawie
faktów z bazy wiedzy o kandydacie. Niczego nie zmyślasz.
"""

_CONTENT_RULES = """
##Zasady bezwzględne dotyczące treści:
+Używaj WYŁĄCZNIE faktów z bazy wiedzy o kandydacie.
+Nie dopisuj technologii, firm, dat, certyfikatów ani osiągnięć, których tam
 nie ma.
+Nie podawaj żadnych liczb, których baza wiedzy nie zawiera - w szczególności
 nie wymyślaj procentów, kwot ani wielkości zespołów.
+Możesz wybierać, skracać, porządkować i zmieniać kolejność.
+Jeżeli baza wiedzy nie zawiera danych do jakiejś sekcji, pomiń tę sekcję.
 Nigdy nie wypełniaj jej zmyślonymi danymi.

##Zasady redakcji doświadczenia:
+Każdy punkt zaczynaj od czasownika opisującego DZIAŁANIE (zaprojektował,
 wdrożył, zautomatyzował, opracował), a nie od nazwy obowiązku
 ("odpowiedzialny za...").
+Jeżeli baza wiedzy zawiera liczbę - skalę zespołu, długość projektu, wielkość
 systemu, wynik - UŻYJ jej. Liczba czyni osiągnięcie konkretnym.
+Jeżeli liczby NIE ma, opisz działanie i rezultat bez niej. Nie wolno dopisać
 procentu ani kwoty, żeby punkt wyglądał mocniej. Brak liczby jest mniejszym
 błędem niż liczba zmyślona.
+Nie pisz o gotowości, chęciach ani zainteresowaniach kandydata ("gotowość do
 pracy zmianowej", "otwartość na wyjazdy"). Baza wiedzy nie zawiera niczyich
 deklaracji, więc każde takie zdanie jest zmyśleniem - i to takim, które
 wygląda niewinnie, bo dotyczy przyszłości, a nie faktu z przeszłości.

##Do kogo piszesz:
+To CV czyta REKRUTER. Nie wolno w nim umieszczać uwag skierowanych do
 kandydata ani komentarzy o samym CV: "wymaga weryfikacji przed aplikacją",
 "kontekst uzupełniający", "w zakresie pracy inżynierskiej", "do
 potwierdzenia".
+Jeżeli nie masz pewności, czy coś wynika z bazy wiedzy - NIE dopisuj tego
 z zastrzeżeniem, tylko pomiń. Zastrzeżenie niczego nie ratuje: zmyślenie
 zostaje zmyśleniem, a rekruter dostaje w dokumencie zdanie, które czyta
 jako niepewność kandydata.
"""

_TEMPLATE = """
##Szkielet CV - trzymaj się tej kolejności i tych sekcji:

[IMIĘ I NAZWISKO]
[telefon] | [e-mail] | [LinkedIn]
[Miasto], [Kraj]

[TYTUŁ DOCELOWY - WERSALIKI]
[2-6 słów kluczowych oddzielonych znakiem |, każde z wielkiej litery]

[Podsumowanie: 4-6 zdań, akapit ciągły, BEZ nagłówka sekcji.]

KLUCZOWE UMIEJĘTNOŚCI
[Kategoria]: pozycja, pozycja, pozycja

DOŚWIADCZENIE ZAWODOWE

[NAZWA FIRMY], [Miejscowość]
[TYTUŁ ROLI - WERSALIKI]                          [rok] - [rok]
- [bullet bez kropki na końcu]

WYKSZTAŁCENIE
[Stopień i kierunek]
[Uczelnia], [rok]

CERTYFIKATY                      <- SEKCJA WARUNKOWA
JĘZYKI                           <- SEKCJA WARUNKOWA

---
*(tu klauzula zgody, przepisana co do słowa - patrz zasady poniżej)*

##Zasady szkieletu:
+**Danych kontaktowych NIE wypełniaj.** Zostaw nawiasy kwadratowe jako miejsca
 do uzupełnienia: [IMIĘ I NAZWISKO], [telefon], [e-mail], [LinkedIn],
 [Miasto], [Kraj]. Baza wiedzy jest zanonimizowana i tych danych nie zawiera -
 wpisanie czegokolwiek w to miejsce byłoby zmyśleniem.
+**To jedyne miejsce w całym CV, gdzie wolno zostawić nawias kwadratowy.**
 Wszędzie indziej nawias oznacza "wstaw tu prawdziwe dane". Jeżeli danych nie
 ma - usuwasz całą linię albo całą sekcję.
+**SEKCJE WARUNKOWE** wypisuj TYLKO wtedy, gdy baza wiedzy naprawdę zawiera
 odpowiednie dane. Nie wpisuj "Polski" ani "Angielski" tylko dlatego, że
 kandydat pewnie je zna - to zmyślenie, a nie oczywistość.
+Linia pod tytułem docelowym to same słowa kluczowe rozdzielone znakiem
 "|". NIE zaczynaj jej od "TAGLINE:" ani od żadnej innej etykiety - to
 nasza nazwa tej linii, a nie tekst, który ma się znaleźć w CV. Każde
 słowo kluczowe zaczynaj wielką literą: "Python | Analiza danych |
 Uczenie maszynowe".
+Nazwę firmy podawaj RAZ, a role w niej wymieniaj pod spodem.
+Daty ról podawaj jako same lata: "2019 - 2023"; rolę trwającą obecnie jako
 "2019 - obecnie".
+Nagłówki sekcji i tytuły ról zapisuj WERSALIKAMI.
+Kategorie w KLUCZOWYCH UMIEJĘTNOŚCIACH ustaw w kolejności, w jakiej
 odpowiadające im wymagania pojawiają się w ogłoszeniu. Maksymalnie 6.
+TYTUŁ DOCELOWY ma odpowiadać tytułowi z ogłoszenia, ale musi mieć pokrycie
 w doświadczeniu. Nie nadawaj poziomu "Senior" komuś, czyj profil go nie
 potwierdza.
+Nazwy sekcji tłumacz na język CV: KEY SKILLS, PROFESSIONAL EXPERIENCE,
 EDUCATION, CERTIFICATIONS, LANGUAGES.
+**Na samym końcu dokumentu, po ostatniej sekcji, umieść klauzulę zgody.**
 Oddziel ją poziomą linią (---) i zapisz kursywą. Przepisz CO DO SŁOWA
 ten tekst:
{gdpr_clause}
+Klauzula jest cytatem z przepisu, a nie zdaniem do redagowania. Nie
 zmieniaj w niej numerów, dat ani nazw aktów prawnych, nie skracaj jej
 i nie tłumacz na inny język.
"""

_FORM_RULES = """
+Długość: od 600 do 900 słów, czyli około półtorej do dwóch stron.
+Jeżeli tekst wychodzi za krótki, rozwiń opisy stanowisk o kolejne fakty
 OBECNE W BAZIE WIEDZY - więcej technologii, więcej ról, pełniejsze
 wykształcenie.
+WYJĄTEK - baza uboga. Kandydat na początku drogi zawodowej może mieć jedno
 krótkie zatrudnienie i brak certyfikatów. Wtedy 600 słów NIE DA SIĘ napisać
 uczciwie i krótsze CV jest poprawnym wynikiem: lepsze 400 słów treści niż 600
 rozwodnionych.
+Czego nie wolno NIGDY, nawet za cenę zbyt krótkiego CV: dopisywać czegokolwiek
 spoza bazy wiedzy, powtarzać tej samej informacji w kilku sekcjach ani
 zastępować konkretów ogólnikami o "zaangażowaniu".
+Format: Markdown. Zwróć samo CV - bez planu, bez komentarza, bez wyjaśnień
 przed nim ani po nim.
+Klauzulę o przetwarzaniu danych osobowych (RODO) dopisz na końcu, dokładnie
 w brzmieniu podanym w szkielecie poniżej. Dokument, który zwrócisz, jest
 tym, który przeczyta rekruter - nic się już do niego nie dokłada.

##Zasady stylu - TYLKO dla CV pisanego po polsku:
+Pisz w trzeciej osobie liczby pojedynczej, w czasie przeszłym: "Zajmował
 stanowisko...", "Wdrożył...". Nie w pierwszej osobie ani bezosobowo.
+WYJĄTEK - stanowisko zajmowane obecnie opisuj w czasie teraźniejszym.
+Baza wiedzy nie zawiera płci kandydata. Używaj form męskich, konsekwentnie.
+CV po angielsku pisz zwięzłymi punktami od czasownika ("Designed...",
 "Implemented...").
"""

WRITE_USER = """
##Baza wiedzy o kandydacie:
[BAZA-WIEDZY]
{profile_text}
[/BAZA-WIEDZY]

##Oferta pracy, pod którą powstaje CV:
[OFERTA]
{offer_text}
[/OFERTA]

##Napisz CV tego kandydata pod tę konkretną ofertę.
""" + _CONTENT_RULES + """
+Treść oferty służy WYŁĄCZNIE do tego, żeby zdecydować, co z bazy wiedzy
 wyeksponować, a czego nie wymieniać.

##Zasady dotyczące formy:
{language_rule}
""" + _FORM_RULES + _TEMPLATE

# The revision prompt: the same three blocks plus one rule of its own - a
# remark may not be paid for with an invented fact. The weakest criterion
# is usually C (concreteness), and the cheapest way to raise it is a number
# nobody measured, which the fact check then rejects.
REVISE_USER = """
##Baza wiedzy o kandydacie:
[BAZA-WIEDZY]
{profile_text}
[/BAZA-WIEDZY]

##Oferta pracy, pod którą powstaje CV:
[OFERTA]
{offer_text}
[/OFERTA]

##Poprzednia wersja CV, którą masz poprawić:
[POPRZEDNIE-CV]
{previous_cv}
[/POPRZEDNIE-CV]

##Uwagi oceniających do tej wersji:
[UWAGI]
{feedback}
[/UWAGI]

##Na czym skupić się w tej poprawce:
[POLECENIE]
{notes}
[/POLECENIE]

##Napisz nową, poprawioną wersję CV.
+Zwróć CAŁE CV od nowa, a nie listę zmian ani sam poprawiony fragment.
+Zdanie, które oceniający wskazał jako niepoparte bazą wiedzy, usuń albo
 złagodź do tego, co baza naprawdę mówi. To wykonaj bezwzględnie.
+POPRAWKA NIE MOŻE WNIEŚĆ ANI JEDNEGO FAKTU, KTÓREGO NIE MA W BAZIE WIEDZY.
 Jeżeli uwaga prosi o liczby, procenty, wielkości zespołów albo rezultaty,
 których baza nie zawiera - ZIGNORUJ tę uwagę. CV bez liczby jest gorzej
 oceniane, ale CV z liczbą wymyśloną jest odrzucane w całości.
+To, co było dobre, zostaw. Poprawiaj to, o czym mówią uwagi.
""" + _CONTENT_RULES + """
##Zasady dotyczące formy:
{language_rule}
""" + _FORM_RULES + _TEMPLATE


# --------------------------------------------------------------------------
# Small things done to the finished text
# --------------------------------------------------------------------------

def count_words(text: str) -> int:
    return len([word for word in re.split(r"\s+", text or "") if word])


# Function words: frequent in one language, absent from the other, and never
# dragged in by a place name. "a", "i", "to", "on", "by", "do" are words in
# both, so they are left out.
POLISH_FUNCTION_WORDS = {
    "w", "z", "na", "nie", "oraz", "dla", "jest", "się", "że", "przez",
    "jako", "przy", "który", "która", "które", "jego", "ich", "tego",
    "był", "była", "roku", "lat", "praca", "oraz",
}
ENGLISH_FUNCTION_WORDS = {
    "the", "and", "of", "for", "with", "was", "were", "from", "which",
    "their", "this", "these", "has", "have", "been", "including", "about",
    "into", "over", "across",
}

# Only for the tie-break below.
POLISH_LETTERS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")


def detect_language(text: str) -> str:
    """"pl" or "en", decided by whose function words are in the text.

    Place names bring no function words with them, so an English text naming
    Polish cities still reads as English. The share of words with a Polish
    letter is only the tie-break, for a text too short to hold a function
    word.
    """
    words = re.findall(r"[^\W\d_]+", (text or "").lower(), re.UNICODE)
    if not words:
        return "pl"

    polish = sum(1 for word in words if word in POLISH_FUNCTION_WORDS)
    english = sum(1 for word in words if word in ENGLISH_FUNCTION_WORDS)
    if polish != english:
        return "pl" if polish > english else "en"

    with_a_polish_letter = sum(
        1 for word in words if any(letter in POLISH_LETTERS for letter in word))
    return ("pl" if with_a_polish_letter / len(words) >= config.POLISH_WORD_SHARE
            else "en")


# The one line of the prompt that fixes the language of the document.
LANGUAGE_RULES = {
    "pl": "+Całe CV napisz po polsku.",
    "en": "+Całe CV napisz po angielsku. Write the whole CV in English.",
}


def language_rule(language):
    return LANGUAGE_RULES.get(language, LANGUAGE_RULES["pl"])


def flattened(text):
    """One line, single spaces. A different line wrap is not a change."""
    return " ".join((text or "").split())


# The skeleton asks for the clause in italics, so emphasis marks are ours.
_EMPHASIS = re.compile(r"[*_`]")


def clause_fragment(text):
    """The form in which any two pieces of clause text are compared.

    Used by both clause_is_intact (canonical inside document) and
    review.about_the_frame (quote inside canonical), so the two cannot
    disagree on what counts as the clause.
    """
    return _EMPHASIS.sub("", flattened(text)).lower()


def clause_is_intact(markdown, language):
    """Is the consent clause there, word for word?

    Checked by code: the reviewer compares the CV with the knowledge base,
    and a statute is in neither. Case and emphasis do not count; every
    number, date and name of an act does.
    """
    return clause_fragment(gdpr_clause(language)) in clause_fragment(markdown)


def contact_block_is_intact(markdown):
    """Is the first line still nothing but placeholders?

    A name a model puts there is invented and reads as plausible, which is
    why the fact check cannot be relied on to catch it. Only the first line:
    an English CV reasonably writes "[FULL NAME]" further down, and a city
    filled in on line three is left to the fact check.
    """
    for line in (markdown or "").split("\n"):
        if line.strip():
            return only_placeholders(line)
    return False


def document_defects(markdown, language):
    """What is wrong with the document itself. Found by code, never repaired.

    Each defect becomes a remark to the writer, and passes_quality refuses
    while one stands.
    """
    defects = []

    if not clause_is_intact(markdown, language):
        defects.append(
            "Klauzula zgody na przetwarzanie danych nie zgadza się co do "
            "słowa z obowiązującym brzmieniem - brakuje jej albo została "
            "zmieniona. To cytat z przepisu: przepisz go dokładnie, razem "
            "z numerami i datami.")

    if not contact_block_is_intact(markdown):
        defects.append(
            "Dokument nie zaczyna się od bloku kontaktowego w nawiasach "
            "kwadratowych. Baza wiedzy jest zanonimizowana, więc nazwisko, "
            "telefon albo adres wpisany w to miejsce jest zmyśleniem - "
            "przywróć nawiasy.")

    return defects


# A label in front of placeholders ("Lokalizacja:") is not a fact about
# anybody. Two to twenty letters and a colon.
_LEADING_LABEL = re.compile(r"^\s*[^\W\d_]{2,20}\s*:")


def only_placeholders(text):
    """Is there anything here except [placeholders], a label and punctuation?

        [Miasto], [Kraj]                     -> yes
        Lokalizacja: [Miejscowość], [Kraj]   -> yes, a label and holes
        Lokalizacja: [Miejscowość], Polska   -> NO. "Polska" is a claim
        [NAZWA FIRMY], 2019 - 2023           -> NO. The years are a claim
        Wdrożył [nazwa systemu] w Norwegii   -> NO

    Digits count as content, like letters. The label rule is a known rough
    edge - "Python: [poziom]" is filtered though "Python" is a claim - kept
    because "Lokalizacja: [Miejscowość], [Kraj]" is the common case.
    """
    if "[" not in (text or ""):
        return False
    rest = re.sub(r"\[[^\]]*\]", "", text)
    rest = _LEADING_LABEL.sub("", rest)
    return not re.search(r"[^\W_]", rest)


# --------------------------------------------------------------------------
# The two calls
# --------------------------------------------------------------------------

def write_cv(profile, offer_text, llm, language):
    """The first draft for one offer. Returns (markdown, word_count).

    The language comes from the caller, decided once per offer.
    """
    text = llm.ask_text(CV_SYSTEM, WRITE_USER,
                        {"profile_text": profile.raw_text,
                         "offer_text": offer_text,
                         "language_rule": language_rule(language),
                         "gdpr_clause": gdpr_clause(language)},
                        role="cv_writer")
    return text, count_words(text)


def revise_cv(profile, offer_text, previous_cv, feedback, notes, llm,
              language):
    """One improved version. Same shape back as write_cv."""
    text = llm.ask_text(CV_SYSTEM, REVISE_USER,
                        {"profile_text": profile.raw_text,
                         "offer_text": offer_text,
                         "previous_cv": previous_cv,
                         "feedback": feedback,
                         "notes": notes or "Popraw to, o czym mówią uwagi.",
                         "language_rule": language_rule(language),
                         "gdpr_clause": gdpr_clause(language)},
                        role="cv_writer")
    return text, count_words(text)
