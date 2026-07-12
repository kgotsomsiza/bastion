from __future__ import annotations

from datetime import date
import re
import time
from dataclasses import dataclass

from frugalrouter.math_solver import solve_simple_math
from frugalrouter.types import Answer, Task


NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

US_STATE_NAMES = [
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
]

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@dataclass(frozen=True)
class LocalCandidate:
    answer: Answer
    confidence: float
    reasons: list[str]


class LocalProvider:
    name = "local"
    model = "deterministic-shortcuts"

    def answer(self, task: Task) -> LocalCandidate:
        started = time.perf_counter()
        text, confidence, reasons = self._answer_text(task)
        latency_ms = int((time.perf_counter() - started) * 1000)
        answer = Answer(text=text, provider=self.name, model=self.model, latency_ms=latency_ms)
        return LocalCandidate(answer=answer, confidence=confidence, reasons=reasons)

    def _answer_text(self, task: Task) -> tuple[str, float, list[str]]:
        prompt = task.input.strip()
        lower = prompt.lower()

        exact = self._exact_response(prompt)
        if exact is not None:
            return exact, 0.99, ["exact_response_instruction"]

        structured_ner = self._structured_ner(prompt)
        if structured_ner is not None:
            return structured_ner, 0.99, ["computed_structured_ner"]

        if self._is_sentiment_prompt(lower):
            return self._sentiment(prompt)

        code_debugging = self._code_debugging_shortcut(prompt)
        if code_debugging is not None:
            return code_debugging, 0.99, ["computed_loop_update"]

        factual = self._factual_shortcut(prompt)
        if factual is not None:
            return factual, 0.99, ["computed_factual_shortcut"]

        logic = self._logic_shortcut(prompt)
        if logic is not None:
            return logic, 0.99, ["computed_logic_shortcut"]

        calendar = self._calendar_weekday(prompt)
        if calendar is not None:
            return calendar, 0.99, ["computed_calendar_weekday"]

        time_answer = self._time_word_problem(prompt)
        if time_answer is not None:
            return time_answer, 0.98, ["computed_time_word_problem"]

        together = self._together_cost_algebra(prompt)
        if together is not None:
            return together, 0.98, ["computed_together_cost_algebra"]

        sequence = self._number_sequence(prompt)
        if sequence is not None:
            return sequence, 0.98, ["computed_number_sequence"]

        percent = self._percent_chain(prompt)
        if percent is not None:
            return percent, 0.98, ["computed_percent_chain"]

        math_answer = solve_simple_math(prompt)
        if math_answer is not None:
            text, reason = math_answer
            return text, 0.97, [reason]

        return "", 0.0, ["no_local_shortcut"]

    def _structured_ner(self, prompt: str) -> str | None:
        """Extract only mechanically complete, explicitly requested spans.

        This shortcut deliberately supports two closed-form entity types. It
        requires a colon-delimited source passage and refuses mixed entity
        requests, so a partial extraction cannot masquerade as a full answer.
        Every returned value is an exact source span.
        """
        if ":" not in prompt:
            return None
        instruction, source = (part.strip() for part in prompt.split(":", 1))
        if not instruction or not source:
            return None

        intent = instruction.lower()
        if re.search(r"\b(?:write|create|build|generate)\b.*\b(?:code|function|program|script|regex|regexp)\b", intent):
            return None
        if not re.search(r"\b(?:extract|identify|list|find|pull|return|contain)\w*\b", intent):
            return None

        email_requested = bool(re.search(r"\b(?:e-?mail)(?:\s+addresses?)?\b", intent))
        money_requested = bool(
            re.search(
                r"\b(?:monetary|money|currency|currencies|financial)\s+(?:values?|amounts?)\b"
                r"|\b(?:prices?|currency amounts?)\b",
                intent,
            )
        )
        if email_requested == money_requested:
            return None

        other_entity_kinds = (
            r"\b(?:people|persons?|names?|organizations?|organisations?|companies|locations?|"
            r"cities|countries|dates?|times?|phones?|telephone|regulations?|acronyms?|"
            r"formulas?|compounds?|products?)\b"
        )
        intent_without_target = re.sub(
            r"\b(?:e-?mail)(?:\s+addresses?)?\b"
            r"|\b(?:monetary|money|currency|currencies|financial)\s+(?:values?|amounts?)\b"
            r"|\b(?:prices?|currency amounts?)\b",
            "",
            intent,
        )
        if re.search(other_entity_kinds, intent_without_target):
            return None

        if email_requested:
            matches = re.findall(
                r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b(?![\w-])",
                source,
                flags=re.IGNORECASE,
            )
        else:
            number = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
            currency = re.compile(
                rf"(?<!\w)(?:(?:US\$|C\$|A\$|[$€£])\s*{number}|"
                rf"(?:USD|EUR|GBP|CHF|ZAR|R)\s*{number}|"
                rf"{number}\s*(?:US\s+dollars?|dollars?|euros?|pounds?|rand|francs?|"
                rf"USD|EUR|GBP|CHF|ZAR))(?!\w)",
                flags=re.IGNORECASE,
            )
            matches = [match.group(0) for match in currency.finditer(source)]

        if not matches:
            return None
        return self._format_structured_spans(prompt, matches)

    def _format_structured_spans(self, prompt: str, matches: list[str]) -> str | None:
        lower = prompt.lower()
        # Do not improvise structured containers whose exact schema is not
        # mechanically specified by this shortcut.
        if re.search(r"\b(?:json|xml|yaml|table|object|mapping|dictionary)\b", lower):
            return None
        if re.search(r"\bpipe-separated\b|\bseparated by (?:a )?pipe\b|\bpipe character\b", lower):
            return "|".join(matches)
        if re.search(r"\bsemicolon-separated\b|\bseparated by semicolons?\b", lower):
            return "; ".join(matches)
        if re.search(r"\bcomma-separated\b|\bseparated by commas?\b", lower):
            return ", ".join(matches)
        if re.search(r"\b(?:one per line|newline-separated|separate lines?)\b", lower):
            return "\n".join(matches)
        return "\n".join(matches)

    def _together_cost_algebra(self, prompt: str) -> str | None:
        """Classic 'together cost A; one costs B more than the other' algebra.

        cheap = (A - B) / 2, expensive = cheap + B. Fires only when both a
        combined amount and a more-than difference parse cleanly and the
        question names which item is asked; anything murkier goes remote.
        """
        together = re.search(
            r"together\s+(?:cost|costs|is|are|total)\s*\$?\s*(\d+(?:\.\d+)?)|"
            r"(?:cost|costs|total)\s*\$?\s*(\d+(?:\.\d+)?)\s+(?:together|in total|combined)",
            prompt,
            flags=re.IGNORECASE,
        )
        difference = re.search(
            r"(?:the\s+)?([\w-]+)\s+costs?\s*\$?\s*(\d+(?:\.\d+)?)\s+more\s+than\s+(?:the\s+)?([\w-]+)",
            prompt,
            flags=re.IGNORECASE,
        )
        if not together or not difference:
            return None
        total = float(together.group(1) or together.group(2))
        expensive_word = difference.group(1).lower()
        diff = float(difference.group(2))
        cheap_word = difference.group(3).lower()
        if total <= diff:
            return None
        cheap = (total - diff) / 2
        asks_cents = re.search(r"\bin\s+cents\b", prompt, flags=re.IGNORECASE)
        asked = re.search(r"how much (?:does|do|is|are)\s+(?:the\s+)?([\w-]+)", prompt, flags=re.IGNORECASE)
        if not asked:
            return None
        asked_word = asked.group(1).lower()
        # Only answer when the asked item is unambiguously one of the two.
        if asked_word == cheap_word:
            value = cheap
        elif asked_word == expensive_word:
            value = cheap + diff
        else:
            return None
        if asks_cents:
            return str(int(round(value * 100)))
        return str(int(value)) if float(value).is_integer() else f"{value:.2f}"

    def _number_sequence(self, prompt: str) -> str | None:
        """Next number in a sequence, only for exactly-recognized patterns.

        Requires >=4 terms and fires only when one of the classic generators
        (constant difference, constant ratio, constant second difference,
        Fibonacci-style sum) reproduces the WHOLE sequence; anything fuzzier
        goes to the remote model.
        """
        if not re.search(r"\bnext\s+(?:number|term)\b", prompt, flags=re.IGNORECASE):
            return None
        if not re.search(r"\bsequence|series\b", prompt, flags=re.IGNORECASE):
            return None
        tail = prompt.split(":", 1)[1] if ":" in prompt else prompt
        numbers = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", tail)]
        if len(numbers) < 4:
            return None

        def fmt(value: float) -> str:
            return str(int(value)) if float(value).is_integer() else f"{value:g}"

        diffs = [b - a for a, b in zip(numbers, numbers[1:])]
        if len(set(diffs)) == 1:
            return fmt(numbers[-1] + diffs[0])
        if all(n != 0 for n in numbers):
            ratios = [b / a for a, b in zip(numbers, numbers[1:])]
            if all(abs(r - ratios[0]) < 1e-9 for r in ratios):
                return fmt(numbers[-1] * ratios[0])
        second = [b - a for a, b in zip(diffs, diffs[1:])]
        if len(second) >= 2 and len(set(second)) == 1:
            return fmt(numbers[-1] + diffs[-1] + second[0])
        if len(numbers) >= 4 and all(
            abs(numbers[i] - (numbers[i - 1] + numbers[i - 2])) < 1e-9 for i in range(2, len(numbers))
        ):
            return fmt(numbers[-1] + numbers[-2])
        return None

    def _percent_chain(self, prompt: str) -> str | None:
        """Price after discount/tax chains, e.g. '$80, 25% off, then 10% tax'.

        Fires only when there is exactly one base amount, at least one
        percentage operation, and every percentage in the prompt is matched to
        a recognized operation - partial parses fall through to remote.
        """
        if not re.search(r"\b(?:final|total|after)\b.*\b(?:price|cost|amount|pay)\b|\b(?:price|cost|amount)\b.*\bafter\b", prompt, flags=re.IGNORECASE | re.DOTALL):
            return None
        # Any operation this solver does not model (splitting, per-person
        # shares, comparisons) means the parse is incomplete: refuse.
        if re.search(r"\b(?:split|divid\w*|among|between|each|per person|share[ds]?|apiece|evenly)\b", prompt, flags=re.IGNORECASE):
            return None
        amounts = re.findall(r"\$\s*(\d+(?:\.\d+)?)", prompt)
        if len(amounts) != 1:
            return None
        value = float(amounts[0])
        # Every number in the prompt must be either the base amount or one of
        # the percentages; a stray number means unmodeled arithmetic.
        percents = re.findall(r"(\d+(?:\.\d+)?)\s*%", prompt)
        accounted = {amounts[0], *percents}
        all_numbers = re.findall(r"\d+(?:\.\d+)?", prompt)
        if any(n not in accounted for n in all_numbers):
            return None
        operations = re.findall(
            r"\b(discount(?:ed)?|off|reduc\w+|markdown|tax|surcharge|fee|tip|increas\w+|markup)\b[^%]{0,40}?(\d+(?:\.\d+)?)\s*%"
            r"|(\d+(?:\.\d+)?)\s*%\s*(discount|off|tax|tip|surcharge|increase|markup)",
            prompt,
            flags=re.IGNORECASE,
        )
        total_percents = len(re.findall(r"\d+(?:\.\d+)?\s*%|\bpercent\b", prompt, flags=re.IGNORECASE))
        if not operations or len(operations) != total_percents:
            return None
        decrease_words = {"discount", "discounted", "off", "markdown"}
        for op in operations:
            word = (op[0] or op[3]).lower()
            percent = float(op[1] or op[2])
            base_word = re.sub(r"(ed|es|s)$", "", word)
            if word in decrease_words or base_word.startswith(("reduc", "markdown", "discount")):
                value *= 1 - percent / 100
            else:
                value *= 1 + percent / 100
        return str(int(round(value))) if abs(value - round(value)) < 0.005 else f"{value:.2f}"

    NEGATION_MARKERS = {"not", "never", "no", "hardly", "barely", "isn't", "wasn't", "aren't", "won't", "don't", "doesn't", "didn't", "can't", "couldn't", "nothing", "neither", "nor", "lacks", "without"}

    def _sentiment(self, prompt: str) -> tuple[str, float, list[str]]:
        positive = {"good", "great", "fast", "clear", "useful", "love", "excellent", "happy", "impressed"}
        negative = {"bad", "slow", "confusing", "broken", "hate", "poor", "wrong", "sad", "terrible", "awful"}
        words = set(re.findall(r"[a-zA-Z']+", prompt.lower()))
        if words & self.NEGATION_MARKERS:
            return "", 0.0, ["sentiment_negation_present"]
        # One keyword is too weak (sarcasm: "Oh great, another crash");
        # require a clear multi-keyword majority before answering locally.
        score = len(words & positive) - len(words & negative)
        if score >= 2:
            return "positive", 0.94, ["clear_sentiment_keywords"]
        if score <= -2:
            return "negative", 0.94, ["clear_sentiment_keywords"]
        return "", 0.0, ["no_clear_sentiment_keywords"]

    def _is_sentiment_prompt(self, prompt: str) -> bool:
        return "sentiment" in prompt and any(word in prompt for word in ["classify", "label", "positive", "negative"])

    def _exact_response(self, prompt: str) -> str | None:
        match = re.search(
            r"(?:reply|respond|answer)\s+with\s+exactly\s*(?::\s*([^\n.]+)|['\"]([^'\"]+)['\"])",
            prompt,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        # "Answer with exactly 'yes' or 'no': ..." offers alternatives; the right
        # choice depends on the actual question, so it is not a safe shortcut.
        tail = prompt[match.end() :]
        if re.match(r"\s*(?:,|or\b|and\b)", tail, flags=re.IGNORECASE):
            return None
        literal = (match.group(1) or match.group(2)).strip()
        if re.search(r"\bor\b", literal, flags=re.IGNORECASE):
            return None
        return literal

    def _factual_shortcut(self, prompt: str) -> str | None:
        lower = prompt.lower()
        if (
            "only letter" in lower
            and "not appear" in lower
            and ("u.s. state" in lower or "us state" in lower or "united states state" in lower)
        ):
            used = {char for state in US_STATE_NAMES for char in state.lower() if "a" <= char <= "z"}
            missing = [chr(code) for code in range(ord("a"), ord("z") + 1) if chr(code) not in used]
            if len(missing) == 1:
                return missing[0].upper()
        return None

    def _code_debugging_shortcut(self, prompt: str) -> str | None:
        lower = prompt.lower()
        if not re.search(r"\b(?:missing|omit(?:ted)?|forgot(?:ten)?)\b", lower):
            return None
        if not re.search(r"\b(?:operation|update|increment|decrement)\b", lower):
            return None
        if not re.search(r"\b(?:freeze|freezes|frozen|infinite\s+loop)\b", lower):
            return None

        loop = re.search(
            r"\bwhile\s*\(\s*([A-Za-z_]\w*)\s*(<=?|>=?)\s*[^)]+\)\s*\{([^}]*)\}",
            prompt,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not loop:
            return None

        variable, comparison, body = loop.groups()
        if re.search(
            rf"\b{re.escape(variable)}\b\s*(?:\+\+|--|[+\-*/]?=)",
            body,
        ):
            return None
        return f"{variable}++" if comparison.startswith("<") else f"{variable}--"

    def _logic_shortcut(self, prompt: str) -> str | None:
        power_strip_answer = self._power_strip_empty_outlets(prompt)
        if power_strip_answer is not None:
            return power_strip_answer
        return None

    def _power_strip_empty_outlets(self, prompt: str) -> str | None:
        lower = prompt.lower()
        if not all(marker in lower for marker in ["power strip", "outlet"]):
            return None
        if "wall outlet" not in lower or not re.search(r"\bempty\b|\bavailable\b|\bremain", lower):
            return None

        strips = self._extract_number_before(r"(?:identical\s+)?power strips?", lower)
        outlets = self._extract_number_before(r"outlets?", lower)
        if strips is None or outlets is None or strips < 1 or outlets < 1:
            return None

        # First strip plugs into the wall; each additional strip consumes one
        # outlet on an earlier strip. Empty outlets = all strip outlets minus
        # the strip-to-strip connections.
        return str(strips * outlets - max(0, strips - 1))

    def _time_word_problem(self, prompt: str) -> str | None:
        lower = prompt.lower()
        if not re.search(r"\b(?:what time|earliest time|completely done|finish(?:ed|es)?|ready)\b", lower):
            return None

        start = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap]\.?m\.?)\b", lower, flags=re.IGNORECASE)
        if not start:
            return None

        duration_minutes = 0
        for match in re.finditer(
            r"\b(?:(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+hours?)"
            r"(?:\s+and\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
            r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\s+minutes?)?"
            r"|\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
            r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\s+minutes?",
            lower,
        ):
            hours = self._parse_number(match.group(1)) if match.group(1) else 0
            minutes = self._parse_number(match.group(2) or match.group(3)) if (match.group(2) or match.group(3)) else 0
            duration_minutes += hours * 60 + minutes

        if duration_minutes <= 0:
            return None

        hour = int(start.group(1))
        minute = int(start.group(2) or 0)
        suffix = start.group(3).replace(".", "").lower()
        if hour == 12:
            hour = 0
        if suffix == "pm":
            hour += 12

        total = (hour * 60 + minute + duration_minutes) % (24 * 60)
        out_hour_24, out_minute = divmod(total, 60)
        out_suffix = "AM" if out_hour_24 < 12 else "PM"
        out_hour_12 = out_hour_24 % 12 or 12

        if re.search(r"\bHH:MM\b", prompt):
            return f"{out_hour_12:02d}:{out_minute:02d} {out_suffix}"
        return f"{out_hour_12}:{out_minute:02d} {out_suffix}"

    def _calendar_weekday(self, prompt: str) -> str | None:
        lower = prompt.lower()
        if not re.search(r"\bday of the week\b|\bwhat day\b", lower):
            return None

        date_pattern = (
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)"
            r"\s+(\d{1,2})(?:st|nd|rd|th)?[,]?\s+(\d{4})\b"
        )
        matches = list(re.finditer(date_pattern, lower, flags=re.IGNORECASE))
        if len(matches) < 2:
            return None

        base_weekday = None
        base_context = lower[matches[0].end() : matches[0].end() + 80]
        for index, name in enumerate(WEEKDAYS):
            if re.search(rf"\b{name.lower()}\b", base_context):
                base_weekday = index
                break
        if base_weekday is None:
            return None

        base_date = self._date_from_match(matches[0])
        target_date = self._date_from_match(matches[-1])
        if base_date is None or target_date is None:
            return None

        offset = (target_date - base_date).days % 7
        return WEEKDAYS[(base_weekday + offset) % 7]

    def _extract_number_before(self, following_pattern: str, text: str) -> int | None:
        pattern = rf"\b(\d+|{'|'.join(NUMBER_WORDS)})\s+{following_pattern}"
        match = re.search(pattern, text)
        return self._parse_number(match.group(1)) if match else None

    def _parse_number(self, value: str) -> int:
        return int(value) if value.isdigit() else NUMBER_WORDS[value.lower()]

    def _date_from_match(self, match: re.Match[str]) -> date | None:
        try:
            return date(int(match.group(3)), MONTHS[match.group(1).lower()], int(match.group(2)))
        except ValueError:
            return None
