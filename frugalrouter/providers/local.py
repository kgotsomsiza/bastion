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

        work_rate = self._work_rate(prompt)
        if work_rate is not None:
            return work_rate, 0.98, ["computed_work_rate"]

        distance = self._distance_speed_time(prompt)
        if distance is not None:
            return distance, 0.98, ["computed_distance_speed_time"]

        age = self._age_ratio(prompt)
        if age is not None:
            return age, 0.98, ["computed_age_ratio"]

        conversion = self._exact_unit_conversion(prompt)
        if conversion is not None:
            return conversion, 0.98, ["computed_unit_conversion"]

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

    @staticmethod
    def _fmt_number(value: float) -> str:
        return str(int(value)) if float(value).is_integer() else f"{value:g}"

    def _work_rate(self, prompt: str) -> str | None:
        """'A finishes a job in X hours, B in Y hours; how long together?'

        together = XY/(X+Y). Fires only with exactly two rates, a together-ask,
        and no unaccounted numbers.
        """
        if not re.search(r"\b(?:together|working together|both work)\b", prompt, flags=re.IGNORECASE):
            return None
        if not re.search(r"\bhow (?:long|many hours|much time)\b", prompt, flags=re.IGNORECASE):
            return None
        rates = re.findall(
            r"\bin\s+(\d+(?:\.\d+)?)\s+(hours?|minutes?|days?)\b", prompt, flags=re.IGNORECASE
        )
        if len(rates) != 2 or rates[0][1].rstrip("s").lower() != rates[1][1].rstrip("s").lower():
            return None
        all_numbers = re.findall(r"\d+(?:\.\d+)?", prompt)
        if sorted(all_numbers) != sorted([rates[0][0], rates[1][0]]):
            return None
        x, y = float(rates[0][0]), float(rates[1][0])
        if x <= 0 or y <= 0:
            return None
        value = (x * y) / (x + y)
        unit = rates[0][1].rstrip("s").lower() + ("s" if value != 1 else "")
        return f"{self._fmt_number(round(value, 4))} {unit}"

    def _distance_speed_time(self, prompt: str) -> str | None:
        """Pure d = s*t / t = d/s with one speed and one time/distance, no extras."""
        speed = re.search(r"\b(\d+(?:\.\d+)?)\s*(km/h|kph|mph|miles per hour|kilometers per hour|m/s)\b",
                          prompt, flags=re.IGNORECASE)
        if not speed:
            return None
        unit = speed.group(2).lower()
        dist_unit = "km" if unit in {"km/h", "kph", "kilometers per hour"} else (
            "miles" if unit in {"mph", "miles per hour"} else "m")
        time_m = re.search(r"\bfor\s+(\d+(?:\.\d+)?)\s+hours?\b", prompt, flags=re.IGNORECASE)
        dist_m = re.search(rf"\b(\d+(?:\.\d+)?)\s*(?:{dist_unit}|kilometers|miles|meters)\b",
                           prompt, flags=re.IGNORECASE)
        all_numbers = re.findall(r"\d+(?:\.\d+)?", prompt)
        if time_m and re.search(r"\bhow (?:far|many (?:km|kilometers|miles|meters))\b", prompt, flags=re.IGNORECASE):
            if sorted(all_numbers) != sorted([speed.group(1), time_m.group(1)]):
                return None
            if unit == "m/s":
                return None  # hours * m/s mixes units; refuse
            return f"{self._fmt_number(float(speed.group(1)) * float(time_m.group(1)))} {dist_unit}"
        if dist_m and dist_m.group(1) != speed.group(1) and re.search(
                r"\bhow (?:long|many hours|much time)\b", prompt, flags=re.IGNORECASE):
            if sorted(all_numbers) != sorted([speed.group(1), dist_m.group(1)]):
                return None
            if unit == "m/s":
                return None
            hours = float(dist_m.group(1)) / float(speed.group(1))
            return f"{self._fmt_number(round(hours, 4))} hour" + ("s" if hours != 1 else "")
        return None

    def _age_ratio(self, prompt: str) -> str | None:
        """'X is twice/three times as old as Y. X is N years old. How old is Y?'"""
        ratio_m = re.search(
            r"\b([A-Z][a-z]+)\s+is\s+(twice|three times|four times|half)\s+as\s+old\s+as\s+([A-Z][a-z]+)",
            prompt,
        )
        if not ratio_m:
            return None
        elder, ratio_word, younger = ratio_m.group(1), ratio_m.group(2).lower(), ratio_m.group(3)
        factor = {"twice": 2.0, "three times": 3.0, "four times": 4.0, "half": 0.5}[ratio_word]
        age_m = re.search(rf"\b({elder}|{younger})\s+is\s+(\d+(?:\.\d+)?)(?:\s+years?\s+old)?\b", prompt)
        asked_m = re.search(r"[Hh]ow old is\s+([A-Z][a-z]+)", prompt)
        if not age_m or not asked_m:
            return None
        all_numbers = re.findall(r"\d+(?:\.\d+)?", prompt)
        if all_numbers != [age_m.group(2)]:
            return None
        known_name, known_age = age_m.group(1), float(age_m.group(2))
        asked = asked_m.group(1)
        if asked == known_name:
            return None
        # elder = factor * younger (factor 0.5 inverts the relation).
        if known_name == elder and asked == younger:
            value = known_age / factor
        elif known_name == younger and asked == elder:
            value = known_age * factor
        else:
            return None
        if value <= 0 or not float(value).is_integer():
            return None  # non-integer ages usually mean we mis-parsed; refuse
        return str(int(value))

    _EXACT_CONVERSIONS = {
        ("km", "m"): 1000.0, ("m", "cm"): 100.0, ("cm", "mm"): 10.0, ("m", "mm"): 1000.0,
        ("kg", "g"): 1000.0, ("g", "mg"): 1000.0,
        ("l", "ml"): 1000.0, ("liter", "ml"): 1000.0, ("litre", "ml"): 1000.0,
        ("hour", "minute"): 60.0, ("minute", "second"): 60.0, ("hour", "second"): 3600.0,
        ("day", "hour"): 24.0, ("week", "day"): 7.0,
    }

    def _exact_unit_conversion(self, prompt: str) -> str | None:
        """'Convert X km to meters' / 'How many minutes are in X hours' — exact factors only."""
        m = re.search(
            r"(?:convert\s+(\d+(?:\.\d+)?)\s*([a-z]+)\s+(?:to|into)\s+([a-z]+)"
            r"|how many\s+([a-z]+)\s+(?:are\s+)?in\s+(\d+(?:\.\d+)?)\s*([a-z]+))",
            prompt, flags=re.IGNORECASE,
        )
        if not m:
            return None
        if m.group(1):
            qty, src, dst = float(m.group(1)), m.group(2).lower(), m.group(3).lower()
        else:
            qty, src, dst = float(m.group(5)), m.group(6).lower(), m.group(4).lower()
        all_numbers = re.findall(r"\d+(?:\.\d+)?", prompt)
        if len(all_numbers) != 1:
            return None

        def canon(u: str) -> str:
            u = u.rstrip("s").lower()
            return {"meter": "m", "metre": "m", "kilometer": "km", "kilometre": "km",
                    "centimeter": "cm", "centimetre": "cm", "millimeter": "mm", "millimetre": "mm",
                    "gram": "g", "kilogram": "kg", "milligram": "mg",
                    "milliliter": "ml", "millilitre": "ml"}.get(u, u)
        src_c, dst_c = canon(src), canon(dst)
        if (src_c, dst_c) in self._EXACT_CONVERSIONS:
            return self._fmt_number(qty * self._EXACT_CONVERSIONS[(src_c, dst_c)])
        if (dst_c, src_c) in self._EXACT_CONVERSIONS:
            return self._fmt_number(qty / self._EXACT_CONVERSIONS[(dst_c, src_c)])
        return None

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
