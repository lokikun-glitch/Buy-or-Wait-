"""Deterministic OCR-based amount extraction for financial_events rows whose
`amount` is blank.

No vision LLM is used (no API key available); instead we run local Tesseract
OCR (already installed) on the linked receipt/bill/payslip image and pick the
figure that matches the event's description using a priority-ordered list of
label keywords. Results are cached per image so each PNG is only OCR'd once.
"""
from __future__ import annotations

import os
import re
import shutil
from functools import lru_cache
from typing import Optional

import pytesseract
from PIL import Image

MEDIA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "media", "images")
)

if shutil.which("tesseract") is None:
    for _candidate in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ):
        if os.path.exists(_candidate):
            pytesseract.pytesseract.tesseract_cmd = _candidate
            break

_NUMBER = r"[\d][\d,\.\s]*\d|\d"


def _clean_number(raw: str) -> Optional[float]:
    raw = raw.strip().rstrip(".").replace(" ", "")
    # Indian/Indonesian/European grouping all use "," or "." as thousands
    # separators in these documents; the amount is always a whole/decimal
    # number with at most one trailing ".dd" fraction.
    if raw.count(",") and raw.count("."):
        # whichever separator appears last is the decimal separator
        if raw.rfind(",") > raw.rfind("."):
            raw = raw.replace(".", "").replace(",", ".")
        else:
            raw = raw.replace(",", "")
    elif raw.count(",") > 1:
        raw = raw.replace(",", "")
    elif raw.count(",") == 1:
        # could be decimal (European) or thousands -- treat as thousands
        # unless exactly 2 digits follow (decimal cents)
        head, tail = raw.split(",")
        raw = head + "." + tail if len(tail) == 2 else head + tail
    try:
        return float(raw)
    except ValueError:
        return None


@lru_cache(maxsize=None)
def ocr_text(image_id: str) -> str:
    """OCR the image, trying a couple of page-segmentation modes and
    concatenating the results -- tables/receipts often OCR better under one
    mode than another, and combining just gives the label search more
    candidate lines to work with."""
    path = os.path.join(MEDIA_DIR, f"{image_id}.png")
    if not os.path.exists(path):
        return ""
    img = Image.open(path)
    texts = []
    for psm in (6, 4, 11):
        try:
            texts.append(pytesseract.image_to_string(img, config=f"--psm {psm}"))
        except Exception:
            continue
    return "\n".join(texts)


# Priority-ordered label keywords. We scan the OCR'd text line by line and
# return the amount on the first line whose label matches, trying labels in
# this order (most specific/definitive first).
_LABEL_PRIORITY = [
    r"amount due till",
    r"balance due",
    r"balance\s*:",
    r"amount payable",
    r"total payable",
    r"total bill amount",
    r"net pay",
    r"total amount to be receive",
    r"grand total",
    r"net amount",
    r"total\s*[:\(]",
    r"total$",
    r"amount received",
    r"cash paid",
]

_STANDALONE_NUMBER_LINE = re.compile(r"^[=:\s]*(" + _NUMBER + r")\s*$")


def _find_amount_near_label(lines, label_pattern: str) -> Optional[float]:
    pat = re.compile(label_pattern, re.IGNORECASE)
    for line in lines:
        if pat.search(line):
            nums = re.findall(_NUMBER, line)
            if nums:
                val = _clean_number(nums[-1])
                if val is not None:
                    return val
    return None


_WORD_NUMS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_WORD_SCALES = {"hundred": 100, "thousand": 1000, "lakh": 100000, "lac": 100000, "million": 1000000}


_CENTS_WORDS = {"paise", "paisa", "cents", "cent", "sen"}
_SKIP_WORDS = {"and", "only", "rupees", "rupiahs", "rand", "euro", "euros", "dollars", "dollar", "rupiah", "rs"}


def _words_tokens_to_int(tokens: list) -> Optional[int]:
    total = 0
    current = 0
    found_any = False
    for tok in tokens:
        if tok in _SKIP_WORDS:
            continue
        if tok in _WORD_NUMS:
            current += _WORD_NUMS[tok]
            found_any = True
        elif tok in _WORD_SCALES:
            scale = _WORD_SCALES[tok]
            if scale == 100:
                current = (current or 1) * scale
            else:
                total += (current or 1) * scale
                current = 0
            found_any = True
    total += current
    return total if found_any else None


def _words_to_number(text: str) -> Optional[float]:
    """Parse a British/Indian-style spelled-out amount, e.g.
    'Four Million Three Hundred Sixty Five Thousand' -> 4365000,
    'Three Hundred and Ninety Three Rupees and Twenty Two Paise' -> 393.22.
    """
    tokens = re.findall(r"[a-z]+", text.lower().replace("-", " "))

    # cents are always the 1-2 number words immediately preceding a
    # paise/cents token (skipping a leading "and"); split there rather than
    # searching for "and" from the left, which can span an earlier "and"
    # inside the main amount (e.g. "Three Hundred AND Ninety Three ... Paise").
    cents = 0.0
    for i, tok in enumerate(tokens):
        if tok in _CENTS_WORDS:
            j = i - 1
            cent_tokens = []
            while j >= 0 and len(cent_tokens) < 2 and tokens[j] in _WORD_NUMS:
                cent_tokens.insert(0, tokens[j])
                j -= 1
            if j >= 0 and tokens[j] == "and":
                j -= 1
            cents_val = _words_tokens_to_int(cent_tokens)
            if cents_val is not None:
                cents = cents_val / 100.0
            tokens = tokens[: j + 1]
            break

    total = _words_tokens_to_int(tokens)
    if total is None and cents == 0.0:
        return None
    return float(total or 0) + round(cents, 2)


_WORD_CONNECTORS = {
    "and", "only", "rupees", "rupiahs", "rupiah", "rand", "euro", "euros",
    "dollars", "dollar", "paise", "paisa", "cents", "cent", "sen", "rs",
}


def _amount_in_words(text: str) -> Optional[float]:
    """Scan every line for a run of spelled-out-number tokens (with at least
    two actual numeric words, to avoid false positives) and parse the
    longest such run. This catches both "Amount in Words: ..." style labels
    and bare "Total: Seven Hundred Four Rupees and Five Paise Only" lines."""
    candidates: list[tuple[int, float]] = []  # (num_numeric_tokens, value)
    for line in text.splitlines():
        tokens = re.findall(r"[A-Za-z]+", line.lower())
        run: list[str] = []
        numeric_count = 0
        best_run = None
        best_count = 0

        def flush():
            nonlocal run, numeric_count, best_run, best_count
            if numeric_count >= 2 and numeric_count > best_count:
                best_run, best_count = list(run), numeric_count
            run, numeric_count = [], 0

        for tok in tokens:
            if tok in _WORD_NUMS or tok in _WORD_SCALES:
                run.append(tok)
                numeric_count += 1
            elif tok in _WORD_CONNECTORS:
                run.append(tok)
            else:
                flush()
        flush()

        if best_run:
            val = _words_to_number(" ".join(best_run))
            if val is not None and val > 0:
                candidates.append((best_count, val))

    if not candidates:
        return None
    # multiple OCR passes usually agree; take the most common value among the
    # longest (most-specific) candidates.
    max_count = max(c[0] for c in candidates)
    top = [v for c, v in candidates if c == max_count]
    return max(set(top), key=top.count)


def extract_amount(image_id: str) -> Optional[float]:
    text = ocr_text(image_id)
    if not text:
        return None

    words_value = _amount_in_words(text)

    lines = [ln for ln in text.splitlines() if ln.strip()]
    label_value = None
    for label in _LABEL_PRIORITY:
        val = _find_amount_near_label(lines, label)
        if val is not None:
            label_value = val
            break

    if words_value is not None:
        # Numeric table OCR is failure-prone (merged columns, stray digits);
        # prefer the spelled-out amount when it's present, since it's far
        # more robust to that kind of corruption. Fall back to the label
        # match only if no words total could be parsed.
        return words_value
    if label_value is not None:
        return label_value

    # Last resort: a bare "= 1,00,000.00" style line near the bottom of the
    # document (label/value split across OCR columns) is very often the
    # balance/total figure on these receipts.
    standalone = []
    for line in lines:
        m = _STANDALONE_NUMBER_LINE.match(line.strip())
        if m:
            val = _clean_number(m.group(1))
            if val is not None and val >= 10:
                standalone.append(val)
    if standalone:
        return max(set(standalone), key=standalone.count)
    return None
