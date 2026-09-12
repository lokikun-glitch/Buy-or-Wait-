"""Deterministic, rule-based fact extraction from messages.csv.

No LLM is used (per project decision -- no API key available). Messages are
untrusted evidence: we only ever extract structured facts via pattern
matching against known templates (English + Indonesian, the two languages
present in this dataset); embedded instructions in the text are never
executed, only currency/amount/date facts are pulled out.

Extracted facts feed src/events.py as overrides on top of the
recurrence-detected forecast (see build_income_facts / build_event_facts).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Optional

import pandas as pd

AMOUNT_RE = r"([A-Z]{3})\s*(\d[\d,]*(?:\.\d+)?)"
DATE_RE = r"(\d{4}-\d{2}-\d{2})"


def _to_date(s: str) -> Optional[date]:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def _parse_amount(num: str) -> float:
    # Dataset amounts are plain "1037.52" / "42750000" style -- comma is never
    # used as a thousands separator in these generated messages, but guard anyway.
    cleaned = num.replace(",", "")
    return float(cleaned)


@dataclass
class MessageFact:
    message_id: str
    user_id: str
    kind: str  # see KIND_* constants below
    currency: Optional[str] = None
    amount: Optional[float] = None
    extra_amount: Optional[float] = None
    effective_date: Optional[date] = None
    related_event_id: Optional[str] = None
    sent_at: Optional[str] = None


# income-stream facts (applied to the user's recurring "salary" category)
KIND_SALARY_NEXT_AMOUNT = "salary_next_amount"       # next occurrence amount changes (temporary/reduced)
KIND_SALARY_NEXT_DATE = "salary_next_date"            # next occurrence date changes
KIND_SALARY_CONFIRMED = "salary_confirmed"            # one-time confirmed amount on a specific date, recurring from there
KIND_SALARY_BASE_ONLY = "salary_base_only"            # confirmed base amount; variable component excluded
KIND_SALARY_ARREARS = "salary_arrears"                # regular amount + one-time extra on same date
KIND_SALARY_STREAM_ENDED = "salary_stream_ended"      # no more recurring income after last settled date
# one-off confirmed future income (e.g. approved invoice payout)
KIND_CONFIRMED_ONE_OFF_INCOME = "confirmed_one_off_income"


_EMPLOYER_PATTERNS: List[tuple] = [
    # "Your first salary will be EUR 620. The confirmed credit date is 2026-07-15."
    (
        re.compile(
            r"first salary\D*" + AMOUNT_RE + r".*?(?:confirmed credit date is|confirmed for)\D*" + DATE_RE,
            re.IGNORECASE | re.DOTALL,
        ),
        KIND_SALARY_CONFIRMED,
    ),
    (
        re.compile(
            r"[Gg]aji pertama\D*" + AMOUNT_RE + r".*?(?:dikonfirmasi untuk|dijadwalkan pada)\D*" + DATE_RE,
            re.DOTALL,
        ),
        KIND_SALARY_CONFIRMED,
    ),
    # "Your salary of EUR 1485 is confirmed for 2025-05-15."
    (
        re.compile(r"salary of\s*" + AMOUNT_RE + r"\s*is confirmed for\D*" + DATE_RE, re.IGNORECASE),
        KIND_SALARY_CONFIRMED,
    ),
    # "Regular salary of INR 251000 resumes on 2026-01-15."
    (
        re.compile(r"[Rr]egular salary of\s*" + AMOUNT_RE + r"\s*resumes on\D*" + DATE_RE),
        KIND_SALARY_CONFIRMED,
    ),
    # "Remaining confirmed monthly salary is IDR X" (Indonesian, income source ended)
    (
        re.compile(r"[Ss]isa gaji bulanan yang dikonfirmasi adalah\s*" + AMOUNT_RE),
        KIND_SALARY_BASE_ONLY,
    ),
    # temporary / reduced pay continuing "for the next payroll"
    (
        re.compile(r"temporary monthly pay is\s*" + AMOUNT_RE, re.IGNORECASE),
        KIND_SALARY_NEXT_AMOUNT,
    ),
    (
        re.compile(r"next salary is reduced to\s*" + AMOUNT_RE, re.IGNORECASE),
        KIND_SALARY_NEXT_AMOUNT,
    ),
    (
        re.compile(r"monthly salary has increased to\s*" + AMOUNT_RE, re.IGNORECASE),
        KIND_SALARY_NEXT_AMOUNT,
    ),
    (
        re.compile(r"[Gg]aji bulanan Anda naik menjadi\s*" + AMOUNT_RE),
        KIND_SALARY_NEXT_AMOUNT,
    ),
    # "confirmed salary is now expected on DATE"
    (
        re.compile(r"confirmed salary is now expected on\D*" + DATE_RE, re.IGNORECASE),
        KIND_SALARY_NEXT_DATE,
    ),
    # base salary confirmed, commission/bonus still pending -> count base only
    (
        re.compile(r"confirmed base salary is\s*" + AMOUNT_RE, re.IGNORECASE),
        KIND_SALARY_BASE_ONLY,
    ),
    (
        re.compile(r"[Gg]aji pokok yang dikonfirmasi adalah\s*" + AMOUNT_RE),
        KIND_SALARY_BASE_ONLY,
    ),
    # regular salary + one-time arrears shown separately
    (
        re.compile(
            r"regular salary for the next payroll is\s*" + AMOUNT_RE
            + r".*?one-time arrears adjustment of\s*" + AMOUNT_RE,
            re.IGNORECASE | re.DOTALL,
        ),
        KIND_SALARY_ARREARS,
    ),
    (
        re.compile(
            r"[Gg]aji rutin Anda untuk penggajian berikutnya adalah\s*" + AMOUNT_RE
            + r".*?penyesuaian tunggakan satu kali sebesar\s*" + AMOUNT_RE,
            re.DOTALL,
        ),
        KIND_SALARY_ARREARS,
    ),
    # employment ended / seasonal contract ended -> stream stops
    (
        re.compile(r"employment has ended", re.IGNORECASE),
        KIND_SALARY_STREAM_ENDED,
    ),
    (
        re.compile(r"seasonal contract has ended", re.IGNORECASE),
        KIND_SALARY_STREAM_ENDED,
    ),
]

_SERVICE_PROVIDER_CONFIRMED_INVOICE = re.compile(
    r"approved an invoice payment of\s*" + AMOUNT_RE + r".*?[Ss]ettlement is expected on\D*" + DATE_RE,
    re.DOTALL,
)


def extract_facts(message_row: pd.Series) -> List[MessageFact]:
    text = str(message_row.get("message_text") or "")
    mid = message_row.get("message_id")
    uid = message_row.get("user_id")
    related = message_row.get("related_event_id")
    related = None if pd.isna(related) else related
    source = message_row.get("source_type")
    facts: List[MessageFact] = []

    if source == "employer":
        for pattern, kind in _EMPLOYER_PATTERNS:
            m = pattern.search(text)
            if not m:
                continue
            groups = m.groups()
            if kind == KIND_SALARY_STREAM_ENDED:
                facts.append(MessageFact(mid, uid, kind, sent_at=message_row.get("sent_at")))
            elif kind == KIND_SALARY_ARREARS:
                cur, amt, cur2, extra = groups
                facts.append(
                    MessageFact(
                        mid, uid, kind, currency=cur, amount=_parse_amount(amt),
                        extra_amount=_parse_amount(extra), sent_at=message_row.get("sent_at"),
                    )
                )
            elif kind == KIND_SALARY_NEXT_DATE:
                (dt,) = groups
                facts.append(
                    MessageFact(mid, uid, kind, effective_date=_to_date(dt), sent_at=message_row.get("sent_at"))
                )
            elif kind in (KIND_SALARY_CONFIRMED,) and len(groups) == 3:
                cur, amt, dt = groups
                facts.append(
                    MessageFact(
                        mid, uid, kind, currency=cur, amount=_parse_amount(amt),
                        effective_date=_to_date(dt), sent_at=message_row.get("sent_at"),
                    )
                )
            elif kind in (KIND_SALARY_NEXT_AMOUNT, KIND_SALARY_BASE_ONLY) and len(groups) == 2:
                cur, amt = groups
                facts.append(
                    MessageFact(
                        mid, uid, kind, currency=cur, amount=_parse_amount(amt),
                        sent_at=message_row.get("sent_at"),
                    )
                )
            break  # first matching template wins for this message

    elif source == "service_provider":
        m = _SERVICE_PROVIDER_CONFIRMED_INVOICE.search(text)
        if m:
            cur, amt, dt = m.groups()
            facts.append(
                MessageFact(
                    mid, uid, KIND_CONFIRMED_ONE_OFF_INCOME, currency=cur, amount=_parse_amount(amt),
                    effective_date=_to_date(dt), related_event_id=related, sent_at=message_row.get("sent_at"),
                )
            )

    return facts


def extract_all_facts(messages_df: pd.DataFrame) -> List[MessageFact]:
    facts: List[MessageFact] = []
    for _, row in messages_df.iterrows():
        facts.extend(extract_facts(row))
    return facts


def latest_salary_facts(facts: List[MessageFact]) -> List[MessageFact]:
    """Sort salary-affecting facts chronologically by sent_at so later facts
    (newer records from the same source) win on conflicts, per the dataset's
    conflict-resolution priority."""
    salary_kinds = {
        KIND_SALARY_NEXT_AMOUNT, KIND_SALARY_NEXT_DATE, KIND_SALARY_CONFIRMED,
        KIND_SALARY_BASE_ONLY, KIND_SALARY_ARREARS, KIND_SALARY_STREAM_ENDED,
    }
    relevant = [f for f in facts if f.kind in salary_kinds]
    relevant.sort(key=lambda f: f.sent_at or "")
    return relevant
