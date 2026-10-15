"""Best-effort PHI pattern redaction for demo data.

IMPORTANT: this is not a de-identification system and does not establish HIPAA
compliance. Use synthetic data unless a separately validated privacy process is
in place. Regex patterns can both miss identifiers and redact legitimate values.
"""
import re


class PHIRedactor:
    PATTERNS = {
        "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "US_PHONE": re.compile(
            r"(?<!\d)(?:\+?1[ .-]?)?(?:\(?\d{3}\)?[ .-]?)\d{3}[ .-]?\d{4}(?!\d)"
        ),
        "US_SSN": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
        "MRN_LABELLED": re.compile(
            r"(?i)\b(?:medical record(?: number)?|mrn|patient id)\s*[:#-]?\s*[A-Z0-9-]{4,}\b"
        ),
        "DOB_LABELLED": re.compile(
            r"(?i)\b(?:date of birth|dob)\s*[:#-]?\s*"
            r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b"
        ),
    }

    @classmethod
    def redact(cls, text: str) -> str:
        redacted = text
        for label, pattern in cls.PATTERNS.items():
            redacted = pattern.sub(f"[REDACTED_{label}]", redacted)
        return redacted
