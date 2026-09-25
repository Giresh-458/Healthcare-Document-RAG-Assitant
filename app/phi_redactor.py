import re

class PHIRedactor:
    """
    Identifies and redacts Protected Health Information (PHI) like
    Phone Numbers, SSNs, and Dates of Birth using regular expressions.
    """
    
    # Regex patterns for common PHI
    PATTERNS = {
        "PHONE": r'\b(\+\d{1,2}\s?)?1?\-?\.?\s?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
        "SSN": r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
        "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
    }

    @classmethod
    def redact(cls, text: str) -> str:
        redacted_text = text
        for phi_type, pattern in cls.PATTERNS.items():
            # Replace matches with [REDACTED_<TYPE>]
            redacted_text = re.sub(pattern, f"[REDACTED_{phi_type}]", redacted_text)
            
        # Optional: Redact generic names using a simple heuristic if needed,
        # but regex covers the most concrete PHI types for now.
        return redacted_text
