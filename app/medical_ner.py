import re
from typing import List

class MedicalNER:
    """
    Lightweight Medical Named Entity Recognizer to extract key healthcare terms
    from chunks. This helps with metadata filtering and keyword extraction.
    """
    
    # Common medical terms/categories to flag
    MEDICAL_TERMS = [
        r'\bhemoglobin\b',
        r'\brbc\b', r'\bred blood cell\b',
        r'\bwbc\b', r'\bwhite blood cell\b',
        r'\bplatelets\b',
        r'\bglycemic\b', r'\bglucose\b', r'\bhba1c\b',
        r'\bcholesterol\b', r'\bldl\b', r'\bhdl\b',
        r'\bdiagnosis\b', r'\bmedication\b', r'\bprescription\b',
        r'\bmg/dl\b', r'\bg/dl\b', r'\bmmol/l\b'
    ]

    @classmethod
    def extract_entities(cls, text: str) -> List[str]:
        """
        Extracts known medical entities and concepts from the text.
        """
        text_lower = text.lower()
        found_entities = set()
        
        for pattern in cls.MEDICAL_TERMS:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                # Add the matched string itself as an entity
                found_entities.add(match.group(0))
                
        return list(found_entities)
