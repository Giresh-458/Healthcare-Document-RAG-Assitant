from pydantic import BaseModel, Field
from typing import List, Literal

class MedicalAnswer(BaseModel):
    """Structured response contract requested from the model.

    Pydantic validates the returned shape, not the truth or clinical safety of
    the model's claims. Confidence is self-reported and is not calibrated.
    """
    answer: str = Field(description="The detailed text answer to the user's question, based strictly on the context.")
    confidence: Literal["high", "medium", "low"] = Field(description="Confidence level in the answer based on the clarity and presence of information in the context.")
    medical_entities: List[str] = Field(description="List of key medical terms, lab tests, or medications mentioned in the answer.")
    requires_doctor_review: bool = Field(description="True if the answer discusses abnormal lab values, diagnoses, or critical health metrics that a human doctor should verify.")
