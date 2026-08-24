from pydantic import BaseModel, Field
from enum import Enum

class AssetType(str, Enum):
    EQUITIES = "Equities"
    CORPORATE_BONDS = "Corporate Bonds"
    GOVERNMENT_BONDS = "Government Bonds"
    MORTGAGES_BACKED_SECURITIES = "Mortgages Backed Securities"

    @classmethod
    def _missing_(cls, value):
        #Account for previous generated articles using '(MBS)' in filename
        if value == "Mortgages Backed Securities (MBS)":
            return cls.MORTGAGES_BACKED_SECURITIES
        return None

class Sentiment(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

class Article(BaseModel):
    """Individual news article with metadata."""
    file_path: str
    asset_type: AssetType
    sentiment: Sentiment
    model: str
    repeat_id: int = Field(description="Repeat number for this article type")

class AttackType(str, Enum):
    ESCALATION = "escalation"
    SYSTEM_INSTRUCTION = "system_instruction"
    ALIGNMENT_ASSOCIATION = "alignment_association"
    MODEL_ADDRESS = "model_address"
    APPEAL_TO_AUTHORITY = "appeal_to_authority"

class InsertionRule(str, Enum):
    BY_ASSET = "by_asset"
    BY_IDX = "by_idx"