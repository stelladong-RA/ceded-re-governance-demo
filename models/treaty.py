from dataclasses import dataclass
from typing import Optional


@dataclass
class Treaty:
    """
    Represents a reinsurance treaty or placement.
    """

    treaty_id: str
    name: str

    treaty_type: str          # QS, XoL, Layered
    business_line: str
    geography: str

    attachment_point: float
    limit: float
    cession_percentage: float

    reinsurer: str
    share_percentage: float

    inception_date: str
    expiry_date: str

    broker: Optional[str] = None
    notes: Optional[str] = None

    def max_recoverable(self) -> float:
        return self.limit * (self.share_percentage / 100)
