from dataclasses import dataclass
from typing import Optional


@dataclass
class Counterparty:
    """
    Represents a reinsurer / counterparty.
    """

    counterparty_id: str
    name: str
    country: str

    rating_agency: str
    credit_rating: str

    credit_limit: float
    utilized_limit: float

    probability_of_default: float

    outlook: Optional[str] = None

    def available_credit(self) -> float:
        return max(self.credit_limit - self.utilized_limit, 0)

    def utilization_ratio(self) -> float:
        if self.credit_limit == 0:
            return 0.0
        return self.utilized_limit / self.credit_limit
