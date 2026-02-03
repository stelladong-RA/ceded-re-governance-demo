from dataclasses import dataclass
from typing import Optional


@dataclass
class Cashflow:
    """
    Represents a cash transaction related to reinsurance.
    """

    cashflow_id: str
    treaty_id: str
    counterparty: str

    transaction_date: str
    currency: str

    gross_amount: float
    ceded_amount: float
    net_amount: float

    transaction_type: str      # Premium / Recovery / Commission / Adjustment
    status: str                # Pending / Matched / Disputed / Settled

    reference_id: Optional[str] = None
    statement_id: Optional[str] = None
    notes: Optional[str] = None

    def is_settled(self) -> bool:
        return self.status.lower() == "settled"

    def is_disputed(self) -> bool:
        return self.status.lower() == "disputed"
