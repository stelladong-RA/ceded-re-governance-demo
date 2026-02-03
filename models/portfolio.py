from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Portfolio:
    """
    Represents a ceded reinsurance portfolio.
    """

    portfolio_id: str
    name: str
    line_of_business: str
    geography: str

    total_gross_exposure: float
    total_ceded_exposure: float
    net_exposure: float

    capital_allocated: float
    expected_loss: float

    peril_mix: Dict[str, float]
    treaties: List[str]

    def utilization_ratio(self) -> float:
        if self.capital_allocated == 0:
            return 0.0
        return self.total_ceded_exposure / self.capital_allocated
