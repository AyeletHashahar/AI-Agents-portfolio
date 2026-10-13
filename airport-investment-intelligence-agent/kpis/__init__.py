"""KPI layer: raw KPI construction, comparison-group normalization, Investment Opportunity Score.

Deterministic, no API calls, no LLM. Pipeline:
    data_sources (retrieve) -> metrics (derive) -> kpis.build (KPIs per airport)
    -> kpis.scoring (percentile normalization + weighted score + ranking)

Public API (intended for a future agent):
    build_airport_kpis(airport, flights=..., t100_rows=..., runways=..., first_date=..., last_date=...)
    score_comparison_group([kpi_record, ...])
Definitions: docs/KPIS.md. Weights and coverage thresholds: config.py (re-exported by kpis/config.py).
"""

from kpis.build import build_airport_kpis
from kpis.scoring import score_comparison_group

__all__ = ["build_airport_kpis", "score_comparison_group"]
