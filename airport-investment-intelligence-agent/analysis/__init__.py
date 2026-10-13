"""Deterministic analysis pipelines that combine data_sources, metrics and kpis (no LLM)."""

from analysis.comparison import analyze_comparison_group, index_flights_by_airport

__all__ = ["analyze_comparison_group", "index_flights_by_airport"]
