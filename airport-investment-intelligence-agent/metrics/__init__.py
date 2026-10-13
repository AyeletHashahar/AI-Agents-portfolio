"""Deterministic airport metrics calculated from raw records already retrieved by data_sources.

Pure functions only: no API calls, no LLM, no scoring. Definitions: docs/METRICS.md.
- flights:    BTS On-Time rows  -> counts, cancellation rate, hourly counts, spacing,
                                   95th-percentile peak traffic, peak-period spacing, long-haul
- passengers: BTS T-100 rows    -> annual totals, YoY growth, CAGR trend, seasonality index
- airports:   OurAirports rows  -> state / New England, runway count, great-circle distance
"""
