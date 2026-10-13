"""Live behavioral evaluation of the Agent (separate from the production code).

    evaluation/cases.py               the 10 approved cases (EVALUATION_VERSION)
    evaluation/evaluator.py           deterministic checks -> six dimensions (no LLM judge)
    evaluation/storage.py             resumable run directory, atomic JSON writes, report
    evaluation/run_live_evaluation.py CLI runner (real Gemini; start/--resume)
"""
