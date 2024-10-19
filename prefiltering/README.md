# Prefiltering
1. Run `common_hallicination_removal.py` to remove data with common hallucinated vocabs
2. Run `run_validator.py` to generate predication of validator, "whisper-base"
3. Run `elim_hallucination.py` to remove all hallucination by the result of validator and 