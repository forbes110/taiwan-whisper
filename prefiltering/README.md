# Prefiltering
1. Run `common_hallicination_removal.py` to remove data with common hallucinated vocabs from the result of `initial_inference.sh`
2. Run `validator_inference.py` to generate predication of validator, "whisper-base"
3. Run `elim_hallucination.py` to remove all hallucination by the result of validator and generate a cleaned dataset