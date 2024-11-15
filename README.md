# Exploration of Transformer architecture in JAX from first principle for GPU

- **train_aiayn.ipynb**: (almost) replicates the results from AIAYN paper (signle GPU only). Exploration into how feasible (& easy) it is to write very efficient code in JAX for GPU
- **train_gpt2.ipynb**: replicates the setup from GPT2 paper, but it trains on FineWeb-Edu, and evaluates on HellaSwag
- **model.py**: AIAYN/GPT2 architectures written from first principle
- **tokenizer.py**: BPE tokenizer writtern from scrach
