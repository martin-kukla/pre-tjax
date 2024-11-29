# Exploration of Transformer architecture in JAX and torch.func from first principle for single GPU

- **train_aiayn_jax.ipynb**: (almost) replicates the results from AIAYN paper (signle GPU only). Exploration into how feasible (& easy) it is to write very efficient code in JAX for GPU
- **train_gpt2_jax.ipynb**: replicates the setup from GPT2 paper, but it trains on FineWeb-Edu, and evaluates on HellaSwag
- **model_jax.py**: AIAYN/GPT2 architectures written in JAX from first principle
- **model_torch_func.py**: GPT2 architecture written in funcational PyTorch (i.e. torch.func) from first principle
- **tokenizer.py**: BPE tokenizer writtern from scrach
