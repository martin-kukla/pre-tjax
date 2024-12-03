# Pretty JAX, Mighty Triton

Transformers written from first principle in JAX/Torch.Func/Triton; Comparison of their training efficiency on 1GPU

- **train_aiayn_jax.ipynb**: (almost) replicates the results from AIAYN paper (signle GPU only). Exploration into how feasible (& easy) it is to write very efficient code in JAX for GPU
- **train_gpt2_jax.ipynb**: replicates the setup from GPT2 paper, but it trains on FineWeb-Edu, and evaluates on HellaSwag
- **model_jax.py**: AIAYN/GPT2 architectures written in JAX from first principle
- **model_torch_func.py**: GPT2 architecture written in funcational PyTorch (i.e. torch.func) from first principle
- **model_triton.py**: (WIP) GPT2 architecture written in Triton from first principle. Current status: Jacobians implemented for the basic blocks
- **tokenizer.py**: BPE tokenizer writtern from scrach
