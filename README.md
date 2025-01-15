# Pretty JAX, Mighty Triton (WIP)

Transformers written from first principle in JAX/Torch.Func/Triton; Comparison of their training efficiency on 1GPU

# Updates:

- **15th January**: Coded up memory-efficient backward pass for pytroch implementation from first principle. This can operate on the equally big model (and same data volume) as `torch.func+autograd+jit` version. Currently, my backward implementation is 16times (sic!) slower than `torch.func+autograd+jit` one.

# How to use:
- For JAX: `python train_gpt2_jax.py`
- For torch.func+autograd+jit: `python train_gpt2_triton.py torchfunc_jit`
- For Triton (WIP):  `python train_gpt2_triton.py triton`

# Files
- **train_aiayn_jax.py**: (almost) replicates the results from AIAYN paper (signle GPU only). Exploration into how feasible (& easy) it is to write very efficient code in JAX for GPU
- **train_gpt2_jax.py**: replicates the setup from GPT2 paper, but it trains on FineWeb-Edu, and evaluates on HellaSwag
- **model_jax.py**: AIAYN/GPT2 architectures written in JAX from first principle
- **model_torch_func.py**: GPT2 architecture written in funcational PyTorch (i.e. torch.func) from first principle
- **model_triton.py**: (WIP) GPT2 architecture written in Triton from first principle. Current status: implemeted memory-efficient backward passes
- **loss_and_optimizer_jax.py**: loss&optimizer logic for AIAYN/GPT2 written in JAX
- **loss_and_optimizer_triton.py**: loss&optimizer logic for GPT2 written in Torch.Func/Triton
- **tokenizer.py**: BPE tokenizer written from scrach
