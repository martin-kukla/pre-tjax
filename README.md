# Pretty JAX, Mighty Triton (WIP)

Transformers written from first principle in JAX/Torch.Func/Triton; Comparison of their training efficiency on 1GPU

# How to use:
- For JAX: `python train_gpt2_jax.py`
- For torch.func+autograd+jit: `python train_gpt2_triton.py torchfunc_jit`
- For Triton (WIP):  `python train_gpt2_triton.py triton`
  
# Updates:
- **31th January 2025**:
  - Fixes to the (reverse) VJP backward pass for pytorch, which bring speed to half of `torch.func+autograd+jit` one (`2it/s` vs `4it/s`).
  - First version of Triton kernels for basic ops. They bring speed up to `2.5it/s`, and memory usage is significantly improved (I will produce the datapoints later). The kernels are availble in `model_triton.py`, but they are not yet plugged in the train script of the main branch.
- **15th January 2025**:
  - Coded up memory-efficient backward pass for pytroch implementation from first principle. This can operate on the equally big model (and same data volume) as `torch.func+autograd+jit` version. Currently, my backward implementation is 16times (sic!) slower than `torch.func+autograd+jit` one.

# Files
- **train_aiayn_jax.py**: (almost) replicates the results from AIAYN paper (signle GPU only). Exploration into how feasible (& easy) it is to write very efficient code in JAX for GPU
- **train_gpt2_jax.py**: replicates the setup from GPT2 paper in JAX, but it trains on FineWeb-Edu, and evaluates on HellaSwag
- **train_gpt2_triton.py**: replicates the setup from GPT2 paper in Torch.Func/Triton, but it trains on FineWeb-Edu, and evaluates on HellaSwag
- **model_jax.py**: AIAYN/GPT2 architectures written in JAX from first principle
- **model_torch_func.py**: GPT2 architecture written in funcational PyTorch (i.e. torch.func) from first principle
- **model_triton.py**: (WIP) GPT2 architecture written in Triton from first principle. Current status: implemeted memory-efficient backward passes
- **loss_and_optimizer_jax.py**: loss&optimizer logic for AIAYN/GPT2 written in JAX
- **loss_and_optimizer_triton.py**: loss&optimizer logic for GPT2 written in Torch.Func/Triton
- **tokenizer.py**: BPE tokenizer written from scrach
