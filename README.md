# Pretty JAX, Mighty Triton

Transformers written from first principle in JAX/Torch.Func/Triton; Comparison of their training efficiency on 1GPU.

1. Here is the comparison of a day-long GPT2 training on a single `A10G GPU` (24GB GPU, which is not really intended to be used for training).
The below graph plots validation loss against global step. One can see that the results are quite consistent across the implementations in three different frameworks (JAX/Torch.Func/Triton).
![Screenshot 2025-07-02 at 06 04 22](https://github.com/user-attachments/assets/571871c7-d189-4a5d-8da6-8da77ad205d9)

2. The next graph plots validation loss against relative wall time. You can see that the JaX implementaion is the fastest (`4.7it/s`), followed by Torch.Func (`4.06it/s`) and Triton (`3.15it/s`). <ins>Triton implementation requires more work.</ins>
![Screenshot 2025-07-02 at 06 05 41](https://github.com/user-attachments/assets/16d3c2a8-406c-4733-9395-2a0aca44b64a)

3. Finally, when it comes to the memory usage, **Triton implementation uses the peak GPU memory of 5.8GB, while Torch.Func's peak memory is 18.8GB**. I don't discuss JaX here (one can still make meaningful comparison despite its specific memory allocation). 

# How to use:
- For JAX: `python train_gpt2_jax.py`
- For torch.func+autograd+jit: `python train_gpt2_triton.py torchfunc_jit`
- For Triton:  `python train_gpt2_triton.py triton`

There are additional flags `--test` and `--profile` available. The former tests how close the Triton implementation of the forward and backward is to the Torch.Func's one. The latter profiles the forward and backward passes for Triton.

# Updates:
- **2nd July 2025**:
  - Fixed some major bugs (e.g., the bug in `t_layernorm_bkwd2_x_k` was causing gradient explosion in longer runs).
  - Added bunch of tests to `tests/` directory. This consists of the speed tests (prefixed with `test_t_speed_` phrase) and the accuracy tests.
  - The results of 1day runs on 1GPU are consistent across three different ML frameworks.
- **20th February 2025**:
  - Added Triton Kernels for remaining ops (matmul, FlashAttentionv1), and improved the existing kernels (Cross Entropy Loss).
  - These kernels bring speed up to `3.15it/s` for `triton` (vs `4.06it/s` `torch.func+autograd+jit`). The memory consumption for `triton` is `7GB`, while `torch.func+autograd+jit` uses `21GB` of memory.
  - There are some numerical differences between `triton` and `torch.func+autograd+jit` versions, but they are rather small, and loss curves overlap very well.
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
- **tests/ directory**: the accuracy and speed tests. The speed tests are prefixed with `test_t_speed_` phrase
