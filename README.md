# Pretty JAX, Mighty Triton

Transformers written from first principle in JAX/Torch.Func(jitted)/Triton; Comparison of their training efficiency on 1GPU.

1. Here is the comparison of the GPT2 (113M) training on the FineWeb-Edu dataset using a single `L40S` GPU node.
The below graph plots validation loss against global step. One can see that the results are quite consistent across the three different implementations (JAX/Torch.Func/Triton).
<img width="1320" height="622" alt="Screenshot 2025-07-16 at 01 56 32" src="https://github.com/user-attachments/assets/1e0a15a0-9c51-4748-b90b-eeeee73a1e1b" />

2. The next graph plots validation loss against relative wall time (up to 60 hours of training). You can see that the JAX implementation is the fastest (`10.75t/s`), followed by ~~Triton (`7.98it/s`) and Torch.Func (`7.50it/s`)~~. Although Triton version includes the efficient kernels implementation (e.g., FlashAttention), the linking code between the kernels remains inefficient. See the other project of mine for more efficient E2E implementation. <br/>
$${\color{red}UPDATE \space (29th \space July)}$$: **There is a performance bug affecting Triton and Torch.Func versions. The bug is in how heads projection is applied before softmax attention. I fixed it in another repository of mine, but might not fix it here due to lack of time. The new time measurements are: Triton (`11.5it/s`) and Torch.Func (`10.8it/s`), which makes <ins>the Triton implementention to be the fastest.</ins>**
<img width="1329" height="621" alt="Screenshot 2025-07-16 at 02 00 20" src="https://github.com/user-attachments/assets/7b6478fb-4e4b-4ae5-a367-faf6f6e72e15" />

4. Finally, when it comes to the memory usage, **Triton implementation uses the peak GPU memory of 5.8GB, while Torch.Func's peak memory is 18.8GB**. However, some of that difference might be attributed to the fact that `torch.compile`, which Torch.Func uses, likely allocates the fixed space to the temporary inputs/output of the operations (this behaviour can be adjusted). I don't discuss JAX here (one can still make meaningful comparison despite its specific memory allocation pattern). 

# How to use:
- For JAX: `python train_gpt2_jax.py`
- For Torch.Func+autograd+jit: `python train_gpt2_triton.py torchfunc_jit`
- For Triton:  `python train_gpt2_triton.py triton`

There are some additional flags available:
- `--test` tests how close the Triton implementation of the forward and backward passes is to the Torch.Func's one
- `--profile` profiles the forward and backward passes for Triton
- `--eval_only` runs evaluation only (not available for JAX right now)
- `--from_checkpoint` loads the model from the checkpoint (not available for JAX right now)

# Updates:
- **16th July 2025**:
  - Tuned the Triton kernel hyperparameters for `L40S` GPU, and updated the README with the final plots.
- **2nd July 2025**:
  - Fixed some major bugs (e.g., the bug in `t_layernorm_bkwd2_x_k` was causing gradient explosion in longer runs).
  - Added bunch of tests to `tests/` directory. This consists of the speed tests (prefixed with `test_t_speed_` phrase) and the accuracy tests.
  - The results of 1day runs on 1GPU are consistent across three different ML frameworks.
- **20th February 2025**:
  - Added Triton Kernels for remaining ops (matmul, FlashAttentionv1), and improved the existing kernels (Cross Entropy Loss).
  - These kernels bring speed up to `3.15it/s` for `triton` (vs `4.06it/s` `torch.func+autograd+jit`). The memory consumption for `triton` is `7GB`, while `torch.func+autograd+jit` uses `21GB` of memory.
  - There are some numerical differences between `triton` and `torch.func+autograd+jit` versions, but they are rather small, and loss curves overlap very well.
- **31st January 2025**:
  - Fixes to the (reverse) VJP backward pass for PyTorch, which bring speed to half of `torch.func+autograd+jit` one (`2it/s` vs `4it/s`).
  - First version of Triton kernels for basic ops. They bring speed up to `2.5it/s`, and memory usage is significantly improved (I will produce the datapoints later). The kernels are available in `model_triton.py`, but they are not yet plugged in the train script of the main branch.
- **15th January 2025**:
  - Coded up memory-efficient backward pass for the PyTorch implementation from first principle. This can operate on the equally big model (and same data volume) as `torch.func+autograd+jit` version. Currently, my backward implementation is 16times (sic!) slower than `torch.func+autograd+jit` one.

# Files
- **train_aiayn_jax.py**: (almost) replicates the results from AIAYN paper (single GPU only). Exploration into how feasible (& easy) it is to write very efficient code in JAX for GPU
- **train_gpt2_jax.py**: replicates the setup from GPT2 paper in JAX, but it trains on FineWeb-Edu, and evaluates on HellaSwag
- **train_gpt2_triton.py**: replicates the setup from GPT2 paper in Torch.Func/Triton, but it trains on FineWeb-Edu, and evaluates on HellaSwag
- **model_jax.py**: AIAYN/GPT2 architectures written in JAX from first principle
- **model_torch_func.py**: GPT2 architecture written in functional PyTorch (i.e. Torch.func) from first principle
- **model_triton.py**: (WIP) GPT2 architecture written in Triton from first principle. Current status: implemented memory-efficient backward passes
- **loss_and_optimizer_jax.py**: loss&optimizer logic for AIAYN/GPT2 written in JAX
- **loss_and_optimizer_triton.py**: loss&optimizer logic for GPT2 written in Torch.Func/Triton
- **tokenizer.py**: BPE tokenizer written from scratch
- **tests/ directory**: the accuracy and speed tests. The speed tests are prefixed with `test_t_speed_` phrase
