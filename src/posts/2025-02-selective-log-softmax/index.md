---
title: Reducing VRAM Footprint in PPO and GRPO Using Selective Log-Softmax
subtitle: Slash VRAM usage by half when computing log probs by selectively applying log-softmax only to tokens of interest
date: 2025-02-06T00:00:00-08:00
blurb: Reduce VRAM usage by half when computing log probabilities by selectively applying log-softmax to only the necessary tokens. Perfect for many RLHF post-training algorithms (such as PPO and GRPO) where typically only one token's log probability is needed from the entire vocabulary at each sequence position.
tags: ["post", "grpo", "ppo", "logprobs", "logits", "log-softmax", "log_softmax", "logsumexp", "log-probabilities"]
---

When training language models, we often need to convert logits (raw model outputs) into log probabilities. The standard approach uses `log_softmax` which requires computing probabilities for every token in the vocabulary at every position in the sequence. For large vocabulary sizes, this can consume significant VRAM[^vram]. This is the code you might see:

[^vram]: VRAM is a GPU's fast, onboard memory. VRAM is the main bottleneck to training larger models on a fixed number of GPUs. It is also a bottleneck on batch size, which affects training throughput and stability.

```python
def naive_selective_log_softmax(logits, index):
    logprobs = logits.log_softmax(dim=-1)  # shape: (batch_size, seq_len, vocab_size)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
```

For example, with a modest vocabulary size of 32768, sequence length of 1024, and batch size of 16, computing `log_softmax` naively can consume **2.1GB** of VRAM! And that is in addition to the 2.1GB required to hold the logits in the first place. **However, in many cases, we only need the log probabilities for specific tokens** - usually the ones that were actually generated or appear in the training data.

This optimization is especially valuable for reinforcement learning algorithms like PPO and GRPO that fine-tune language models. These methods only require log probabilities for the tokens that were actually generated in the model's output, not for every possible token in the vocabulary. Additionally, for a typical implementation of one of these algorithms, **peak VRAM consumption occurs from materializing these log probabilities!** So optimizing[^optimized] this operation can directly allow us to train with a larger batch size.

[^optimized]: [To jump to the optimized solution, click here](#efficient-solution).

Let's remind ourselves what `log_softmax` is actually computing for every input logit $x_i$:

$$
\log \text{softmax}(x_i) = \log\left(\frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}\right) \\
= \log(e^{x_i}) - \log\left(\sum_{j=1}^n e^{x_j}\right) \\
= x_i - \log \sum_{j=1}^n e^{x_j}
$$

Essentially it is just taking every individual logit and subtracting the [`logsumexp`](https://pytorch.org/docs/stable/generated/torch.logsumexp.html#torch-logsumexp) over the full logit distribution.

 We can optimize this by:

1. Computing the `logsumexp` values over the full logit distribution
2. Gathering just the logits for the tokens we care about
3. Subtracting the `logsumexp` values from our gathered logits to get the final log probabilities

Here's what this looks like in code:
```python
def selective_log_softmax_take1(logits, index):
    logsumexp_values = torch.logsumexp(logits, dim=-1)  # shape: (batch_size, seq_len)
    token_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)  # shape: (batch_size, seq_len)
    token_logprobs = token_logits - logsumexp_values  # shape: (batch_size, seq_len)
    return token_logprobs
```

On the surface, it looks like this should decrease the memory requirements of the selective log-softmax operation -- we are now only outputting tensors of size `batch_size * sequence_length` rather than `batch_size * sequence_length * vocab_size`. However, there is a catch. Internally, `torch.logsumexp()` allocates a tensor of size `batch_size * sequence_length * vocab_size` in order to exponentiate the logits. So, unfortunately, our peak memory consumption has not decreased at all.

What can we do to improve this situation?

Well, we could just compute the logsumexp values one-by-one for each sequence in the batch. That would mean that `torch.logsumexp()` only materializes a `sequence_length * vocab_size` tensor internally.

```python
def selective_log_softmax_take2(logits, index):
    logsumexp_values = torch.stack([torch.logsumexp(l, dim=-1) for l in logits])
    token_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logits - logsumexp_values
    return token_logprobs
```

This approach should effectively reduce peak memory usage by only allocating tensors that are proportional to `batch_size * sequence_length` and `sequence_length * vocab_size` rather than `batch_size * sequence_length * vocab_size`.

Lets run a benchmark to see if we are correct. We'll also include the following ablation that simply computes `log_softmax` in a loop over the batch dimension.
```python
def selective_log_softmax_ablation1(logits, index):
    token_logprobs = []
    for logits_row, index_row in zip(logits, index):
        logprobs_row = logits_row.log_softmax(dim=-1)  # (seq_len, vocab_size)
        token_logprobs_row = torch.gather(logprobs_row, dim=-1, index=index_row.unsqueeze(-1)).squeeze(-1)
        token_logprobs.append(token_logprobs_row)
    return torch.stack(token_logprobs)
```

Here is the benchmark script:
```python
import time
import torch

def measure_memory_and_time(func, logits, index, n_runs=100):
    torch.cuda.reset_peak_memory_stats()
    result = func(logits, index)
    mem_peak = torch.cuda.max_memory_allocated()
    start_time = time.perf_counter()
    for _ in range(n_runs):
        func(logits, index)
    avg_time = (time.perf_counter() - start_time) / n_runs
    return result, avg_time, mem_peak

# Simulated data
torch.manual_seed(42)
vocab_size = 32768
seq_len = 1024
batch_size = 16

device = "cuda" if torch.cuda.is_available() else "cpu"
logits = torch.randn(batch_size, seq_len, vocab_size, device=device, dtype=torch.float32)
index = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
logit_mem = torch.cuda.max_memory_allocated()

# Run all methods
naive_result, naive_time, naive_mem = measure_memory_and_time(naive_selective_log_softmax, logits, index)
take1_result, take1_time, take1_mem = measure_memory_and_time(selective_log_softmax_take1, logits, index)
take2_result, take2_time, take2_mem = measure_memory_and_time(selective_log_softmax_take2, logits, index)
ablation1_result, ablation1_time, ablation1_mem = measure_memory_and_time(selective_log_softmax_ablation1, logits, index)

# Check equivalence
print("Logits Dtype:", logits.dtype)
print("Max absolute difference (naive and take1):", (naive_result - take1_result).abs().max().item())
print("Max absolute difference (naive and take2):", (naive_result - take2_result).abs().max().item())
print("Max absolute difference (naive and ablation1):", (naive_result - ablation1_result).abs().max().item())
print("Memory consumed by logits: {:.2f} MB".format(logit_mem / 1e6))
print("Naive method time:      {:.6f} sec, Memory peak: {:.2f} MB".format(naive_time, naive_mem / 1e6))
print("Take1 method time:      {:.6f} sec, Memory peak: {:.2f} MB".format(take1_time, take1_mem / 1e6))
print("Take2 method time:      {:.6f} sec, Memory peak: {:.2f} MB".format(take2_time, take2_mem / 1e6))
print("Ablation1 method time:  {:.6f} sec, Memory peak: {:.2f} MB".format(ablation1_time, ablation1_mem / 1e6))
```

Running this benchmark[^benchmark] script with logits stored in `float32` gives the following output:
```text
Logits Dtype: torch.float32
Max absolute difference (naive and take1): 1.9073486328125e-06
Max absolute difference (naive and take2): 1.9073486328125e-06
Max absolute difference (naive and ablation1): 0.0
Memory consumed by logits: 2147.61 MB
Naive method time:      0.000018 sec, Memory peak: 4295.16 MB
Take1 method time:      0.000965 sec, Memory peak: 4295.29 MB
Take2 method time:      0.012608 sec, Memory peak: 2282.03 MB
Ablation1 method time:  0.004153 sec, Memory peak: 2416.31 MB
```

[^benchmark]: {-} Memory usage vs. vocabulary size. `take1` is obscured by `naive` because they have the same memory requirements. ![Memory usage vs vocabulary size for different selective log softmax implementations](/assets/img/selective-logsoftmax-memory-vocab.png)

In this benchmark setting, **peak VRAM usage for this operation was reduced by 47% (from 4295MB to 2282MB)** while maintaining numerical stability. And **most of the memory consumed now is due to the size of the input logits (2147MB)**. The proposed method is notably slower than the naive method, although, in practice (for LLM post-training), the speed of this operation is not very consequential.

### Ablation Analysis
One might note that the ablation method also only allocates tensors proportional to `sequence_length * vocab_size`, so why does it consume more memory than `selective_log_softmax_take2`? This is because of the gradient formulas for `log_softmax()` and `logsumexp()` require different intermediate values to be stored for the backward computation.

For `log_softmax`, the gradient formula is:
$$
\frac{\partial \text{log\_softmax}(x_i)}{\partial x_j} = \delta_{ij} - \text{softmax}(x_j)
$$

For `logsumexp`, the gradient formula is:
$$
\frac{\partial \text{logsumexp}(x)}{\partial x_i} = \text{softmax}(x_i)
$$

For the backward pass through `logsumexp`, we need:
- Softmax of input: `(sequence_length, vocab_size)`

For the backward pass through `log_softmax`, we need:
- Softmax of input: `(sequence_length, vocab_size)`
- Original `log_softmax` output: `(sequence_length, vocab_size)`

So while both methods avoid allocating additional `vocab_size`-scale tensors during the forward pass, `selective_log_softmax_ablation1` needs to store the full `log_softmax` output for the backward pass, leading to higher memory usage.

### Numerical Stability
It is important to note that the `selective_log_softmax_take2` is not numerically stable when logits are cast to `bfloat16` or `float16`:
```text
Logits Dtype: torch.bfloat16
Max absolute difference (naive and take1): 0.0625
Max absolute difference (naive and take2): 0.0625  # <- this is the issue
Max absolute difference (naive and ablation1): 0.0
Memory consumed by logits: 1073.87 MB
Naive method time:      0.000018 sec, Memory peak: 2147.65 MB
Take1 method time:      0.000474 sec, Memory peak: 2147.75 MB
Take2 method time:      0.005142 sec, Memory peak: 1141.11 MB
Ablation1 method time:  0.002016 sec, Memory peak: 1208.22 MB
```

Therefore, we should use `selective_log_softmax_take2` when working with full precision (`torch.float32` and `torch.float64`) tensors, and fall back to `selective_log_softmax_ablation1` when using reduced precision (`torch.bfloat16` and `torch.float16`) tensors to maintain accuracy.

### Efficient Solution
The complete code snippet is as follows:

```python
def selective_log_softmax(logits, index):
    """Compute log softmax probabilities for selected tokens.

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.
    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])  # loop to reduce peak mem consumption
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        token_logprobs = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        token_logprobs = []
        for logits_row, index_row in zip(logits, index):  # loop to reduce peak mem consumption
            logprobs_row = logits_row.log_softmax(dim=-1)
            token_logprobs_row = torch.gather(logprobs_row, dim=-1, index=index_row.unsqueeze(-1)).squeeze(-1)
            token_logprobs.append(token_logprobs_row)
        token_logprobs = torch.stack(token_logprobs)
    return token_logprobs
```

I have contributed this optimization to several popular open-source RLHF libraries, including [huggingface/TRL](https://github.com/huggingface/trl) \[[PR 1](https://github.com/huggingface/trl/pull/2773), [PR 2](https://github.com/huggingface/trl/pull/2799)\], [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) \[[PR 3](https://github.com/OpenRLHF/OpenRLHF/pull/718)\], [Verl](https://github.com/volcengine/verl) \[[PR 4](https://github.com/volcengine/verl/pull/220)\], and [allenai/open-instruct](https://github.com/allenai/open-instruct) \[[PR 5](https://github.com/allenai/open-instruct/pull/584)\].

Here is the actual GPU memory usage on an RTX 4090 (24GB VRAM) before and after implementing selective log-softmax in TRL's `GRPOTrainer`: ![Memory usage reduction from selective log-softmax in TRL](/assets/img/trl-selective-log-softmax.png)

A 10% reduction in peak VRAM requirements is a great improvement for such a simple change!

### A note on `torch.compile`

When using `torch.compile()`, PyTorch will attempt to fuse operations and generate optimized CUDA kernels using [Triton](https://github.com/openai/triton). For our selective log-softmax implementation, this means PyTorch may be able to take the naive implementation and fuse the `log_softmax` and `gather` operations into a single kernel, potentially reducing memory consumption.

```python
@torch.compile(dynamic=True)
def compiled_selective_log_softmax(logits, index):
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
```

If we benchmark this method using `torch==2.6.0` and `triton==3.2.0`, we see these results when logits are in `float32`:
```text
Max absolute difference (naive and compiled): 9.5367431640625e-07
Compiled method time:  0.000073 sec, Memory peak: 2147.94 MB
```

And these results when logits are in `bfloat16`[^exact]:

[^exact]: Interestingly, `torch.compile` generates a kernel that maintains exact numerical equivalence for half-precision dtypes.

```text
Max absolute difference (naive and compiled): 0.0
Compiled method time:  0.000129 sec, Memory peak: 1074.04 MB
```

Very impressive! This is both faster and more memory efficient than our hand-rolled solution, while being numerically stable. And the `dynamic=True` flag means that we shouldn't need to recompile every time a new sequence length is used.

The only reason not to use this method is if you are in a setting where `torch.compile` usage is supposed to be enabled/disabled via a user-passed flag. Which is the case most open-source libraries that use `torch`. For your own projects, the compiled version is recommended!


---
Thanks to [Quentin Gallouédec](https://github.com/qgallouedec) for providing the initial benchmarking script and suggesting to pull `gather` out of the loop over `logsumexp` to improve performance.