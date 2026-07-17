---
title: Reducing VRAM Footprint in PPO and GRPO Using Selective Log-Softmax
subtitle: Slash VRAM usage by half when computing log probs by selectively applying log-softmax only to tokens of interest
date: 2025-02-06T00:00:00-08:00
blurb: Reduce VRAM usage by half when computing log probabilities by selectively applying log-softmax to only the necessary tokens.
tags: ["post", "grpo", "ppo", "logprobs", "logits", "log-softmax", "log_softmax", "logsumexp", "log-probabilities"]
math: true
code: true
---

When training language models, we often need to convert logits (raw model outputs) into log probabilities. The standard approach uses `log_softmax`, which computes a log probability for every token in the vocabulary at every position in the sequence:

```python
def naive_selective_log_softmax(logits, index):
    logprobs = logits.log_softmax(dim=-1)  # (batch_size, seq_len, vocab_size)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
```

For a vocabulary size of 32768, sequence length of 1024, and batch size of 16, the full `log_softmax` output consumes another **2.1GB** of VRAM[^vram] on top of the 2.1GB already occupied by the logits.

[^vram]: VRAM is a GPU's fast, onboard memory. VRAM is the main bottleneck to training larger models on a fixed number of GPUs. It is also a bottleneck on batch size, which affects training throughput and stability.

However, in many workloads, we only need the log probability of one token at each position. PPO and GRPO are good examples: they need the log probabilities of the tokens that were actually generated, not every possible token in the vocabulary. In typical implementations of these algorithms, materializing the full log-probability tensor can determine peak VRAM usage.

Let's remind ourselves what `log_softmax` computes for each input logit (x_i):

\[
\log \text{softmax}(x_i) = \log\left(\frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}\right) \\
= x_i - \log \sum_{j=1}^n e^{x_j}
\]

If we only care about one selected token (i), then the operation we actually want is:

\[
\text{output} = x_i - \text{logsumexp}(x)
\]

This is an important shift in perspective. Selective log-softmax does not need to be implemented as a full `log_softmax` followed by a gather. It is a row-wise reduction plus one selected logit. That observation tells us what an ideal GPU implementation should do:

1. Stream over the vocabulary once.
2. Compute a numerically stable log-sum-exp for each row.
3. Load the selected logit.
4. Write one scalar instead of a full vocabulary-sized row.

### A Dedicated Kernel

I initially approached this as a PyTorch memory optimization. The same operation can also be implemented directly as a CUDA kernel. I wrote [one implementation using CuTe DSL](https://www.tylerromero.com/greyhound/kernels/selective_log_softmax/) as part of Greyhound. It is useful here because it makes the reduction strategy and memory traffic explicit.

The kernel flattens every leading dimension into rows and assigns one CUDA thread block to each row. In pseudocode:

```text
# One thread block per row
row = block_id
m, s = -infinity, 0.0

# Each thread scans a strided slice of the vocabulary
for col = thread_id; col < vocab_size; col += block_size:
    x = fp32(logits[row, col])
    new_m = max(m, x)
    s = s * exp(m - new_m) + exp(x - new_m)
    m = new_m

# Warp reductions followed by one cross-warp reduction
m, s = block_reduce(merge, (m, s))

if thread_id == 0:
    logsumexp = m + log(s)
    out[row] = logits[row, index[row]] - logsumexp
```

The `block_reduce` step uses the online, numerically stable form of log-sum-exp.<label for="sn-online-softmax" class="margin-toggle sidenote-number"></label><input type="checkbox" id="sn-online-softmax" class="margin-toggle" aria-label="Toggle sidenote" /><span class="sidenote">This is the online softmax recurrence described by Milakov and Gimelshein in <a href="https://arxiv.org/abs/1805.02867"><em>Online normalizer calculation for softmax</em></a> (2018). Their elementwise update is the singleton case of the pairwise merge used here for parallel reduction.</span> Suppose one partial reduction has maximum \(m\) and normalized exponential sum \(s = \sum_i e^{x_i-m}\), while another has state \((m', s')\). We can merge them using:

\[
m_{new} = \max(m, m')
\]

\[
s_{new} = s e^{m-m_{new}} + s' e^{m'-m_{new}}
\]

This implementation performs the accumulation in FP32 even when the input is `float16` or `bfloat16`. It supports contiguous `float16`, `bfloat16`, and `float32` logits with `int32` or `int64` indices. Because it writes only the selected result, it does not materialize the full log-probability tensor or an intermediate exponential tensor.

`torch.compile` can eliminate the full log-probability allocation, but it does not currently recover this exact reduction strategy. In the PyTorch 2.10 generated kernel I inspected, Inductor disabled its online-softmax path after splitting the reduction: the emitted Triton kernel made one pass to find the row maximum and a second pass to compute the exponential sum. The dedicated kernel instead maintains `(max, sum)` together in one streaming pass. This is a limitation of the current lowering, not something fundamentally impossible for a compiler to generate.

### Benchmarks

I benchmarked three implementations over multiple sequence lengths, vocabulary sizes, and GPUs. The [complete benchmark sweep is available with the kernel implementation](https://www.tylerromero.com/greyhound/kernels/selective_log_softmax/#benchmarks).

- Eager PyTorch: `log_softmax` followed by `gather`
- `torch.compile`: the same PyTorch expression compiled by Inductor
- CuTe DSL: the dedicated reduction kernel described above

Here is one representative point using `bfloat16` logits with batch size 8, sequence length 4096, and vocabulary size 128256:

| GPU | Eager | Compiled | CuTe DSL | Speedup vs. compiled |
| --- | ---: | ---: | ---: | ---: |
| RTX 4090 | 23.564 ms | 12.545 ms | 8.819 ms | 1.42x faster |
| H100 80GB HBM3 | 10.619 ms | 7.368 ms | 3.123 ms | 2.36x faster |
| B200 | 5.105 ms | 6.196 ms | 2.301 ms | 2.69x faster |

At this shape, the input logits occupy roughly 8.0 GiB. Eager PyTorch uses about twice that much peak allocated memory because it materializes another vocabulary-sized tensor. Both `torch.compile` and the CuTe kernel remain close to the size of the input logits. The CuTe kernel is 1.4--2.7x faster at this particular benchmark point across the three GPUs.

### Practical Tradeoffs

The CuTe implementation is currently forward-only. It can be used for scoring, evaluation, reward-model traces, and other paths that do not differentiate through the selected log probabilities, but it is not yet a drop-in replacement inside PPO or GRPO training.

That leaves three practical cases:

- **Forward-only workloads:** a dedicated kernel can avoid relying on compiler fusion.
- **Training with autograd:** the compiled PyTorch expression below is the most practical option here.
- **Eager-only environments:** the looped PyTorch workaround in the appendix reduces memory at the cost of additional kernel launches.

For the training case, `torch.compile` can fuse the naive `log_softmax` and `gather` expression into a memory-efficient generated kernel:

```python
@torch.compile(dynamic=True)
def compiled_selective_log_softmax(logits, index):
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
```

The benchmarks above confirm that the compiled expression avoids the full log-probability allocation. It also supports autograd, while the dedicated kernel described above does not yet have a backward implementation. The `dynamic=True` flag allows sequence length to vary without specializing a new graph for every value, although changes to the vocabulary dimension can still require new generated code.

### Impact in PPO and GRPO Training

Before the dedicated kernel existed, I contributed the eager PyTorch workaround from the appendix to several popular open-source RLHF libraries, including [huggingface/TRL](https://github.com/huggingface/trl) ([PR 1](https://github.com/huggingface/trl/pull/2773), [PR 2](https://github.com/huggingface/trl/pull/2799)), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) ([PR 3](https://github.com/OpenRLHF/OpenRLHF/pull/718)), [Verl](https://github.com/volcengine/verl) ([PR 4](https://github.com/volcengine/verl/pull/220)), and [allenai/open-instruct](https://github.com/allenai/open-instruct) ([PR 5](https://github.com/allenai/open-instruct/pull/584)). It was also incorporated into [PrimeIntellect-ai/prime-rl](https://github.com/PrimeIntellect-ai/prime-rl/blob/a092a54029549d600d32d3b3f123ea3607498604/src/zeroband/training/loss.py#L47) for training the [INTELLECT-2](https://storage.googleapis.com/public-technical-paper/INTELLECT_2_Technical_Report.pdf) model.

Here is the actual GPU memory usage on an RTX 4090 before and after implementing selective log-softmax in TRL's `GRPOTrainer`:

![Memory usage reduction from selective log-softmax in TRL](/assets/img/trl-selective-log-softmax.png)

A 10% reduction in total peak VRAM is a meaningful improvement for changing one operation. It can translate directly into a larger batch size or longer sequence length.

## Appendix: The Eager PyTorch Experiments

The rest of this post preserves the thought process that led me to the kernel. These implementations are useful for understanding where the memory goes, but they are workarounds rather than ideal GPU programs.

### Take 1: Decompose the Operation

The algebra suggests computing `logsumexp`, gathering the selected logits, and subtracting:

```python
def selective_log_softmax_take1(logits, index):
    logsumexp_values = torch.logsumexp(logits, dim=-1)
    token_logits = torch.gather(
        logits, dim=-1, index=index.unsqueeze(-1)
    ).squeeze(-1)
    return token_logits - logsumexp_values
```

On the surface, this looks like it should solve the problem because the output has shape `(batch_size, sequence_length)` instead of `(batch_size, sequence_length, vocab_size)`. However, eager `torch.logsumexp()` internally materializes a vocabulary-sized exponential tensor. Peak memory therefore remains essentially unchanged.

### Take 2: Limit the Temporary Size

Well, we could compute `logsumexp` one batch row at a time. The largest temporary would then have shape `(sequence_length, vocab_size)` instead of `(batch_size, sequence_length, vocab_size)`:

```python
def selective_log_softmax_take2(logits, index):
    logsumexp_values = torch.stack(
        [torch.logsumexp(logits_row, dim=-1) for logits_row in logits]
    )
    token_logits = torch.gather(
        logits, dim=-1, index=index.unsqueeze(-1)
    ).squeeze(-1)
    return token_logits - logsumexp_values
```

For comparison, we can also loop over `log_softmax` itself:

```python
def selective_log_softmax_ablation(logits, index):
    token_logprobs = []
    for logits_row, index_row in zip(logits, index):
        logprobs_row = logits_row.log_softmax(dim=-1)
        token_logprobs_row = torch.gather(
            logprobs_row, dim=-1, index=index_row.unsqueeze(-1)
        ).squeeze(-1)
        token_logprobs.append(token_logprobs_row)
    return torch.stack(token_logprobs)
```

Both approaches limit the large temporary to one batch row, but they require many separate kernel launches. That is the fundamental weakness of trying to express this operation as a Python loop over eager PyTorch primitives.

### Reproducing the Eager Benchmark

Here is the benchmark helper. The CUDA synchronizations are important because GPU operations are asynchronous:

```python
import time
import torch


def measure_memory_and_time(func, logits, index, n_runs=100):
    # Warm up one-time CUDA work before measuring.
    func(logits, index)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    result = func(logits, index)
    torch.cuda.synchronize()
    mem_peak = torch.cuda.max_memory_allocated()

    start_time = time.perf_counter()
    for _ in range(n_runs):
        func(logits, index)
    torch.cuda.synchronize()
    avg_time = (time.perf_counter() - start_time) / n_runs
    return result, avg_time, mem_peak


torch.manual_seed(42)
vocab_size = 32768
seq_len = 1024
batch_size = 16

logits = torch.randn(
    batch_size,
    seq_len,
    vocab_size,
    device="cuda",
    dtype=torch.float32,
)
index = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")

funcs = {
    "Naive": naive_selective_log_softmax,
    "Take1": selective_log_softmax_take1,
    "Take2": selective_log_softmax_take2,
    "Ablation": selective_log_softmax_ablation,
}
results = {name: measure_memory_and_time(fn, logits, index) for name, fn in funcs.items()}
```

Running this benchmark[^benchmark] with `float32` logits gives:

```text
Logits Dtype: torch.float32
Max abs diff (Naive vs Take1): 1.9073486328125e-06
Max abs diff (Naive vs Take2): 1.9073486328125e-06
Max abs diff (Naive vs Ablation): 0.0
Logits memory: 2147.48 MB
Naive      time: 0.004702 sec, peak: 4295.23 MB
Take1      time: 0.013805 sec, peak: 4295.43 MB
Take2      time: 0.013911 sec, peak: 2282.16 MB
Ablation   time: 0.004778 sec, peak: 2282.16 MB
```

[^benchmark]: These results were rerun on July 17, 2026, using an RTX 4090 and `torch==2.10.0+cu128`. The original version of this post did not synchronize CUDA before reading the timer, which understated execution time.

Peak VRAM drops by 47%, from 4295MB to 2282MB, and almost all of the remaining memory is the 2147MB input. But the looped `logsumexp` implementation is about 3x slower than the naive operation. The looped `log_softmax` ablation reaches the same memory footprint and is much faster.

### Half-Precision Numerical Behavior

The decomposed `logsumexp` implementation also does not exactly match PyTorch's `log_softmax` result for `bfloat16` and `float16` inputs:

```text
Logits Dtype: torch.bfloat16
Max abs diff (Naive vs Take1): 0.0625
Max abs diff (Naive vs Take2): 0.0625
Max abs diff (Naive vs Ablation): 0.0
Logits memory: 1073.74 MB
Naive      time: 0.002378 sec, peak: 2147.68 MB
Take1      time: 0.006921 sec, peak: 2147.78 MB
Take2      time: 0.005678 sec, peak: 1141.15 MB
Ablation   time: 0.002429 sec, peak: 1141.15 MB
```

This is a rounding difference rather than the classic overflow problem that stable log-sum-exp avoids, but it matters when exact agreement with PyTorch is expected. The eager workaround therefore uses looped `logsumexp` for full-precision inputs and looped `log_softmax` for reduced-precision inputs:

```python
def selective_log_softmax(logits, index):
    """Compute log-softmax probabilities for selected tokens."""
    if logits.dtype in (torch.float32, torch.float64):
        lse = torch.stack(
            [torch.logsumexp(logits_row, dim=-1) for logits_row in logits]
        )
        selected = torch.gather(
            logits, dim=-1, index=index.unsqueeze(-1)
        ).squeeze(-1)
        return selected - lse

    token_logprobs = []
    for logits_row, index_row in zip(logits, index):
        logprobs_row = logits_row.log_softmax(dim=-1)
        token_logprobs_row = torch.gather(
            logprobs_row, dim=-1, index=index_row.unsqueeze(-1)
        ).squeeze(-1)
        token_logprobs.append(token_logprobs_row)
    return torch.stack(token_logprobs)
```

This workaround is still useful when `torch.compile` and custom kernels are unavailable. But after following the operation all the way down, the right abstraction is clear: selective log-softmax wants to be a single streaming reduction kernel.

<hr class="section-divider">

Thanks to [Quentin Gallouédec](https://github.com/qgallouedec) for providing the initial benchmarking script and suggesting that I pull `gather` out of the loop over `logsumexp` to improve performance.
