---
title: Selective log-softmax trick to reduce VRAM usage for PPO and GRPO
subtitle: Reduce VRAM usage by half when computing log probs by selectively applying log-softmax only to tokens of interest
date: 2025-02-06T00:00:00-08:00
blurb: Cut VRAM usage in half when computing log probabilities by selectively applying log-softmax only to tokens of interest. Perfect for PPO and GRPO training where you only need log probabilities for a small subset of tokens.
tags: ["post", "grpo", "ppo", "logprobs", "logits", "log-softmax", "log_softmax", "logsumexp", "log-probabilities"]
---

When training language models, we often need to convert logits (raw model outputs) into log probabilities. The standard approach uses `log_softmax` which requires computing probabilities for every token in the vocabulary at every position in the sequence. For large vocabulary sizes, this can consume significant VRAM.

For example, with a modest vocabulary size of 32768, sequence length of 1024, and batch size of 16, computing `log_softmax` naively can consume **2.1GB** of VRAM! And that is in addition to the 2.1GB required to hold the logits in the first place. **However, in many cases, we only need the log probabilities for specific tokens** - usually the ones that were actually generated or appear in the training data.

This is particularly relevant when using reinforcement learning techniques like PPO and GRPO to fine-tune language models with RLHF. These methods only require log probabilities for the tokens that were actually generated in the model's output, not for every possible token in the vocabulary.

Let's remind ourselves what `log_softmax` is actually computing for every input logit $x_i$:

$$
\log \text{softmax}(x_i) = \log\left(\frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}\right) \\
= \log(e^{x_i}) - \log\left(\sum_{j=1}^n e^{x_j}\right) \\
= x_i - \log \sum_{j=1}^n e^{x_j}
$$

Essentially it is just taking every individual logit and subtracting the `logsumexp` over the full logit distribution.

 We can optimize this by:

1. First gathering just the logits for the tokens we care about
2. Computing the softmax denominator (logsumexp) over the full logit distribution
3. Subtracting the denominator from our gathered logits to get the final log probabilities

Here's what this looks like in code:
```python
def selective_log_softmax(logits, input_ids):
    # Gather logits just for the tokens we care about
    token_logits = torch.gather(logits, dim=-1,
    index=input_ids.unsqueeze(-1)).squeeze(-1)

    # Compute logsumexp denominator for each sequence in the batch (loop to reduce memory peak)
    logsumexp_values = torch.stack([torch.logsumexp(l, dim=-1) for l in logits])

    # Subtract to get final log probs
    token_log_probs = token_logits - logsumexp_values
    return token_log_probs
```

This approach reduces the peak memory usage by only allocating tensors that are proportional to (batch_size \* sequence length) rather than (batch_size \* sequence_length * vocab_size).

Lets benchmark this against some alternatives:
```python
import time
import torch

def naive_method(logits, input_ids):
    log_probs = logits.log_softmax(dim=-1)
    return torch.gather(log_probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

def method_1(logits, input_ids):  # compute log_softmax in a loop to reduce peak memory
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=-1, index=input_ids_row.unsqueeze(-1)).squeeze(-1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)

def method_2(logits, input_ids):  # avoid materializing unneeded log_probs to reduce peak memory
    token_logits = torch.gather(logits, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    logsumexp_values = torch.logsumexp(logits, dim=-1)
    token_log_probs = token_logits - logsumexp_values  # log_softmax(logits) = logits - log(sum(exp(logits)))
    return token_log_probs

def method_3(logits, input_ids):  # combine methods 1 and 2
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        token_logits = torch.gather(logits_row, dim=-1, index=input_ids_row.unsqueeze(-1)).squeeze(-1)
        token_log_prob = token_logits - torch.logsumexp(logits_row, dim=-1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)

def efficient_method(logits, input_ids):  # pull everything out of the loop except logsumexp
    token_logits = torch.gather(logits, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    logsumexp_values = torch.stack([torch.logsumexp(l, dim=-1) for l in logits])
    token_log_probs = token_logits - logsumexp_values
    return token_log_probs

def measure_memory_and_time(func, logits, input_ids):
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    result = func(logits, input_ids)
    end_time = time.perf_counter()
    mem_peak = torch.cuda.max_memory_allocated()
    return result, end_time - start_time, mem_peak

# Simulated data
torch.manual_seed(42)
vocab_size = 32768
seq_len = 1024
batch_size = 16

device = "cuda" if torch.cuda.is_available() else "cpu"
logits = torch.randn(batch_size, seq_len, vocab_size, device=device, dtype=torch.float32)
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
logit_mem = torch.cuda.max_memory_allocated()

# Run all methods
naive_result, naive_time, naive_mem = measure_memory_and_time(naive_method, logits, input_ids)
method1_result, method1_time, method1_mem = measure_memory_and_time(method_1, logits, input_ids)
method2_result, method2_time, method2_mem = measure_memory_and_time(method_2, logits, input_ids)
method3_result, method3_time, method3_mem = measure_memory_and_time(method_3, logits, input_ids)
efficient_result, efficient_time, efficient_mem = measure_memory_and_time(efficient_method, logits, input_ids)

# Check equivalence
print("Max absolute difference (naive and 1):", (naive_result - method1_result).abs().max().item())
print("Max absolute difference (naive and 2):", (naive_result - method2_result).abs().max().item())
print("Max absolute difference (naive and 3):", (naive_result - method3_result).abs().max().item())
print("Max absolute difference (naive and efficient):", (naive_result - efficient_result).abs().max().item())
print("Memory consumed by logits: {:.2f} MB".format(logit_mem / 1e6))
print("Naive method time:      {:.6f} sec, Memory peak: {:.2f} MB".format(naive_time, naive_mem / 1e6))
print("Method 1 time:          {:.6f} sec, Memory peak: {:.2f} MB".format(method1_time, method1_mem / 1e6))
print("Method 2 time:          {:.6f} sec, Memory peak: {:.2f} MB".format(method2_time, method2_mem / 1e6))
print("Method 3 time:          {:.6f} sec, Memory peak: {:.2f} MB".format(method3_time, method3_mem / 1e6))
print("Efficient method time:  {:.6f} sec, Memory peak: {:.2f} MB".format(efficient_time, efficient_mem / 1e6))

# Results:
# > Max absolute difference (naive and 1): 0.0
# > Max absolute difference (naive and 2): 1.9073486328125e-06
# > Max absolute difference (naive and 3): 1.9073486328125e-06
# > Max absolute difference (naive and efficient): 1.9073486328125e-06
# > Memory consumed by logits: 2147.61 MB
# > Naive method time:      0.036307 sec, Memory peak: 4295.16 MB
# > Method 1 time:          0.012156 sec, Memory peak: 2416.18 MB
# > Method 2 time:          0.134651 sec, Memory peak: 4295.43 MB
# > Method 3 time:          0.001496 sec, Memory peak: 2282.10 MB
# > Efficient method time:  0.000918 sec, Memory peak: 2282.23 MB
```

In this benchmark setting, **peak VRAM usage for this operation was reduced by 47% (from 4295MB to 2282MB)** while maintaining numerical stability. And **most of the memory consumed now is due to the size of the input logits (2147MB)**. Additionally, the proposed method is about 40x faster than the naive implementation (0.0363s vs 0.0009s). Although, in practice, the speed of this operation is not very consequential.

I have [contributed this optimization to the GRPO trainer](https://github.com/huggingface/trl/pull/2773) in the [TRL](https://github.com/huggingface/trl) library and to [the PPO, GRPO, DPO, and KTO trainers](https://github.com/OpenRLHF/OpenRLHF/pull/718) in the [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) library.

Thanks to [Quentin Gallouédec](https://github.com/qgallouedec) for providing the benchmarking script and for suggesting to pull the `gather` and element-wise subtraction operations out of the for loop in order to improve operation speed (`method_3` -> `efficient_method`).