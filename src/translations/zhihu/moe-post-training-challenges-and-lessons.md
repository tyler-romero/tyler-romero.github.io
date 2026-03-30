---
title: "Post-Training Challenges and Lessons Learned with MoE Architectures"
subtitle: "Translated from [moe架构的post-training难点与经验分享](https://zhuanlan.zhihu.com/p/2018018879109091590) by [燕雄飞的一天](https://www.zhihu.com/people/yan-xiong-fei-19)"
date: 2026-03-19T02:56:00+08:00
blurb: "Practical guide to MoE post-training: balancing auxiliary loss with performance, stabilizing RL with Routing Replay, and choosing between Expert Parallelism (EP) and Expert Tensor Parallelism (ETP)."
tags: ["translation", "mixture-of-experts", "post-training", "reinforcement-learning", "load-balancing"]
math: true
---

*Translator's note (Opus 4.6): This is an English translation of [moe架构的post-training难点与经验分享](https://zhuanlan.zhihu.com/p/2018018879109091590) by [燕雄飞的一天](https://www.zhihu.com/people/yan-xiong-fei-19), originally published on March 19, 2026 on [Zhihu (知乎)](https://www.zhihu.com). The translation preserves the author's informal, first-person voice.*

<hr class="section-divider">

Whether in open-source or top-tier commercial models, MoE architectures have become the dominant paradigm. How does post-training differ from dense models during the SFT and RL stages? What unique challenges arise, and how can we address them? This post starts from first principles and works through how to train MoE architectures.

## TL;DR

1. **Load balancing vs. model performance**: the `aux_loss` hyperparameter requires ablation experiments to tune properly.
2. **RL training instability**: Routing Replay effectively mitigates train–inference discrepancy and policy staleness.
3. **Expert parallelism**: use EP and ETP together as needed.

## Fundamentals

### How MoE Works

In 2017, Noam Shazeer[^shazeer] and colleagues at Google Brain proposed the MoE (Sparsely-Gated Mixture-of-Experts) architecture. The core idea is a **trainable gating network** that decides, for each input, which "experts" should process it. The MoE architecture stems from the concept of *conditional computation* — for different inputs, only a subset of the network is activated while the rest stays "silent" (i.e., does not participate in the computation).

The MoE architecture has remained largely unchanged since then, so the original paper's figures and formulas are still current.

![MoE architecture overview: a gating network routes inputs to a sparse subset of expert networks](/assets/img/moe-architecture-overview.jpg)

**Overall architecture:** A Mixture-of-Experts layer consists of a set of \(n\) expert networks and a gating network, where the gating network's output is a sparse \(n\)-dimensional vector. Each expert is itself a neural network with its own independent parameters, and all experts must accept the same input size and produce the same output size.

**Formal definition:** For a given input \(x\), let \(G(x)\) denote the gating network's output and \(E_i(x)\) denote the output of the \(i\)-th expert network. The MoE module's output \(y\) takes the form:

\[
y = \sum_{i=1}^{n} G(x)_i \cdot E_i(x)
\]

The gating network works as follows: retain the top-\(k\) outputs, set the rest to \(-\infty\), and after softmax the remaining weights become zero:

\[
G(x) = \text{Softmax}(\text{TopK}(x \cdot W_g))
\]

### Load Balancing

Load balancing is one of the most critical challenges in MoE training. During early training, the gating network tends to assign high weights to a handful of specific experts. If an expert gets selected more often, it receives more training, its performance improves faster than the other "idle" experts, and the gating network becomes even more inclined to select it next time. The end result: only a few experts are doing work while thousands of others never get a chance to train, becoming "zombie experts" — an enormous waste of parameter capacity.

There are currently two load-balancing strategies validated in top-tier open-source models: **aux_loss** (used in the Qwen series, among others) and **aux_loss_free** (used in the DeepSeek series).

**Auxiliary loss (aux_loss):** In 2022, Noam Shazeer (again at Google) proposed the highly influential Switch Transformer[^switch], whose load-balancing strategy remains in use today. The core idea is an auxiliary loss that ensures all experts process roughly equal numbers of tokens:

\[
\mathcal{L}_{\text{aux}} = N \cdot \sum_{i=1}^{N} f_i \cdot P_i
\]

where \(f_i\) (the *actual allocation ratio*) is the proportion of tokens in the current batch that were assigned to the \(i\)-th expert, and \(P_i\) (the *predicted probability ratio*) is the gating network's average preference for the \(i\)-th expert.

The goal — for both actual allocation and predicted probability — is for each expert to receive \(\frac{1}{N}\) of the tokens. The minimum value of this loss is therefore 1.

**Loss-free balancing (aux_loss_free):** In 2024, the DeepSeek team introduced loss-free balancing as a core component of DeepSeek-V3. It requires no auxiliary term in the loss function, and therefore does not affect model performance.

\[
g_i'(x) = g_i(x) + b_i
\]

Here, \(g_i(x)\) is the \(i\)-th expert's original gating score (before top-\(k\) selection), and \(g_i'(x)\) is the bias-adjusted score. The key insight: **top-\(k\) selection uses the biased scores, but the output computation still uses the original gating scores.** The bias \(b_i\) is dynamically updated based on load statistics from the previous batch.

## Training Challenges

### Challenge 1: Balancing Load vs. Performance

In MoE training, tuning the auxiliary loss weight is a core challenge. While increasing `aux_loss` can force a more uniform expert load distribution, its gradient interferes with the main task's learning signal — too much weight leads to a significant drop in model performance (eval loss).

**Model performance:** Experimental data shows that smaller `aux_loss` weights yield better model convergence.

![aux_loss weight vs. eval loss and load balance: lower weights improve convergence but reduce balance](/assets/img/moe-aux-loss-performance.jpg)

**Load balance:** Measured by a relative load-balance metric, the data shows that smaller `aux_loss` weights lead to more imbalanced models (some experts overloaded, others idle).

![Load balance heatmap across experts](/assets/img/moe-load-balance-heatmap.jpg)

- **Actual expert load:** the number of tokens an expert processes on the training set.
- **Theoretical balanced load:** the number of tokens each expert should process under uniform distribution (total tokens / number of experts).

**Balancing the two, a weight of 0.001 is a reasonable sweet spot.**

### Challenge 2: RL Training Instability

RL training commonly faces the "reward–gradient mismatch" problem (sequence-level rewards vs. token-level updates). As shown by Zheng et al.[^zheng], this fundamentally stems from two sources of discrepancy:

1. **Train–inference discrepancy:** Numerical errors caused by differing infrastructure between training and inference (different compute kernels, batch-invariant kernels disabled at inference for throughput, FP8 inference vs. BF16 training).
2. **Policy staleness:** The gap between the rollout policy and the current optimization policy, typically caused by off-policy updates (splitting a large batch into mini-batches for multiple updates).

For MoE models, these small discrepancies are **dramatically amplified**:

1. **Routing inconsistency:** Even with identical inputs, tiny numerical differences between training and inference engines can cause completely different experts to be selected, massively amplifying train–inference discrepancy.
2. **Expert drift:** Policy updates not only change parameters but also change routing decisions, exacerbating policy staleness.

**Solution: Routing Replay**

**Vanilla Routing Replay (R2):**

During gradient updates, replay the experts that were selected by the rollout policy in the training engine. This primarily reduces **policy staleness**.

**Rollout Routing Replay (R3):**

In the training engine, replay the experts that were selected in the inference engine. This primarily reduces **train–inference discrepancy**.

**Best practices** (see Zheng et al.[^zheng] for detailed experiments):

- **On-policy / light off-policy:** Use MiniRL + R2 (Vanilla Routing Replay). The bias is small, and R2 is sufficient to stabilize training.
- **Heavy off-policy:** Use MiniRL + R3 (Rollout Routing Replay). Stability is paramount, and R3 more effectively eliminates discrepancies.

### Challenge 3: Expert Parallelism

Referring to the Megatron-Bridge documentation on expert parallelism[^megatron][^swift], **Expert Parallelism (EP)** and **Expert Tensor Parallelism (ETP)** are parallel strategies designed specifically for MoE models.

**EP:** Distributes different experts across different GPUs.

**ETP:** Further shards each individual expert's weights across multiple GPUs.

![Expert Parallelism (EP) distributes experts across GPUs; Expert Tensor Parallelism (ETP) shards each expert's weights](/assets/img/moe-expert-parallelism.jpg)

**Rule of thumb:** If you find that a single expert (with a large hidden size) doesn't fit on one GPU, increase ETP. If you have too many experts (say 256), increase EP.

## Recommended Tool

[MOE-Patch](https://github.com/direction-yxf/moe_patch) is a monitoring tool designed specifically for MoE models. Using a monkey-patching mechanism, it captures and analyzes routing distributions, expert load, token drop rates, and other key metrics in real time — without modifying the source code of training or inference frameworks (such as verl, ms-swift, vllm, etc.)[^moepatch]. It fills the gap in fine-grained MoE monitoring that current frameworks lack.

![MOE-Patch monitoring dashboard showing expert load distribution](/assets/img/moe-patch-screenshot.jpg)

[^shazeer]: Shazeer, N., et al. (2017). [*Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*](https://arxiv.org/abs/1701.06538). arXiv:1701.06538.

[^switch]: Fedus, W., Zoph, B., & Shazeer, N. (2022). [*Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*](https://arxiv.org/abs/2101.03961). JMLR, 23(120), 1–39.

[^zheng]: Zheng, C., et al. (2025). [*Stabilizing Reinforcement Learning with LLMs: Formulation and Practices*](https://arxiv.org/abs/2512.01374). arXiv:2512.01374.

[^megatron]: NVIDIA. [Megatron-Bridge Parallelisms Documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html).

[^moepatch]: [MOE-Patch](https://github.com/direction-yxf/moe_patch) — Fine-grained MoE monitoring tool.

[^swift]: [Megatron-SWIFT Command-Line Parameters](https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Command-line-parameters.html).

<hr class="section-divider">

*Citation: 燕雄飞的一天. (2026, March 19). moe架构的post-training难点与经验分享 [Post-Training Challenges and Lessons Learned with MoE Architectures]. Zhihu. [https://zhuanlan.zhihu.com/p/2018018879109091590](https://zhuanlan.zhihu.com/p/2018018879109091590)*
