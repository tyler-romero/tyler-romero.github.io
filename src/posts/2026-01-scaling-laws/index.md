---
title: Thinking about Scaling Laws
subtitle: How can we use scaling laws to train stronger LLMs?
date: 2026-01-10T00:00:00-08:00
blurb: How can we use scaling laws to train stronger LLMs?
tags: ["post", "scaling-laws", "chinchilla", "llms"]
math: true
---

*Note: This post is a work in progress. I will continue to update and expand it over time.*

Scaling laws are one of the few tools we have for predicting model performance before committing serious compute. When a single training run can cost millions of dollars and take months, the ability to extrapolate from small-scale experiments becomes invaluable. In this post, I want to explore how we can use scaling laws not just as descriptive summaries, but as decision-making frameworks for comparing architectural choices and planning training runs.

This post assumes familiarity with the seminal papers [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (Kaplan et al.) and [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Hoffmann et al.). If you haven't read them, I'd recommend at least skimming Hoffmann et al. (the "Chinchilla" paper) first.

## The Chinchilla Scaling Law

The scaling laws from Hoffmann et al., also known as the "Chinchilla" scaling laws, are the basis for the billions of dollars being spent on training larger models on more data. They are empirical predictions—validated across many orders of magnitude of scale—about how much compute (in terms of data and model size) is required to lower loss. Chinchilla also demonstrates that model size and data should be scaled up at roughly the same rate, giving rise to the well-known heuristic of "20 tokens per parameter" as the optimal ratio.

The Chinchilla paper uses three separate approaches to arrive at the same "compute-optimal" scaling results. In particular, the third approach models the final loss of a language model as a function of model size $N$ and training data $D$ in the following form:

$$
L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

Where:

- **N** = number of model parameters
- **D** = number of training tokens
- **E** = irreducible loss — the theoretical minimum loss achievable with infinite compute
- **A, α** = parameter scaling coefficients (how loss improves with more parameters)
- **B, β** = data scaling coefficients (how loss improves with more training data)

The parameters of this function are fit to empirical data.

The fitted values from the Chinchilla paper are approximately $\alpha \approx 0.34$, $\beta \approx 0.28$, $E \approx 1.69$, $A \approx 406.4$, and $B \approx 410.7$.

The key insight comes from minimizing this loss function subject to a fixed compute budget $C$. Since compute scales roughly as $C \propto 6ND$ (the number of FLOPs required for a forward and backward pass through a model of size $N$ for $D$ tokens), we can derive the optimal allocation of compute between model size and data.

Taking the derivative and setting it to zero, we find that the optimal scaling satisfies:

$$
N_{opt} \propto C^{a}, \quad D_{opt} \propto C^{b}
$$

where $a = \frac{\beta}{\alpha + \beta}$ and $b = \frac{\alpha}{\alpha + \beta}$. Using the fitted values from Chinchilla ($\alpha \approx 0.34$, $\beta \approx 0.28$), we get $a \approx 0.45$ and $b \approx 0.55$. This means that as compute increases, we should scale data slightly faster than model size.

## Decision Making with Scaling Laws

The reason I dive into Chinchilla's third approach is that a fitted function capable of predicting loss as a function of N and D is incredibly powerful for making real decisions about model development.

Let's say we have a baseline model and a modeling change/intervention we want to test. Many researchers will test their change at 1–3 model scales on 1 or 2 different training dataset sizes, then conclude that their approach is better than the baseline on the back of these results. However, this does not tell the full story. A more correct way of conducting this experiment would be to fit empirical scaling laws based on results across many different model scales and training dataset sizes, and then compare the scaling laws to determine *in which regimes* their approach is better or worse than the baseline.

### Scaling Behavior: Efficiency vs. Scalability

It's important to distinguish between three different aspects of model performance:

**1. Offsets (A and B coefficients)**

- These coefficients are multiplicative constants that shift the scaling curve vertically
- Lower A means lower loss for any given model size (a constant factor improvement in the parameter term)
- Lower B means lower loss for any given dataset size (a constant factor improvement in the data term)
- The optimal compute split between parameters and data is *independent* of A and B
- A variant can have better offsets (lower A, B) while scaling at the same rate as baseline

**2. Scalability (α and β exponents)**

- These exponents determine the *slope* of the scaling curve
- A model with higher α and β will *eventually* beat one with lower exponents, regardless of A and B values (assuming a constant E)
- Higher α means loss decreases faster as you add parameters
- Higher β means loss decreases faster as you add data
- A variant with worse scaling (lower α, β) will eventually be overtaken by baseline at large enough scale

**3. Irreducible Loss (E)**

- At large scales, the A/N^α and B/D^β terms shrink toward zero, so E dominates.
- Therefore, a lower E can indicate that a model will achieve a lower loss at large scale even if the marginal scaling efficiency is worse.
- Despite there being a constant entropy floor for a given dataset, not all models can achieve the same irreducible loss on that dataset.

**The Crossover Problem**

A variant that has *better offsets but scales worse* presents an interesting tradeoff:
- At small scale: Variant wins due to lower A and B
- At large scale: Baseline wins due to better scaling (higher α, β)
- The **crossover point** is where baseline catches up

**When Scaling Differences Matter**

Small differences in α or β compound significantly at scale:
- A 1% difference in α over 1000× compute increase → ~7% difference in the parameter term
- For frontier models (70B+ params, 10T+ tokens), even small α/β differences are meaningful

The table below gives practical guidance on how to interpret differences in scaling parameters for different models.

| α, β vs baseline | A, B vs baseline | E vs baseline | Verdict |
|------------------|------------------|---------------|---------|
| Higher (scales better) | Lower (better offsets) | Lower | **Best case** — wins at all scales |
| Higher (scales better) | Lower (better offsets) | Higher | Wins at most scales, but baseline may catch up at very large scale due to E floor |
| Higher (scales better) | Higher (worse offsets) | Lower | Likely wins at large scale (better scaling + lower floor) |
| Higher (scales better) | Higher (worse offsets) | Higher | Mixed — scaling helps but E hurts; depends on target scale |
| Similar | Lower (better offsets) | Lower | **Good** — consistent gains at all scales |
| Similar | Lower (better offsets) | Higher | Wins at small/medium scale, may lose at very large scale |
| Similar | Higher (worse offsets) | Lower | May recover at very large scale due to lower E |
| Similar | Higher (worse offsets) | Higher | **Bad** — loses at all scales |
| Lower (scales worse) | Lower (better offsets) | Lower | Complex tradeoff — E helps at large scale but worse α/β hurts |
| Lower (scales worse) | Lower (better offsets) | Higher | Wins at small scale only — crossover point exists |
| Lower (scales worse) | Higher (worse offsets) | Lower | Only hope is very large scale where E dominates |
| Lower (scales worse) | Higher (worse offsets) | Higher | **Worst case** — loses at all scales |

### Targeting Specific Model and Dataset Sizes

When pretraining models, there is essentially a set of model families of roughly the same size. For example, there are many 7B, 32B, and 70B parameter models. Additionally, labs know how many useful tokens they have available for pretraining (or roughly how many GPU hours can be allotted to a specific run).

Using a chosen model and dataset size (e.g., N=32B, D=10T), we can use our formula to predict the final training loss of a given model architecture based on our empirical scaling law fit. This makes decision making straightforward: whichever intervention predicts the lowest loss at the target scale is the one we should use[^considerations].

[^considerations]: However, this is still ignoring other important considerations, such as inference-time efficiency, long-context performance, training throughput, and training stability. A complete evaluation framework would need to weigh these factors alongside the raw scaling predictions. For example, faster inference-time efficiency is critical for scaling up RL post-training and directly impacts the end-user experience.

## Getting strong scaling law fits

Coming soon.

## Recommended Reading

- [Resolving Discrepancies in Compute-Optimal Scaling of Language Models](https://arxiv.org/abs/2406.19146) — Investigates why different labs arrive at different scaling law coefficients and proposes methods to reconcile them.

- [Chinchilla Scaling: A replication attempt](https://www.semanticscholar.org/paper/Chinchilla-Scaling%3A-A-replication-attempt-Besiroglu-Erdil/2cfe76f2fcb272fd0dde67b5468cdc462416fd38) — A thorough attempt to replicate Chinchilla's results, revealing challenges and nuances in the original methodology.

- [Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://www.semanticscholar.org/paper/Beyond-Chinchilla-Optimal%3A-Accounting-for-Inference-Sardana-Doubov/82f75d838e92196864131bad25b1abc3b5d40a6f) — Argues that optimal training compute allocation should factor in inference costs, leading to smaller, more over-trained models.

- [Scaling Data-Constrained Language Models](https://www.semanticscholar.org/paper/Scaling-Data-Constrained-Language-Models-Muennighoff-Rush/9e16d8cc6096ec0d2733a4ecf41ce09d9a4bd19c) — Explores what happens when you run out of unique data and must repeat tokens, with implications for data-constrained regimes.

- [xLSTM Scaling Laws](https://arxiv.org/abs/2510.02228) — Useful case study that compares xLSTMs to Transformers using scaling laws.

![From the xLSTM Scaling Laws paper.](/assets/img/xlstm-scaling-law.png)
