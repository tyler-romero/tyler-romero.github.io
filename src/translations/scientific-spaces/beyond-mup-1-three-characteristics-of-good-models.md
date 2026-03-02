---
title: "Beyond MuP: 1. Three Characteristics of Good Models"
subtitle: "Translated from [MuP之上：1. 好模型的三个特征](https://kexue.fm/archives/11340) by Jianlin Su (苏剑林)"
date: 2025-10-21T00:00:00+08:00
blurb: "What does it mean for a model to be 'good'? Three stability conditions — forward, dependency, and update — form the foundation for understanding MuP, Muon, and principled model optimization."
tags: ["translation", "mup", "muon", "optimization", "scaling"]
math: true
---

*Translator's note (Opus 4.6): This is an English translation of [MuP之上：1. 好模型的三个特征](https://kexue.fm/archives/11340) by Jianlin Su (苏剑林), originally published on October 21, 2025 on [Scientific Spaces (科学空间)](https://kexue.fm). It is the first article in the "Beyond MuP" series. The translation preserves the author's first-person voice.*

<hr class="section-divider">

I wonder if anyone else has noticed an interesting detail: both Muon and MuP start with "Mu," yet the two "Mu"s have completely different origins. The former stands for "**Mo**ment**U**m Orthogonalized by **N**ewton-Schulz," while the latter stands for "**M**aximal **U**pdate **P**arametrization." And yet, there is a remarkably deep connection between them. In other words, Muon and MuP have entirely different starting points but ultimately converge in the same direction — even inadvertently adopting similar names, as if by fate.

But let me get to the point. Through various coincidences, I happened to learn about Muon and MuP at the same time, which greatly deepened my understanding of model optimization and led me to think about its more fundamental principles. After a period of trial and error, I arrived at some modest insights that I'd like to share here.

## Preface

In chronological order, MuP came before Muon, but my own learning went in reverse — I studied Muon first, then MuP. In hindsight, this turned out to be a perfectly fine learning order.

In previous articles like [*Appreciating the Muon Optimizer: The Essential Leap from Vectors to Matrices*](https://kexue.fm/archives/10592) and [*Muon Sequel: Why We Chose to Try Muon?*](https://kexue.fm/archives/10739), we described Muon as "steepest descent under a spectral norm constraint." The MuP line of work then provides exactly the justification for *why* the spectral norm constraint is needed. The two fit together perfectly.

A clarification on terminology: when we say "MuP," it can refer to two things. First, there is what was introduced in [*A First Look at MuP: Hyperparameter Transfer Across Model Scales*](https://kexue.fm/archives/10770), which is part of the [Tensor Programs](https://arxiv.org/abs/2203.03466) series — we call this "basic MuP." Second, there is what was introduced in [*Higher-Order MuP: A Simpler yet More Sophisticated Spectral Condition Scaling*](https://kexue.fm/archives/10795) — we call this "higher-order MuP." The latter derives richer conclusions than basic MuP in a more concise way. Both are the work of [Greg Yang](https://thegregyang.com/) (salute to the master).

Unless otherwise stated, "MuP" in this article refers to "higher-order MuP." In fact, this series — which I'm calling "Beyond MuP" — is a continuation of thinking and extensions built on higher-order MuP. However, some readers may only be familiar with the Tensor Programs version ("basic MuP"), and might initially wonder how MuP could answer the question of "why we need spectral norm constraints."

Regardless, I will try to make this series self-contained. Although we will reference many related papers and blog posts along the way, readers do not need to study all of them in detail.

## Stable yet Fast

Let me get back to the main topic. As the first article in this series, the goal here is to establish the core objective. More specifically, we want to think clearly about "what kind of model do we actually want?" and "how can we train such a model?"

Intuitively, as long as the model shows no signs of collapse, we can keep training until it converges to a satisfactory result. On top of that, we look for ways to make convergence faster. So really, it all comes down to two things — "stability" and "speed" — or rather, one thing: **being stable yet fast** (稳中求快). How do we determine whether a model is stable? Naturally, we need to monitor various "internal metrics"[^internal-metrics] — the more we monitor, the more problems we can expose.

[^internal-metrics]: The term "internal metrics" (内科指标, literally "internal medicine indicators") comes from a [Zhihu question](https://www.zhihu.com/question/1946325762161483910/answer/1946691536009036242) about a concept Yang Zhilin (杨植麟) mentioned in a recent interview. In his answer, Su Jianlin explains: these are monitoring indicators that tell you whether the model is training normally — analogous to vital signs in medicine. Simple ones like heart rate and blood pressure correspond to **Loss** and **Grad Norm** in training. If either suddenly spikes, it may indicate a data problem or even a hardware issue, and you try to locate the cause. More specialized indicators exist too — for example, K2's MuonClip optimizer targets **MaxLogit**, an important Attention mechanism indicator that is considered harmful when too large. The more metrics you monitor, the more problems you can expose, and the better model you can train by addressing them. Of course, even if some indicators show anomalies, it doesn't necessarily mean training will fail — an occasional Loss spike can still yield a useful model. Which metrics to monitor and which problems to address partly reflects the values of the training team.

However, rather than listing every possible metric, I want to identify the most essential conditions. To do so, let us first define a concept — **RMS (Root Mean Square)**: for \(x = (x_1, x_2, \dots, x_d) \in \mathbb{R}^d\), we define

\[
\|x\|_{\text{RMS}} = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2} = \frac{\|x\|_2}{\sqrt{d}} \tag{1}
\]

This represents the average scale per element, differing from the vector norm \(\|x\|_2\) by a factor of \(\sqrt{d}\).

Some readers might ask: since the difference is just a constant factor, why not observe the norm directly instead of defining a new concept? There are several reasons. For instance, RMSNorm is commonly used, and RMS values are easier to interpret intuitively. Moreover, there is an important reason: most activation functions are element-wise, so we need to examine and control the *per-element* scale to ensure that activation functions behave similarly across models of different sizes.

## Three Conditions

With the RMS notation in hand, we can now state what I consider the three most essential conditions for stably training a good model:

**Forward Stability:**

\[
\max_{x} \|f(x; \omega)\|_{\text{RMS}} = \Theta(1) \tag{2}
\]

**Dependency Stability:**

\[
\max_{x_1, x_2} \|f(x_1; \omega) - f(x_2; \omega)\|_{\text{RMS}} = \Theta(1) \tag{3}
\]

**Update Stability:**

\[
\max_{x} \|f(x; \omega + \Delta\omega) - f(x; \omega)\|_{\text{RMS}} = \Theta(1) \tag{4}
\]

Here, \(f(x; \omega)\) represents a family of models mapping \(\mathbb{R}^{d_{\text{in}}} \to \mathbb{R}^{d_{\text{out}}}\), with input \(x \in \mathbb{R}^{d_{\text{in}}}\), output \(f(x; \omega) \in \mathbb{R}^{d_{\text{out}}}\), and parameters \(\omega\) (which may be scalars, vectors, or matrices). \(\Theta\) denotes "[Big Theta Notation](https://en.wikipedia.org/wiki/Big_O_notation#Family_of_Bachmann%E2%80%93Landau_notations)." Here, \(f(x; \omega)\) can represent a single layer, a block of layers, or even the entire model. In theory, coarser granularity yields looser (i.e., more accurate) constraints, but computing the \(\max\) also becomes harder — so this depends on our ability to evaluate the \(\max\).

Among these three conditions, Equation \((2)\) is probably the easiest to understand. It represents the stability of forward computation. After taking the \(\max\) over \(x\), the only remaining variable is \(\omega\), so this is a constraint on \(\omega\). Note that we do not restrict the domain of \(x\), so by default \(x \in \mathbb{R}^{d_{\text{in}}}\), which means the maximum may not exist — for example, for nonzero \(W\), we have \(\max_{x} \|xW\|_{\text{RMS}} \to \infty\).

To ensure the maximum exists, we typically add some form of normalization, such as:

**In Norm:**

\[
\text{Norm}(x) W \tag{5}
\]

**Out Norm:**

\[
\text{Norm}(xW) \tag{6}
\]

where \(\text{Norm}(x) = x / \|x\|_{\text{RMS}}\), i.e., RMS Norm (though other normalizations could also be used). So condition \((2)\) implicitly imposes requirements on the model architecture as well. Similarly, Equation \((3)\) requires that the model architecture depends smoothly on its input. A simple example: \(f(x; \omega) = x \times \omega \times 0 + 1\). This "model" is certainly stable in the forward pass, but it does not depend on \(x\) at all, so Equation \((3)\) cannot be satisfied — it is not a good model.

Finally, Equation \((4)\) should also be straightforward to understand. After taking the \(\max\) over \(x\), the result constrains \(\omega\) and \(\Delta\omega\). It primarily concerns the effect of the increment \(\Delta\omega\), so it represents our expectation for training stability. We can use it to guide optimizer hyperparameter settings, and we can even construct new optimizers based on it.

## Related Thoughts

In summary, conditions \((2)\), \((3)\), and \((4)\) integrate considerations of model architecture, initialization, and optimization. It is hard to argue that any one of them can be removed, so I believe all three are necessary. That said, there are some details worth discussing further — for instance, the choice between \(\max\) and \(\mathbb{E}\).

In the formulations above, we used \(\max\) to "eliminate" \(x\), obtaining expressions involving only \(\omega\) and \(\Delta\omega\). Some readers might find it more intuitive to take the mathematical expectation \(\mathbb{E}_x\) instead. Why \(\max\) and not \(\mathbb{E}\)? There are several reasons.

First, computing \(\max\) only requires specifying the domain of \(x\), whereas computing \(\mathbb{E}\) requires defining a distribution over \(x\). Different distributions give different results, and defining this distribution accurately is far from straightforward.

Second, \(\max\) has the advantage of being invariant under monotone transformations, while \(\mathbb{E}\) does not. For example, with \(\max\), we have the identity

\[
\left( \max_{x} \|f(x; \omega)\|_{\text{RMS}} \right)^2 = \max_{x} \|f(x; \omega)\|_{\text{RMS}}^2
\]

That is, whether we take the \(\max\) of \(\|f(x; \omega)\|_{\text{RMS}}\) or \(\|f(x; \omega)\|_{\text{RMS}}^2\), it is essentially the same thing. But \(\mathbb{E}\) does not work this way — the expectation of \(\|f(x; \omega)\|_{\text{RMS}}\) and the expectation of \(\|f(x; \omega)\|_{\text{RMS}}^2\) typically differ in computational difficulty and may bear no simple relationship to each other.

Therefore, \(\max\) is simpler both conceptually and in its properties. One possible concern is whether \(\max\) is too strict — something like a "sufficient but not necessary" condition. In fact, \(\max\) is just the intuitive term; mathematically it is the supremum (\(\sup\)), and the "sup" indicates this value is tight and achievable. In practice, the mean and the maximum are usually of the same order, and our target is only \(\Theta(1)\), so the difference is negligible. On the contrary, \(\max\) accounts for extreme cases and provides the strongest guarantee of training stability — which is especially important for training large models like LLMs.

In fact, basic MuP — the Tensor Programs series — performs its analysis based on \(\mathbb{E}\), while higher-order MuP, like this article, is based on \(\max\). In hindsight, the \(\mathbb{E}\)-based analysis is inferior to higher-order MuP in terms of computational simplicity and generality of results, which further corroborates the effectiveness of the \(\max\)-based approach.

## Conclusion

Starting from this article, I will share a top-down understanding of model optimization, building on the earlier "higher-order MuP" as a foundation for further thought and extension. As the first article, we have described three basic conditions for model stability — or equivalently, three characteristics of a good model. These will serve as the cornerstone for the calculations and analysis that follow.

<hr class="section-divider">

*Citation: Su, J. (2025, October 21). MuP之上：1. 好模型的三个特征 [Beyond MuP: 1. Three Characteristics of Good Models]. Scientific Spaces. [https://kexue.fm/archives/11340](https://kexue.fm/archives/11340)*

*Original content licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). This translation is shared under the same license.*
