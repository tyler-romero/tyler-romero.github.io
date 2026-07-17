---
title: "The MoE Journey: 9. The Gate Normalization Debate"
subtitle: "Translated from [MoE环游记：9、门控归一化之争](https://kexue.fm/archives/11782) by Jianlin Su (苏剑林)"
date: 2026-06-17T00:00:00+08:00
blurb: "A first-principles analysis of normalization in MoE routing and gating, deriving normalized gates without Re-Norm through a probabilistic framework based on REINFORCE and straight-through estimation."
tags: ["translation", "mixture-of-experts", "routing", "gating", "reinforce", "straight-through-estimator"]
math: true
---

*Translator's note (GPT-5): This is an English translation of [MoE环游记：9、门控归一化之争](https://kexue.fm/archives/11782) by Jianlin Su (苏剑林), originally published on June 17, 2026 on [Scientific Spaces (科学空间)](https://kexue.fm). The translation preserves the author's first-person voice.*

<hr class="section-divider">

Looking back through the history of MoE, we find that in the early years, when the Router served as a Gate whose output multiplied the Experts, it was almost always activated with Softmax. This remains one of the standard forms of MoE today. To work with [Loss-Free](https://kexue.fm/archives/10757) load balancing, however, DeepSeek changed the activation function to Sigmoid and showed that this was also a highly competitive solution, prompting deeper thought and experimentation around the form a Router should take.

Even within the Softmax family, there are two slightly different approaches: should we apply Softmax before selecting the Top-\(k\), or select the Top-\(k\) before applying Softmax? The latter can also be understood as performing another normalization after selecting the Top-\(k\), i.e. Re-Norm. Should the Gate's activation function be normalized at all? If so, should Top-\(k\) selection happen after normalization, or should we use Re-Norm? This is the topic of this post.

## Problem Description

We know that the general form of MoE is

\[
\boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho}} \rho_i \boldsymbol{e}_i
\tag{1}
\]

Here, \(\boldsymbol{\rho}\) actually plays two roles: when it is used to select the Top-\(k\) Experts, it acts as the Router; when it is multiplied into the Experts, it acts as the Gate. From the standpoint of MoE design, the core role of \(\boldsymbol{\rho}\) is clearly routing. The Gate's purpose is to provide gradients to the Router during training.

The question we want to discuss can therefore be understood as how to construct \(\boldsymbol{\rho}=(\rho_1, \rho_2, \cdots, \rho_n)\) more scientifically so that the Router receives better gradients. For a long time, the standard answer has been Softmax:

\[
\rho_i = \frac{e^{s_i}}{\sum_{j=1}^n e^{s_j}}
\tag{2}
\]

where \(\boldsymbol{s}=(s_1, s_2, \cdots, s_n)\) are the Logits directly projected by a linear layer. Yet although this answer is "standard," I have not found much explanation for it. Everyone seems simply to have accepted and continued using it, which once left me deeply puzzled about the training mechanism of MoE.

## Other Choices

As mentioned at the beginning, DeepSeek tried Sigmoid activation in Loss-Free load balancing and later used it in [*DeepSeek-V3 Technical Report*](https://papers.cool/arxiv/2412.19437). Its success shows that non-Softmax activations can also work well. This has inspired more general approaches: [*ReMoE: Fully Differentiable Mixture-of-Experts with ReLU Routing*](https://papers.cool/arxiv/2412.14711), for example, uses ReLU activation, while the geometric perspective in [*The MoE Journey: 1. Starting from Geometric Meaning*](https://kexue.fm/archives/10699) permits any nonnegative activation function.

MoE also admits the Re-Norm option, which changes (1) to

\[
\boldsymbol{y} = \frac{\sum\limits_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho}} \rho_i \boldsymbol{e}_i}{\sum\limits_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho}} \rho_i}
\tag{3}
\]

In other words, it renormalizes the selected Top-\(k\) values \(\rho_i\). For Softmax, this is equivalent to using \(\boldsymbol{s}\) to select the Top-\(k\), setting the unselected values to \(-\infty\), and then applying Softmax again. Re-Norm has the advantage of making the numerical values in the forward pass more stable. Note, however, that when using Re-Norm, \(k\) must be greater than 1; otherwise, \(\boldsymbol{\rho}\) will have no gradient at all and cannot be trained.

Looking across current practice, these MoE variants perform more or less the same, with none showing a clear advantage. Since practice cannot distinguish a winner, let us investigate theoretically which form is more principled.

## Design Principle

Our goal is to find a first principle that lies closer to the essence of the problem, then use it to derive the gating mechanism used in today's MoEs.

The first question, naturally, is what this "principle" should be. For simplicity, consider \(k=1\). We know that the most important characteristic of MoE is sparsity: a Router first determines which Expert to activate, and then only those Experts are computed, increasing the parameter count while controlling the amount of computation. If this were the entire idea, the naive model would be

\[
\boldsymbol{f}\left(\boldsymbol{e}_{\operatorname*{argmax}\boldsymbol{\rho}}\right)
\tag{4}
\]

That is, select the highest-scoring entry from the Router \(\boldsymbol{\rho}\) and activate the corresponding Expert. This works perfectly well for inference, but during training, the Router receives no gradient and therefore cannot be updated. We must somehow design a gradient for it. To answer how, we must first clarify what kind of Router we want.

Because only one Expert can be activated, we naturally want it to be the best-performing Expert. If \(\ell\) denotes the loss function, our desired outcome can be written as

\[
\operatorname*{argmax} \boldsymbol{\rho} = \operatorname*{argmin}\, [\ell(\boldsymbol{e}_1),\ell(\boldsymbol{e}_2),\cdots,\ell(\boldsymbol{e}_n)]
\tag{5}
\]

This is the design principle we are looking for.

## Transforming the Objective

Objective (5), however, is not yet a loss function that can be used directly for training. It needs to be transformed further. To do this, we construct two distributions. The first is a target distribution \(\boldsymbol{q}=(q_1,q_2,\cdots,q_n)\) based on the loss function, defined as

\[
q_i = \frac{e^{-\ell(\boldsymbol{e}_i)/\tau}}{\sum_{j=1}^n e^{-\ell(\boldsymbol{e}_j)/\tau}}
\tag{6}
\]

This distribution is independent of the Router, so from the perspective of Router learning, it is the "target distribution." The second is a predicted distribution \(\boldsymbol{p}\) constructed from \(\boldsymbol{\rho}\). There are many possibilities: \(\boldsymbol{\rho}\) itself may be the distribution \(\boldsymbol{p}\) if it is already normalized; \(\boldsymbol{p}\) may instead be the Softmax of \(\boldsymbol{\rho}\), in which case \(\boldsymbol{\rho}\) contains the Logits; or a normalization method other than Softmax may be used. In short, \(\boldsymbol{p}\) is some probabilistic representation of the Router, whose parameters we denote by \(\boldsymbol{\theta}\).

We transform objective (5) into bringing \(\boldsymbol{p}\) and \(\boldsymbol{q}\) closer together, thereby providing a gradient for \(\boldsymbol{\theta}\). To do this, we minimize the KL divergence

\[
KL(\boldsymbol{p}\Vert \boldsymbol{q}) = \sum_{i=1}^n p_i \log \frac{p_i}{q_i}
\tag{7}
\]

which, after a little rearrangement, becomes

\[
KL(\boldsymbol{p}\Vert \boldsymbol{q}) = - \mathcal{H}(\boldsymbol{p}) + \frac{1}{\tau}\sum_{i=1}^n p_i \ell(\boldsymbol{e}_i) - \log \sum_{i=1}^n e^{-\ell(\boldsymbol{e}_i)/\tau}
\tag{8}
\]

This objective has three terms. The first is the negative entropy \(-\mathcal{H}(\boldsymbol{p})\); minimizing it means maximizing entropy, which encourages the model to explore thoroughly. Load balancing can be viewed as already serving a similar role, so we will ignore this term for now. The third term is independent of \(\boldsymbol{p}\), and thus of \(\boldsymbol{\theta}\). The equivalent loss function is therefore \(\mathcal{L} = \sum_{i=1}^n p_i \ell(\boldsymbol{e}_i)\).

## Straight-Through Estimation

Taking the gradient of the equivalent loss gives

\[
\nabla_{\boldsymbol{\theta}}\mathcal{L} = \sum_{i=1}^n \nabla_{\boldsymbol{\theta}} p_i \cdot \ell(\boldsymbol{e}_i) = \sum_{i=1}^n p_i \nabla_{\boldsymbol{\theta}} \log p_i \cdot \ell(\boldsymbol{e}_i) = \mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}}\log p_i \cdot \ell(\boldsymbol{e}_i)]
\tag{9}
\]

The key step uses \(\nabla_{\boldsymbol{\theta}} p_i = p_i \nabla_{\boldsymbol{\theta}} \log p_i\) to separate out a factor of \(p_i\). Only then can the sum be converted into an expectation and sampling be used to achieve MoE's goal of sparse computation. Some readers may already recognize this as REINFORCE from policy gradients! See [*Optimization from the Perspective of Sampling: A Unified View of Differentiable and Non-Differentiable Optimization*](https://kexue.fm/archives/7521) and [*Policy Gradients and Zeroth-Order Optimization: Different Paths to the Same Destination*](https://kexue.fm/archives/7737).

The problem with REINFORCE is its high noise. Intuitively, this is because it places \(p_i\) outside the loss function \(\ell\). If possible, we would prefer a "reparameterized" form with \(p_i\) inside \(\ell\). To derive such a form, we use the invariance of REINFORCE to subtracting a baseline:

\[
\begin{aligned}
\mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}}\log p_i \cdot \ell(\boldsymbol{e}_i)] =&\, \mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}}\log p_i \cdot (\ell(\boldsymbol{e}_i) - \ell(\boldsymbol{0}))] \\[4pt]
\approx&\, \mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}}\log p_i \cdot \langle\nabla_{\boldsymbol{e}_i} \ell(\boldsymbol{e}_i), \boldsymbol{e}_i - \boldsymbol{0}\rangle] \\[4pt]
= &\, \mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}} \langle\nabla_{\boldsymbol{e}_i} \ell(\boldsymbol{e}_i), \log p_i \cdot \boldsymbol{e}_i\rangle] \\[4pt]
= &\, \mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}} \ell((\log p_i + [1 - \log p_i]_{\text{sg}}) \cdot\boldsymbol{e}_i)] \\[4pt]
= &\, \nabla_{\boldsymbol{\theta}} \mathbb{E}_{i\sim \boldsymbol{p}} [\ell((\log p_i + [1 - \log p_i]_{\text{sg}}) \cdot\boldsymbol{e}_i)]
\end{aligned}
\tag{10}
\]

The approximation \(\approx\) is a first-order Taylor approximation with respect to \(\boldsymbol{e}_i\), and \([\cdot]_{\text{sg}}\) denotes Stop Gradient. We have ultimately obtained a Straight-Through Estimator (STE) that uses \(1\) in the forward pass and \(\log p_i\) in the backward pass to provide gradients for the Router.

## Final Form

Although the STE provides a workable training scheme, the inconsistency between its forward and backward passes often yields only suboptimal results. At this point, there is a remarkably effective improvement: replace each Expert with \(p_i\boldsymbol{e}_i\)! Repeating the derivation above, we obtain

\[
\begin{aligned}
\mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}}\log p_i \cdot \ell(p_i\boldsymbol{e}_i)] =&\, \mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}}\log p_i \cdot (\ell(p_i\boldsymbol{e}_i) - \ell(\boldsymbol{0}))] \\[4pt]
\approx&\, \mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}}\log p_i \cdot \langle\nabla_{p_i\boldsymbol{e}_i} \ell(p_i\boldsymbol{e}_i), p_i\boldsymbol{e}_i - \boldsymbol{0}\rangle] \\[4pt]
= &\, \mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}} \langle\nabla_{p_i \boldsymbol{e}_i} \ell(p_i \boldsymbol{e}_i), p_i \boldsymbol{e}_i\rangle] \\[4pt]
= &\, \mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}} \ell(p_i \boldsymbol{e}_i)] \\[4pt]
= &\, \nabla_{\boldsymbol{\theta}} \mathbb{E}_{i\sim \boldsymbol{p}} [\ell(p_i \boldsymbol{e}_i)]
\end{aligned}
\tag{11}
\]

This transformation is remarkably subtle and deserves careful reflection. By changing the Expert from \(\boldsymbol{e}_i\) to \(p_i\boldsymbol{e}_i\), we eliminate Stop Gradient and make the forward and backward passes consistent, theoretically raising the ceiling on model performance as well.

We can now answer the question posed at the beginning:

> If we require a top-down probabilistic derivation, then the Router should be normalized when it acts as the Gate, but it should not use Re-Norm.

## To Sample or Not to Sample

One detail worth noting is that \(\mathbb{E}_{i\sim \boldsymbol{p}}\) implies that we should sample from \(\boldsymbol{p}\), yet in practice we usually select the Top-\(k\) directly. How should we understand this discrepancy?

It is a tradeoff between diversity and stability. Random sampling encourages the model to explore more thoroughly, but it also increases gradient variance and introduces additional instability. Directly selecting the Top-\(k\) is more stable, but risks trapping the model in a suboptimal solution or even causing model collapse. Fortunately, today's load-balancing strategies are quite mature and already encourage the model to explore broadly to some extent, so Top-\(k\) selection remains the mainstream approach.

To sample while preserving stability, we can extend Top-\(k\) slightly rather than opening sampling up completely. For example, we could first select the Top-\(k+c\), then randomly choose \(k\) Experts from those \(k+c\); or add a small amount of noise to the Logits of \(\boldsymbol{p}\) before selecting the Top-\(k\). These approaches add randomness without straying too far from the original Top-\(k\), balancing diversity and stability.

## Related Work

To be clear, the derivation in this post is not actually new. I extracted and adapted it from Liyuan Liu's (刘力源) paper [*Sparse Backpropagation for MoE Training*](https://papers.cool/arxiv/2310.00811). That paper is accompanied by an earlier work, [*Bridging Discrete and Backpropagation: Straight-Through and Beyond*](https://papers.cool/arxiv/2304.08612), and a later one, [*GRIN: GRadient-INformed MoE*](https://papers.cool/arxiv/2409.12136).

Although these papers date from 2023 and 2024, I still highly recommend this trilogy to anyone seeking a deeper understanding of MoE Routers. They provide a unified probabilistic framework for designing gradients for various discrete operations. Of course, the probabilistic framework also has limitations: it is highly formalized and can feel somewhat constraining in practice.

For example, when \(k = 2\), a direct extension of the preceding result would be

\[
\mathbb{E}_{i,j\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}}\log p_i p_j \cdot \ell(p_i p_j (\boldsymbol{e}_i + \boldsymbol{e}_j))] \approx \nabla_{\boldsymbol{\theta}} \mathbb{E}_{i,j\sim \boldsymbol{p}} [\ell(p_i p_j (\boldsymbol{e}_i + \boldsymbol{e}_j))]
\tag{12}
\]

This treats the joint distribution \(p_i p_j\) and the sum of an Expert pair \(\boldsymbol{e}_i + \boldsymbol{e}_j\) as the basic unit, converting Top-2 into Top-1. The MoE we have always used, however, takes the form \(\ell(p_i \boldsymbol{e}_i + p_j\boldsymbol{e}_j)\), for which it is difficult to find an exact probabilistic derivation.

At this point, a more relaxed interpretation may be to treat it directly as an analogue of MaxPooling without insisting on a probabilistic explanation. Alternatively, we can understand it through the geometric interpretation in [*The MoE Journey: 1. Starting from Geometric Meaning*](https://kexue.fm/archives/10699). Overall, the probabilistic framework only proves that a particular scheme is feasible; in principle, it does not rule out the feasibility of other schemes.

## Summary

Starting from first principles, this post explored the design of Routers and Gates in MoE and provided a probabilistic explanation for gate normalization.

<hr class="section-divider">

*Citation: Su, J. (2026, June 17). MoE环游记：9、门控归一化之争 [The MoE Journey: 9. The Gate Normalization Debate]. Scientific Spaces. [https://kexue.fm/archives/11782](https://kexue.fm/archives/11782)*

*Original content licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). This translation is shared under the same license.*
