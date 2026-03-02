---
title: "Beyond MuP: 2. Linear Layers and Steepest Descent"
subtitle: "Translated from [MuP之上：2. 线性层与最速下降](https://kexue.fm/archives/11605) by Jianlin Su (苏剑林)"
date: 2026-02-15T00:00:00+08:00
blurb: "Applying the three stability conditions to linear layers recovers MuP initialization and the Muon optimizer from first principles, via spectral norm analysis and steepest descent."
tags: ["translation", "mup", "muon", "optimization", "scaling"]
math: true
---

*Translator's note (Opus 4.6): This is an English translation of [MuP之上：2. 线性层与最速下降](https://kexue.fm/archives/11605) by Jianlin Su (苏剑林), originally published on February 15, 2026 on [Scientific Spaces (科学空间)](https://kexue.fm). It is the second article in the "Beyond MuP" series. The translation preserves the author's first-person voice.*

---

In the previous article [*Beyond MuP: 1. Three Characteristics of Good Models*](/translations/scientific-spaces/beyond-mup-1-three-characteristics-of-good-models/), we proposed three core metrics — forward stability, dependency stability, and update stability — and gave their mathematical definitions. We proposed using whether they satisfy \(\Theta(1)\) to characterize whether a model is "good," and this serves as the theoretical cornerstone for our subsequent analysis and calculations. Next, we will combine these with the idea of steepest descent to design "stable yet fast" update rules tailored to each parameter.

**Forward Stability:**

\[
\max_{\boldsymbol{x}} \| \boldsymbol{f}(\boldsymbol{x};\boldsymbol{\omega})\|_{\text{RMS}} = \Theta(1) \tag{1}
\]

**Dependency Stability:**

\[
\max_{\boldsymbol{x}_1,\boldsymbol{x}_2} \| \boldsymbol{f}(\boldsymbol{x}_1;\boldsymbol{\omega}) - \boldsymbol{f}(\boldsymbol{x}_2;\boldsymbol{\omega})\|_{\text{RMS}} = \Theta(1) \tag{2}
\]

**Update Stability:**

\[
\max_{\boldsymbol{x}} \| \boldsymbol{f}(\boldsymbol{x};\boldsymbol{\omega} + \Delta\boldsymbol{\omega}) - \boldsymbol{f}(\boldsymbol{x};\boldsymbol{\omega})\|_{\text{RMS}} = \Theta(1) \tag{3}
\]

We take the linear layer as our first example. The result will be familiar to some readers — it is precisely the Muon optimizer that has been gaining traction over the past year. Of course, our goal is not to rediscover Muon, but to demonstrate the process of designing models and optimizers from first principles, providing a unified methodology for handling other parameters later.

## Linear Transformation

For a linear layer, the input is a vector \(\boldsymbol{x}\in\mathbb{R}^{d_{\text{in}}}\), the parameter is a matrix \(\boldsymbol{W}\in\mathbb{R}^{d_{\text{in}}\times d_{\text{out}}}\), and the model is \(\boldsymbol{f}(\boldsymbol{x};\boldsymbol{W})=\boldsymbol{x}\boldsymbol{W}\). Note that in the definitions of the three metrics, we did not restrict \(\boldsymbol{x}\) to be bounded, so for a plain linear layer, none of the three metrics necessarily exist — for example, \(\max_{\boldsymbol{x}}\|\boldsymbol{x}\boldsymbol{W}\|_{\text{RMS}}\) is generally infinite. To address this, we simply add some operation that makes the result bounded, such as:

\[
\begin{aligned}
&\text{In Norm:}\quad \text{Norm}(\boldsymbol{x})\boldsymbol{W} \\[5pt]
&\text{Out Norm:}\quad \text{Norm}(\boldsymbol{x}\boldsymbol{W})
\end{aligned}
\]

where \(\text{Norm}(\boldsymbol{x}) = \boldsymbol{x} / \|\boldsymbol{x}\|_{\text{RMS}}\). Here we omit the gamma parameter that RMS Norm carries, assuming its effect is secondary. We know that residual connections have two common variants — Pre Norm and Post Norm. Pre Norm clearly corresponds to In Norm, but the key point here is that Post Norm is actually also In Norm:

\[
\begin{aligned}
&\text{Pre Norm:}\quad \boldsymbol{x}_{t+1} = \boldsymbol{x}_t + \boldsymbol{F}_t(\text{Norm}(\boldsymbol{x}_t)) \\[5pt]
&\text{Post Norm:} \quad \boldsymbol{x}_{t+1} = \text{Norm}(\underbrace{\boldsymbol{x}_t + \boldsymbol{F}_t(\boldsymbol{x}_t)}_{\text{denote as }\boldsymbol{y}_{t+1}}) \quad \Rightarrow\quad \boldsymbol{y}_{t+1} = \text{Norm}(\boldsymbol{y}_t) + \boldsymbol{F}_t(\text{Norm}(\boldsymbol{y}_t))
\end{aligned}
\]

So Post Norm compared to Pre Norm merely replaces \(\boldsymbol{x}_t + \boldsymbol{F}_t(\text{Norm}(\boldsymbol{x}_t))\) with \(\text{Norm}(\boldsymbol{x}_t) + \boldsymbol{F}_t(\text{Norm}(\boldsymbol{x}_t))\). From \(\boldsymbol{F}_t\)'s perspective, both are In Norm. This article also uses In Norm as the running example.

Compared to Out Norm, In Norm has the additional advantage of greater room for speedup. Since \((\boldsymbol{x} / \|\boldsymbol{x}\|_{\text{RMS}})\boldsymbol{W}=\boldsymbol{x}\boldsymbol{W} / \|\boldsymbol{x}\|_{\text{RMS}}\), in principle \(\boldsymbol{x}\boldsymbol{W}\) and \(\|\boldsymbol{x}\|_{\text{RMS}}\) can be computed in parallel, with the division performed last to reduce latency. This idea is reflected in works such as [*FlashNorm: Fast Normalization for LLMs*](https://arxiv.org/abs/2407.09577), [*Block-level AI Operator Fusion*](https://arxiv.org/abs/2505.07829), and [*Superoptimizing RMSNorm and Linear*](https://mirage-project.readthedocs.io/en/latest/tutorials/rms-norm-linear.html).

## Initial Variance

Following the discussion in the previous section, we agree to consider only linear layers with In Norm. Then, by the definition of the spectral norm, we can compute the three metrics:

\[
\begin{aligned}
&\text{Forward Stability:}\quad\max_{\|\boldsymbol{x}\|_{\text{RMS}}=1} \| \boldsymbol{x}\boldsymbol{W}\|_{\text{RMS}} = \sqrt{\frac{d_{\text{in}}}{d_{\text{out}}}}\|\boldsymbol{W}\|_2 \\[5pt]
&\text{Dependency Stability:}\quad\max_{\|\boldsymbol{x}_1\|_{\text{RMS}}=\|\boldsymbol{x}_2\|_{\text{RMS}}=1} \| \boldsymbol{x}_1\boldsymbol{W} - \boldsymbol{x}_2\boldsymbol{W}\|_{\text{RMS}} = 2\sqrt{\frac{d_{\text{in}}}{d_{\text{out}}}}\|\boldsymbol{W}\|_2 \\[5pt]
&\text{Update Stability:}\quad\max_{\|\boldsymbol{x}\|_{\text{RMS}}=1} \| \boldsymbol{x}(\boldsymbol{W} + \Delta\boldsymbol{W}) - \boldsymbol{x}\boldsymbol{W}\|_{\text{RMS}} = \sqrt{\frac{d_{\text{in}}}{d_{\text{out}}}}\|\Delta\boldsymbol{W}\|_2
\end{aligned}
\]

where \(\|\cdot\|_2\) applied to a matrix denotes its spectral norm. As we can see, all three metrics are variants of the spectral norm — or more precisely, the three metrics I proposed are generalizations built from the spectral norm.

The first two metrics are functions of \(\boldsymbol{W}\). They differ only by a factor of \(2\) and are essentially the same. If we want them to be \(\Theta(1)\), then \(\|\boldsymbol{W}\|_2 = \Theta(\sqrt{d_{\text{out}}/d_{\text{in}}})\), which at minimum imposes a requirement on the initialization of \(\boldsymbol{W}\). According to [*Fast Estimation of the Spectral Norm of Random Matrices*](https://kexue.fm/archives/11335), for a \(d_{\text{in}}\times d_{\text{out}}\) standard normal matrix, its spectral norm is approximately \(\sqrt{d_{\text{in}}} + \sqrt{d_{\text{out}}}\). So for the initialization to satisfy \(\|\boldsymbol{W}\|_2 = \Theta(\sqrt{d_{\text{out}}/d_{\text{in}}})\), the initial variance \(\sigma^2\) should satisfy

\[
\sigma = \Theta\left(\sqrt{\frac{d_{\text{out}}}{d_{\text{in}}}}\frac{1}{\sqrt{d_{\text{in}}} + \sqrt{d_{\text{out}}}}\right) \tag{4}
\]

Additionally, we can also consider constraining \(\|\boldsymbol{W}\|_2\) throughout the optimization process. This has inspired several works, such as [*Steepest Descent on Manifolds: 4. Muon + Spectral Sphere*](https://kexue.fm/archives/11241) and [*Controlled LLM Training on Spectral Sphere*](https://arxiv.org/abs/2601.08393). We will discuss this further in later articles.

## Steepest Descent

Next, we focus on the "update stability" metric \(\sqrt{d_{\text{in}}/d_{\text{out}}}\|\Delta\boldsymbol{W}\|_2\), which is a spectral norm variant of the parameter increment \(\Delta\boldsymbol{W}\). As everyone knows, the update is determined by the optimizer, so this part provides guidance for optimizer design. Following the "stable yet fast" principle, we now have "stability" — so when is it fastest?

This is precisely the question that steepest descent answers. We have discussed this previously in articles like [*Muon Sequel: Why We Chose to Try Muon?*](https://kexue.fm/archives/10739), [*Steepest Descent on Manifolds: 1. SGD + Hypersphere*](https://kexue.fm/archives/11196), and [*Steepest Descent on Manifolds: 2. Muon + Orthogonal*](https://kexue.fm/archives/11215), but for completeness of this series, let us go through it once more. Steepest descent refers to finding the update that decreases the loss the fastest under a given constraint. Formally:

\[
\min_{\Delta \boldsymbol{W}} \mathcal{L}(\boldsymbol{W} +\Delta\boldsymbol{W}) \qquad \text{s.t.}\qquad \rho(\Delta\boldsymbol{W})\leq \eta \tag{5}
\]

where \(\mathcal{L}\) is the loss function, and \(\rho(\Delta\boldsymbol{W})\) is the stability metric for the increment \(\Delta\boldsymbol{W}\) — which we already have: \(\sqrt{d_{\text{in}}/d_{\text{out}}}\|\Delta\boldsymbol{W}\|_2\). But solving this problem directly is still too complex. We need to replace \(\mathcal{L}(\boldsymbol{W} +\Delta\boldsymbol{W})\) with its first-order approximation \(\mathcal{L}(\boldsymbol{W}) + \langle \boldsymbol{G}, \Delta\boldsymbol{W}\rangle_F\) to make the problem tractable. The problem then becomes equivalent to

\[
\min_{\Delta \boldsymbol{W}} \text{tr}(\boldsymbol{G}^{\top}\Delta\boldsymbol{W}) \qquad \text{s.t.}\qquad \|\Delta\boldsymbol{W}\|_2\leq\eta\sqrt{\frac{d_{\text{out}}}{d_{\text{in}}}} \tag{6}
\]

where \(\boldsymbol{G}=\nabla_{\boldsymbol{W}}\mathcal{L}(\boldsymbol{W})\) is the gradient of the loss function, and we have used the identity \(\langle \boldsymbol{G}, \Delta\boldsymbol{W}\rangle_F=\text{tr}(\boldsymbol{G}^{\top}\Delta\boldsymbol{W})\).

## Solving the Problem

Going further, we set \(\Delta\boldsymbol{W}=-\kappa \boldsymbol{\Phi}\) and rewrite the optimization objective as

\[
\max_{\kappa,\boldsymbol{\Phi}}\kappa\,\text{tr}(\boldsymbol{G}^{\top}\boldsymbol{\Phi}) \qquad \text{s.t.}\qquad 0\leq \kappa \leq \eta\sqrt{\frac{d_{\text{out}}}{d_{\text{in}}}}, \quad\|\boldsymbol{\Phi}\|_2=1 \tag{7}
\]

Clearly, \(\kappa\) can be optimized independently — the maximum is achieved at \(\kappa = \eta\sqrt{d_{\text{out}}/d_{\text{in}}}\). So we only need to solve

\[
\max_{\boldsymbol{\Phi}} \text{tr}(\boldsymbol{G}^{\top}\boldsymbol{\Phi}) \qquad \text{s.t.}\qquad \|\boldsymbol{\Phi}\|_2=1 \tag{8}
\]

Next, suppose \(\boldsymbol{G}\) has SVD \(\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top} = \sum_{i=1}^r \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^{\top}\), where \(r\) is the rank of \(\boldsymbol{G}\). We have

\[
\text{tr}(\boldsymbol{G}^{\top}\boldsymbol{\Phi})=\text{tr}\left(\sum_{i=1}^r \sigma_i \boldsymbol{v}_i \boldsymbol{u}_i^{\top}\boldsymbol{\Phi}\right) = \sum_{i=1}^r \sigma_i \boldsymbol{u}_i^{\top}\boldsymbol{\Phi}\boldsymbol{v}_i \tag{9}
\]

By definition, when \(\|\boldsymbol{\Phi}\|_2=1\) we have \(\|\boldsymbol{\Phi}\boldsymbol{v}_i\|_2\leq \|\boldsymbol{v}_i\|_2=1\), and therefore \(\boldsymbol{u}_i^{\top}\boldsymbol{\Phi}\boldsymbol{v}_i\leq 1\). Thus

\[
\text{tr}(\boldsymbol{G}^{\top}\boldsymbol{\Phi})\leq \sum_{i=1}^r \sigma_i = \| \boldsymbol{G}\|_* \tag{10}
\]

where \(\|\cdot\|_*\) is the [nuclear norm](https://en.wikipedia.org/wiki/Schatten_norm) of the matrix. Equality holds when every \(\boldsymbol{u}_i^{\top}\boldsymbol{\Phi}\boldsymbol{v}_i\) equals 1, in which case

\[
\boldsymbol{\Phi} = \sum_{i=1}^r \boldsymbol{u}_i \boldsymbol{v}_i^{\top} = \boldsymbol{U}_{[:,:r]}\boldsymbol{V}_{[:,:r]}^{\top} = \text{msign}(\boldsymbol{G}) \tag{11}
\]

## Summary of Results

To briefly summarize: starting from the three stability metrics, we have obtained at least two conclusions. First, the initialization variance \(\sigma^2\) of the parameter \(\boldsymbol{W}\) should satisfy

\[
\sigma = \Theta\left(\sqrt{\frac{d_{\text{out}}}{d_{\text{in}}}}\frac{1}{\sqrt{d_{\text{in}}} + \sqrt{d_{\text{out}}}}\right) \tag{12}
\]

Second, its increment \(\Delta\boldsymbol{W}\) should take the following form

\[
\Delta\boldsymbol{W} = -\eta\sqrt{\frac{d_{\text{out}}}{d_{\text{in}}}}\,\text{msign}(\boldsymbol{G}) \tag{13}
\]

This is precisely the MuP version of Muon (for the differences between several versions, see [*Muon Optimizer Guide: Quick Start and Key Details*](https://kexue.fm/archives/11416); the standard Muon replaces \(\boldsymbol{G}\) with its momentum, which can be viewed as a smoother gradient estimate). Additionally, we still have some work to do regarding constraints on \(\boldsymbol{W}\), which we will explore in later articles.

Since we have already written extensively about MuP and Muon in previous blog posts, neither of these results is new at this point. This article serves only as the first case study, demonstrating the reasonableness of metrics \((1)\), \((2)\), and \((3)\). They provide a unified stability metric formula for the parameters of any layer and their increments, thereby generalizing Muon's conclusions.

## Remaining Questions

Before we generalize further, there is one question we need to answer: the derivation above was entirely based on In Norm — does every linear layer need In Norm? Can Muon still be used without In Norm? To answer this, let us borrow a passage from the previous article:

> Here, \(\boldsymbol{f}(\boldsymbol{x};\boldsymbol{\omega})\) can represent a single layer, a block of layers, or even the entire model. In theory, coarser granularity yields looser (i.e., more accurate) constraints, but computing the \(\max\) also becomes harder — so this depends on our ability to evaluate the \(\max\).

In short, the more accurately we can compute the stability metrics the better, but approximations are allowed. So how well Muon works without In Norm depends on how well "\(\|\boldsymbol{x}\|_{\text{RMS}}=\text{some constant}\)" holds. For example, for an FFN layer \(\boldsymbol{y}=\phi(\boldsymbol{x}\boldsymbol{W}_{\text{up}})\boldsymbol{W}_{\text{down}}\), if we assume the activation function \(\phi\) has Lipschitz constant 1, then we still have

\[
\|\boldsymbol{y}\|_{\text{RMS}} \leq \|\boldsymbol{x}\|_{\text{RMS}} \times\sqrt{\frac{d_{\text{in}}}{d_{\text{mid}}}}\|\boldsymbol{W}_{\text{up}}\|_2\times \sqrt{\frac{d_{\text{mid}}}{d_{\text{out}}}}\|\boldsymbol{W}_{\text{down}}\|_2 \tag{14}
\]

where \(\boldsymbol{W}_{\text{up}}\in\mathbb{R}^{d_{\text{in}}\times d_{\text{mid}}}\) and \(\boldsymbol{W}_{\text{down}}\in\mathbb{R}^{d_{\text{mid}}\times d_{\text{out}}}\). In this case, even if we only apply RMS Norm to \(\boldsymbol{x}\), the same stability metrics approximately hold for the second parameter \(\boldsymbol{W}_{\text{down}}\), and therefore Muon is still applicable.

Similarly, even without any RMS Norm at all, if we still believe that "\(\|\boldsymbol{x}\|_{\text{RMS}}=\text{some constant}\)" holds to some degree, then for the linear layers that follow, we can still try the Muon optimizer.

## Conclusion

This article took the three stability metrics from the previous article as the starting point and demonstrated the process of "recovering" the MuP and Muon conclusions for linear layers. Next, we will apply this same methodology to "customize" initialization and optimizers for parameters beyond linear layers.

---

*Citation: Su, J. (2026, February 15). MuP之上：2. 线性层与最速下降 [Beyond MuP: 2. Linear Layers and Steepest Descent]. Scientific Spaces. [https://kexue.fm/archives/11605](https://kexue.fm/archives/11605)*
