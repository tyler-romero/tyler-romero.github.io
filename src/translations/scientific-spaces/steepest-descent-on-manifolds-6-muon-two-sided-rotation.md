---
title: "Steepest Descent on Manifolds: 6. Muon + Two-Sided Rotation"
subtitle: "Translated from [流形上的最速下降：6. Muon + 双旋转](https://kexue.fm/archives/11777) by Jianlin Su (苏剑林)"
date: 2026-06-08T00:00:00+08:00
blurb: "An introduction to MuonR, a two-sided rotational variant of Muon that updates a matrix's singular vectors while preserving its singular-value distribution, with practical strategies for stable optimizer switching."
tags: ["translation", "muon", "optimization", "manifold-optimization", "singular-values"]
math: true
---

*Translator's note (GPT-5): This is an English translation of [流形上的最速下降：6. Muon + 双旋转](https://kexue.fm/archives/11777) by Jianlin Su (苏剑林), originally published on June 8, 2026 on [Scientific Spaces (科学空间)](https://kexue.fm). The translation preserves the author's first-person voice.*

<hr class="section-divider">

We know that when matrix parameters are updated with optimizers such as Adam and Muon, their singular values and left and right singular vectors all change along with them, and these changes are usually coupled. It is precisely because of this coupling that we cannot easily control a matrix parameter's singular values. When the singular values grow abnormally, we therefore have no simple and effective way to stop them, which may cause training to fail.

Inspired by [*Pion: A Spectrum-Preserving Optimizer via Orthogonal Equivalence Transformation*](https://papers.cool/arxiv/2605.12492) (hereafter abbreviated as Pion), this post proposes a Muon variant that updates a matrix's left and right singular vectors separately: "Rotational Muon" (MuonR). It can keep the matrix's singular-value distribution unchanged, thereby ensuring training stability.

## Review of Previous Work

Because the matrices formed by the left and right singular vectors must be orthogonal, let us first briefly review Muon under an orthogonality constraint. Let the parameter be \(\boldsymbol{W}\in\mathbb{R}^{n\times n}\), satisfying \(\boldsymbol{W}^{\top}\boldsymbol{W}=\boldsymbol{I}\), and let the update be \(\Delta\boldsymbol{W}=-\eta \boldsymbol{\Phi}\). We want the parameter to remain orthogonal after the update, so the corresponding spectral-norm steepest descent problem is

\[
\max_{\boldsymbol{\Phi}} \operatorname{tr}(\boldsymbol{G}^{\top}\boldsymbol{\Phi}) \qquad \text{s.t.}\qquad \Vert\boldsymbol{\Phi}\Vert_2 = 1,\quad(\boldsymbol{W} - \eta \boldsymbol{\Phi})^{\top}(\boldsymbol{W} - \eta \boldsymbol{\Phi})=\boldsymbol{I}
\tag{1}
\]

The solution is \(\boldsymbol{\Phi} = \boldsymbol{W}\boldsymbol{O}\), where \(\boldsymbol{O}=\operatorname{msign}([\boldsymbol{W}^{\top}\boldsymbol{G}]_{\text{skew}})\), and \([\boldsymbol{X}]_{\text{skew}} = (\boldsymbol{X} - \boldsymbol{X}^{\top})/2\) is the skew-symmetrization operator. Including the retraction operation, the complete update rule is

\[
\boldsymbol{W} \quad \leftarrow\quad \boldsymbol{W}(\boldsymbol{I} - \eta\boldsymbol{O})\left(\boldsymbol{I} - \boldsymbol{O}^{\top}\boldsymbol{O} + \frac{\boldsymbol{O}^{\top}\boldsymbol{O}}{\sqrt{1+\eta^2}}\right)
\tag{2}
\]

In particular, if \([\boldsymbol{W}^{\top}\boldsymbol{G}]_{\text{skew}}\) is full rank, this simplifies to

\[
\boldsymbol{W} \quad \leftarrow\quad \frac{\boldsymbol{W}(\boldsymbol{I} - \eta\boldsymbol{O})}{\sqrt{1+\eta^2}}
\tag{3}
\]

The derivation can be found in [*Steepest Descent on Manifolds: 2. Muon + Orthogonal*](https://kexue.fm/archives/11215), so we will not expand on it here. Both (2) and (3) are fully analytic. They add only a few matrix multiplications on top of Muon, with no significant increase in complexity, so the result is entirely practical.

Now consider a matrix \(\boldsymbol{W}\in\mathbb{R}^{n\times m}(n \geq m)\). If it also satisfies \(\boldsymbol{W}^{\top}\boldsymbol{W}=\boldsymbol{I}\), then we say that \(\boldsymbol{W}\) lies on the Stiefel manifold, which generalizes the concept of an orthogonal matrix. In principle, the result above can be extended to the Stiefel manifold, but for non-square matrices this requires solving a system of nonlinear equations, making it difficult to put into practice. For details, see [*Steepest Descent on Manifolds: 3. Muon + Stiefel*](https://kexue.fm/archives/11221).

## Instantaneous Reparameterization

Next, we turn our attention to an arbitrary parameter matrix \(\boldsymbol{W}\in\mathbb{R}^{n\times m}\). Our goal is to keep its singular values unchanged throughout the update process, thereby eliminating the possibility of abnormal singular-value growth.

To achieve this, we use the idea of "instantaneous reparameterization": before beginning the update, we reparameterize \(\boldsymbol{W}\) as \(\tilde{\boldsymbol{W}} = \boldsymbol{L}\boldsymbol{W}\boldsymbol{R}\), where \(\boldsymbol{L}\in\mathbb{R}^{n\times n},\boldsymbol{R}\in\mathbb{R}^{m\times m}\), both initialized to the identity. Thus, at initialization, \(\tilde{\boldsymbol{W}}=\boldsymbol{W}\). Writing \(\boldsymbol{G} = \nabla_{\boldsymbol{W}}\mathcal{L}\), we have

\[
\nabla_{\boldsymbol{L}}\mathcal{L} = \boldsymbol{G}\boldsymbol{W}^{\top},\qquad \nabla_{\boldsymbol{R}}\mathcal{L} = \boldsymbol{W}^{\top}\boldsymbol{G}
\tag{4}
\]

We then freeze \(\boldsymbol{W}\) and update only \(\boldsymbol{L}\) and \(\boldsymbol{R}\), while maintaining their orthogonality throughout the update. The updated \(\tilde{\boldsymbol{W}}\) therefore has the same singular values as \(\boldsymbol{W}\). From the perspective of \(\boldsymbol{L}\) and \(\boldsymbol{R}\), the problem has once again become steepest descent on the orthogonal manifold. Moreover, both \(\boldsymbol{L}\) and \(\boldsymbol{R}\) are square matrices, so the corresponding steepest descent has a fully analytic solution. According to (3), we can directly write the update rules as

\[
\boldsymbol{L}\quad\leftarrow\quad (\boldsymbol{I} - \eta\boldsymbol{O}_L)\left(\boldsymbol{I} - \boldsymbol{O}_L^{\top}\boldsymbol{O}_L + \frac{\boldsymbol{O}_L^{\top}\boldsymbol{O}_L}{\sqrt{1+\eta^2}}\right)
\tag{5}
\]

\[
\boldsymbol{R}\quad\leftarrow\quad (\boldsymbol{I} - \eta\boldsymbol{O}_R)\left(\boldsymbol{I} - \boldsymbol{O}_R^{\top}\boldsymbol{O}_R + \frac{\boldsymbol{O}_R^{\top}\boldsymbol{O}_R}{\sqrt{1+\eta^2}}\right)
\tag{6}
\]

where \(\boldsymbol{O}_L = \operatorname{msign}([\boldsymbol{G}\boldsymbol{W}^{\top}]_{\text{skew}})\) and \(\boldsymbol{O}_R = \operatorname{msign}([\boldsymbol{W}^{\top}\boldsymbol{G}]_{\text{skew}})\). Multiplying the new \(\boldsymbol{L}\) and \(\boldsymbol{R}\) with \(\boldsymbol{W}\) gives the complete update rule

\[
\boldsymbol{W} \quad \leftarrow\quad \boldsymbol{L}\boldsymbol{W}\boldsymbol{R}
\tag{7}
\]

This is "Rotational Muon" (Muon under Rotation, MuonR), derived through instantaneous reparameterization. In practice, momentum is usually present as well. We interpret it as a smoothed gradient, so we only need to replace the gradient \(\boldsymbol{G}\) with the momentum \(\boldsymbol{M}\).

## Some Details

Because updating \(\boldsymbol{L}\) and \(\boldsymbol{R}\) each requires one \(\operatorname{msign}\) computation, MuonR takes twice as much computation as Muon even in the ideal case \(n=m\). For sufficiently large models, however, doubling this computation has only a minor effect on end-to-end training time and is usually acceptable. To reduce this overhead, one could update \(\boldsymbol{L}\) and \(\boldsymbol{R}\) alternately, spreading the computation across steps.

In fact, MuonR's biggest problem is that it keeps every singular value of the matrix unchanged from beginning to end. This means that we must determine the full singular-value spectrum of every parameter at initialization. That is not easy, because matrices in different positions may require different scales, and forcing them all to use the same set of values will most likely be suboptimal.

One feasible approach is to begin with a suitable random initialization and add, before or after each matrix, a vector that is multiplied elementwise to restore freedom over the scale. For matrices immediately following RMSNorm, RMSNorm's own gamma parameter already plays this role, so this extra operation can be omitted for those matrices.

As for how to choose the initial singular values, we can use a conventional random initialization or construct them according to [Zipf's law](https://en.wikipedia.org/wiki/Zipf%27s_law). Going further, we could try adjusting the singular-value entropy to the optimum calculated in [*Is Higher Singular-Value Entropy Always Better for Matrix Parameters?*](https://kexue.fm/archives/11767), in the hope of achieving better results.

Of course, if we really can determine a matrix's singular values in advance—for example, if we already want a parameter somewhere in the model to remain orthogonal throughout training—then none of this needs to be considered, and we can simply apply MuonR directly.

## Switching Midway

Another option is to "switch midway," using MuonR only as a means of maintaining stability.

Specifically, we begin with conventional Muon and monitor the matrix's spectral norm or Frobenius norm. Once the matrix norm exceeds our desired range, we switch to MuonR. Since both forms of Muon depend on the same gradient or momentum and differ only in their computation, this switch is valid. MuonR does not change the matrix's singular values, so neither its spectral norm nor its Frobenius norm will continue to grow, making it a natural stabilizing measure.

However, we should align the update magnitudes before and after switching as closely as possible to avoid introducing a sudden jump. To do so, consider the first-order approximation of MuonR:

\[
\boldsymbol{L}\boldsymbol{W}\boldsymbol{R} \approx (\boldsymbol{I} - \eta\boldsymbol{O}_L) \boldsymbol{W} (\boldsymbol{I} - \eta\boldsymbol{O}_R) \approx \boldsymbol{W} - \eta(\boldsymbol{O}_L \boldsymbol{W} + \boldsymbol{W} \boldsymbol{O}_R)
\tag{8}
\]

The singular values of \(\boldsymbol{O}_L\) and \(\boldsymbol{O}_R\) are no greater than 1. Note that we cannot guarantee that both \([\boldsymbol{G}\boldsymbol{W}^{\top}]_{\text{skew}}\) and \([\boldsymbol{W}^{\top}\boldsymbol{G}]_{\text{skew}}\) are full rank, so we cannot directly use the orthogonality of \(\boldsymbol{O}_L\) and \(\boldsymbol{O}_R\). Therefore, \(\Vert\boldsymbol{O}_L \boldsymbol{W}\Vert_F \leq \Vert \boldsymbol{W}\Vert_F\) and \(\Vert \boldsymbol{W}\boldsymbol{O}_R\Vert_F\leq \Vert\boldsymbol{W}\Vert_F\), giving

\[
\Vert\boldsymbol{O}_L \boldsymbol{W} + \boldsymbol{W} \boldsymbol{O}_R\Vert_F \leq \Vert\boldsymbol{O}_L \boldsymbol{W}\Vert_F + \Vert\boldsymbol{W} \boldsymbol{O}_R\Vert_F \leq 2\Vert\boldsymbol{W}\Vert_F
\tag{9}
\]

Conventional Muon uses \(\boldsymbol{W} - \eta \operatorname{msign}(\boldsymbol{G})\), and the Frobenius norm of \(\operatorname{msign}(\boldsymbol{G})\) is generally \(\sqrt{\min(n,m)}\). Therefore, to align the Frobenius norm of the update when switching from Muon to MuonR, the learning rate should be multiplied by approximately \(\frac{\sqrt{\min(n,m)}}{2\Vert\boldsymbol{W}\Vert_F}\).

In practice, the first inequality above may not be tight enough. \(\boldsymbol{O}_L \boldsymbol{W}\) and \(\boldsymbol{W} \boldsymbol{O}_R\) are more nearly orthogonal to each other, so by the Pythagorean theorem the result should be approximately \(\sqrt{2}\Vert\boldsymbol{W}\Vert_F\), and the multiplier should therefore include an additional factor of \(\sqrt{2}\). However, since the difference between \(\sqrt{2}\) and \(1\) is not especially large, and to ensure usability in extreme cases, I recommend retaining the form above.

## Comparison and Contrast

As stated at the beginning, MuonR was inspired by [Pion](https://papers.cool/arxiv/2605.12492). Let us now examine their similarities and differences.

First, the idea of restricting the update rule to a two-sided rotation consisting of left and right multiplication by orthogonal matrices comes mainly from Pion. Once this update form has been fixed, obtaining the corresponding gradients through instantaneous reparameterization is a fairly natural step. From there, Pion and MuonR begin to go their separate ways:

> 1. Pion achieves orthogonality through the matrix exponential \(\exp(\text{skew-symmetric matrix})\), approximated in practice by a second-order expansion.
>
> 2. Pion follows the Adam route and separately takes moving averages of the gradients of \(\boldsymbol{L}\) and \(\boldsymbol{R}\), bringing its cached state to four groups.
>
> 3. MuonR follows the Muon route and, like Muon, caches only momentum, allowing us to switch between Muon and MuonR at any time.
>
> 4. MuonR is based on the analytic solution for steepest descent on the orthogonal manifold, requiring only a finite number of extra steps to achieve orthogonality exactly.

Overall, Pion's orthogonality design is relatively heuristic, and its four groups of cached variables are somewhat daunting. MuonR, by contrast, is a comparatively natural product of the line of work on Muon and steepest descent on the orthogonal manifold. In my view, its overall design adheres more closely to first principles.

## Summary

This post proposed MuonR, a Muon variant whose update is constrained to left and right rotation matrices. It can keep a matrix's singular-value distribution unchanged, providing a simple training scheme for maintaining stability.

<hr class="section-divider">

*Citation: Su, J. (2026, June 8). 流形上的最速下降：6. Muon + 双旋转 [Steepest Descent on Manifolds: 6. Muon + Two-Sided Rotation]. Scientific Spaces. [https://kexue.fm/archives/11777](https://kexue.fm/archives/11777)*

*Original content licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). This translation is shared under the same license.*
