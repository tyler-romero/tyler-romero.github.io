---
title: "Slow Is Fast: 2. Does Adam Also End in Muon?"
subtitle: "Translated from [慢即是快：2. Adam的尽头还是Muon？](https://kexue.fm/archives/11964) by Jianlin Su (苏剑林)"
date: 2026-10-15T21:28:00+08:00
blurb: "Extending orthogonal slowest descent from Kronecker-product preconditioners to diagonal ones (the RMSProp/Adam family) shows that the preconditioner's basis is again washed out, so the update shares the gradient's singular vectors and differs from Muon at most in its singular values, which heuristically tend toward uniformity."
tags: ["translation", "muon", "adam", "optimization", "preconditioning"]
math: true
---

*Translator's note (Opus 5.5): This is an English translation of [慢即是快：2. Adam的尽头还是Muon？](https://kexue.fm/archives/11964) by Jianlin Su (苏剑林), originally published on October 15, 2026 on [Scientific Spaces (科学空间)](https://kexue.fm). The translation preserves the author's first-person voice and is published with permission for educational use.*

<hr class="section-divider">

In the previous post, [*Slow Is Fast: 1. Is Muon Where Shampoo Ends Up?*](/translations/scientific-spaces/slow-is-fast-1-is-muon-where-shampoo-ends-up/), we started from the average-convergence results of convex optimization, built the "orthogonal slowest descent" framework, and used it to improve preconditioned optimizers of arbitrary Kronecker-product form. It turned out that, once optimized under this framework, all of these optimizers showed signs of tending toward Muon, which led me to the pessimistic lament that "Shampoo ends in Muon."

Besides the Kronecker product, another, simpler structured approximation of the preconditioning matrix is the diagonal approximation, with common examples being RMSProp, Adam, and so on. Note that although the diagonal approximation is computationally simpler than the Kronecker product, neither one contains the other, so a new question arises: if we apply orthogonal slowest descent to this class of optimizers, do we get any new results?

## Objective Analysis

Following the notation of the previous post, let the gradient be \(\boldsymbol{G}\in\mathbb{R}^{n\times m},\boldsymbol{g}=\mathop{\text{vec}}(\boldsymbol{G})\in\mathbb{R}^{nm}\), and introduce a matrix \(\boldsymbol{N}\in\mathbb{R}_+^{n\times m}\). The so-called diagonal approximation means that, in the vectorized formulation, the preconditioning matrix is a diagonal matrix, i.e., \(\boldsymbol{H}^{-1}=\operatorname{diag}(\mathop{\text{vec}}(\boldsymbol{N}))\), so that
\[
\boldsymbol{H}^{-1} \boldsymbol{g} = \mathop{\text{vec}}(\boldsymbol{N}\odot\boldsymbol{G})
\tag{1}
\]
or, put another way, the update rule is \(\boldsymbol{W} - \eta \boldsymbol{N}\odot\boldsymbol{G}\), where \(\odot\) is the Hadamard product. What we want to do is introduce an orthogonal transformation of Kronecker-product form for \(\boldsymbol{N}\), so that, under the principle of orthogonal slowest descent, it becomes a better optimizer.

Specifically, we introduce orthogonal matrices \(\boldsymbol{X}\in\mathbb{R}^{n\times n},\boldsymbol{Y}\in\mathbb{R}^{m\times m}\), let \(\boldsymbol{O} = \boldsymbol{Y}\otimes\boldsymbol{X}\), and replace the diagonal preconditioning matrix with \(\boldsymbol{O}^{\top}\operatorname{diag}(\mathop{\text{vec}}(\boldsymbol{N}))\boldsymbol{O}\). That is, the update changes from \(\boldsymbol{N}\odot\boldsymbol{G}\) to \(\boldsymbol{X}^{\top}(\boldsymbol{N}\odot\boldsymbol{X}\boldsymbol{G}\boldsymbol{Y}^{\top})\boldsymbol{Y}\). This transformation requires a bit of background on Kronecker products, but I won't expand on it here.

Intuitively, this amounts to finding a new orthogonal basis for \(\boldsymbol{G}\), applying diagonal preconditioning in that basis, and then transforming back. According to the principle of orthogonal slowest descent, we want to minimize its inner product with \(\boldsymbol{G}\), i.e.,
\[
\min_{\boldsymbol{X},\boldsymbol{Y}} \operatorname{tr}(\boldsymbol{G}^{\top}\boldsymbol{X}^{\top}(\boldsymbol{N}\odot\boldsymbol{X}\boldsymbol{G}\boldsymbol{Y}^{\top})\boldsymbol{Y})\qquad\text{s.t.}\qquad \boldsymbol{X}^{\top}\boldsymbol{X} = \boldsymbol{I}_n,\quad \boldsymbol{Y}^{\top}\boldsymbol{Y} = \boldsymbol{I}_m
\tag{2}
\]

## A Unified Form

Note that
\[
\operatorname{tr}(\boldsymbol{G}^{\top}\boldsymbol{X}^{\top}(\boldsymbol{N}\odot\boldsymbol{X}\boldsymbol{G}\boldsymbol{Y}^{\top})\boldsymbol{Y}) = \operatorname{tr}(\boldsymbol{Y}\boldsymbol{G}^{\top}\boldsymbol{X}^{\top}(\boldsymbol{N}\odot\boldsymbol{X}\boldsymbol{G}\boldsymbol{Y}^{\top})) = \Vert\boldsymbol{N}^{\odot/2}\odot\boldsymbol{X}\boldsymbol{G}\boldsymbol{Y}^{\top}\Vert_F^2
\tag{3}
\]
where \(\boldsymbol{N}^{\odot/2}\) denotes the square root in the Hadamard-product sense, i.e., the element-wise square root. In addition, in the previous post we solved
\[
\min_{\boldsymbol{X},\boldsymbol{Y}} \operatorname{tr}(\boldsymbol{G}^{\top}\boldsymbol{X}^{\top}\boldsymbol{P}\boldsymbol{X}\boldsymbol{G}\boldsymbol{Y}^{\top}\boldsymbol{Q}\boldsymbol{Y})\qquad\text{s.t.}\qquad \boldsymbol{X}^{\top}\boldsymbol{X} = \boldsymbol{I}_n,\quad \boldsymbol{Y}^{\top}\boldsymbol{Y} = \boldsymbol{I}_m
\tag{4}
\]
whose objective function can be rewritten as
\[
\operatorname{tr}(\boldsymbol{G}^{\top}\boldsymbol{X}^{\top}\boldsymbol{P}\boldsymbol{X}\boldsymbol{G}\boldsymbol{Y}^{\top}\boldsymbol{Q}\boldsymbol{Y}) = \operatorname{tr}(\boldsymbol{Y}\boldsymbol{G}^{\top}\boldsymbol{X}^{\top}\boldsymbol{P}\boldsymbol{X}\boldsymbol{G}\boldsymbol{Y}^{\top}\boldsymbol{Q}) = \Vert\boldsymbol{P}^{1/2}\boldsymbol{X}\boldsymbol{G}\boldsymbol{Y}^{\top}\boldsymbol{Q}^{1/2}\Vert_F^2
\tag{5}
\]
We can see that the objective functions of both problems take the form \(f(\boldsymbol{X} \boldsymbol{G} \boldsymbol{Y}^{\top})\), so next we consider, in general,
\[
\min_{\boldsymbol{X}, \boldsymbol{Y}} f(\boldsymbol{X} \boldsymbol{G} \boldsymbol{Y}^{\top}) \qquad\text{s.t.}\qquad \boldsymbol{X}^{\top}\boldsymbol{X} = \boldsymbol{I}_n,\quad \boldsymbol{Y}^{\top}\boldsymbol{Y} = \boldsymbol{I}_m
\tag{6}
\]

## Stationarity Conditions

First, let us find the stationarity conditions of the above problem. To this end, we introduce Lagrange multipliers \(\boldsymbol{A}\in\mathbb{R}^{n\times n},\boldsymbol{B}\in\mathbb{R}^{m\times m}\) and define the auxiliary function
\[
F = f(\boldsymbol{X} \boldsymbol{G} \boldsymbol{Y}^{\top}) - \operatorname{tr}(\boldsymbol{A}^{\top}(\boldsymbol{X}^{\top}\boldsymbol{X} - \boldsymbol{I}_n)) - \operatorname{tr}(\boldsymbol{B}^{\top}(\boldsymbol{Y}^{\top}\boldsymbol{Y} - \boldsymbol{I}_m))
\tag{7}
\]
Let \(\boldsymbol{Z} = \boldsymbol{X} \boldsymbol{G} \boldsymbol{Y}^{\top}, \boldsymbol{K} = \nabla_{\boldsymbol{Z}} f(\boldsymbol{Z})\). Taking the gradients of \(F\) with respect to \(\boldsymbol{X}\) and \(\boldsymbol{Y}\) and setting them to zero gives
\[
\nabla_{\boldsymbol{X}} F = \boldsymbol{K} \boldsymbol{Y} \boldsymbol{G}^{\top} - \boldsymbol{X}(\boldsymbol{A} + \boldsymbol{A}^{\top}) = \boldsymbol{0}
\tag{8}
\]
\[
\nabla_{\boldsymbol{Y}} F = \boldsymbol{K}^{\top} \boldsymbol{X} \boldsymbol{G} - \boldsymbol{Y}(\boldsymbol{B} + \boldsymbol{B}^{\top}) = \boldsymbol{0}
\tag{9}
\]
Multiplying both sides of the first equation by \(\boldsymbol{X}^{\top}\) and both sides of the second by \(\boldsymbol{Y}^{\top}\), and using the orthogonality of \(\boldsymbol{X},\boldsymbol{Y}\), we obtain the stationarity conditions
\[
\boldsymbol{X}^{\top}\boldsymbol{K}\boldsymbol{Y}\boldsymbol{G}^{\top}\,\,\text{symmetric},\qquad\boldsymbol{G}^{\top}\boldsymbol{X}^{\top}\boldsymbol{K}\boldsymbol{Y}\,\,\text{symmetric}
\tag{10}
\]
In the previous post we already proved that this means \(\boldsymbol{X}^{\top}\boldsymbol{K}\boldsymbol{Y}\) has the form \(\boldsymbol{U}\boldsymbol{D}\boldsymbol{V}^{\top}\), where \(\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}\) is the SVD of \(\boldsymbol{G}\) and \(\boldsymbol{D}\) is a diagonal matrix. This is a common feature of the optimal solutions of the above optimization problem, independent of the specific choice of \(f\), which makes it quite a strong conclusion.

## Washing Out the Basis

Let us first use this conclusion to revisit the objective of the previous post, \(\Vert\boldsymbol{P}^{1/2}\boldsymbol{X}\boldsymbol{G}\boldsymbol{Y}^{\top}\boldsymbol{Q}^{1/2}\Vert_F^2\). Here \(\boldsymbol{K} = 2\boldsymbol{P}\boldsymbol{X}\boldsymbol{G}\boldsymbol{Y}^{\top}\boldsymbol{Q}\), and the update we care about is \(\boldsymbol{\Phi} = \boldsymbol{X}^{\top}\boldsymbol{P}\boldsymbol{X}\boldsymbol{G}\boldsymbol{Y}^{\top}\boldsymbol{Q}\boldsymbol{Y} = \boldsymbol{X}^{\top}\boldsymbol{K}\boldsymbol{Y}/2\). By the result of the previous section, \(\boldsymbol{\Phi}\) has the form \(\boldsymbol{U}\boldsymbol{D}\boldsymbol{V}^{\top}\); that is, the eigenvector matrices of \(\boldsymbol{P},\boldsymbol{Q}\) are completely eliminated.

What surprised me greatly is that the same conclusion also generalizes to the protagonist of this post, namely the objective \(\Vert\boldsymbol{N}^{\odot/2}\odot\boldsymbol{X}\boldsymbol{G}\boldsymbol{Y}^{\top}\Vert_F^2\). Here \(\boldsymbol{K} = 2\boldsymbol{N}\odot\boldsymbol{X}\boldsymbol{G}\boldsymbol{Y}^{\top}\), and the update we care about is \(\boldsymbol{\Phi} = \boldsymbol{X}^{\top}(\boldsymbol{N}\odot\boldsymbol{X}\boldsymbol{G}\boldsymbol{Y}^{\top})\boldsymbol{Y} = \boldsymbol{X}^{\top}\boldsymbol{K}\boldsymbol{Y}/2\), so, again by the result on the stationarity conditions, \(\boldsymbol{\Phi}\) has the form \(\boldsymbol{U}\boldsymbol{D}\boldsymbol{V}^{\top}\).

In other words, whether we use a Kronecker product or a diagonal approximation, the "basis" of the preconditioning matrix is washed out by orthogonal slowest descent: the left and right singular matrices of the update uniformly degenerate into the left and right singular matrices of the gradient (or momentum), and the most the preconditioning can change is the singular values. This is exactly how Muon is constructed. This is the first sense of "ends in Muon" in the title.

## Can It Be Uniform?

If we could prove that the diagonal entries of \(\boldsymbol{D}\) tend toward balance, the result would converge even further with Muon. This is the second sense of "ends in Muon."

In the previous post we gave a detailed argument for the Kronecker-product case, but that process does not generalize easily, so here we instead use a heuristic approach to get a sense of the properties of \(\boldsymbol{D}\). Without loss of generality, assume \(n\geq m\). We have
\[
\operatorname{tr}(\boldsymbol{G}^{\top}\boldsymbol{\Phi}) = \operatorname{tr}(\boldsymbol{V}\boldsymbol{\Sigma}^{\top} \boldsymbol{D} \boldsymbol{V}^{\top}) = \operatorname{tr}(\boldsymbol{\Sigma}^{\top} \boldsymbol{D}) = \sum_{i=1}^m \sigma_i d_i \geq m\left(\prod_{i=1}^m \sigma_i d_i\right)^{1/m} = m\left(\prod_{i=1}^m \sigma_i \prod_{i=1}^m d_i\right)^{1/m}
\tag{11}
\]
where \(\boldsymbol{\Sigma}=\operatorname{diag}(\sigma_1,\sigma_2,\cdots),\boldsymbol{D}=\operatorname{diag}(d_1,d_2,\cdots)\), and equality holds when \(d_i\propto \sigma_i^{-1}\). This result shows that, as long as the product of the singular values of \(\boldsymbol{\Phi}\) is constant, the ideal minimum of \(\operatorname{tr}(\boldsymbol{G}^{\top}\boldsymbol{\Phi})\) is attained at \(\boldsymbol{D}\propto \boldsymbol{\Sigma}^{-1}\) (taking the reciprocal of each diagonal entry while keeping the shape unchanged).

Of course, since we can only change the orthogonal matrices in \(\boldsymbol{\Phi}\), this ideal minimum is hard to achieve. We can only guess that the result lies somewhere between the most ideal case (\(\boldsymbol{D}\propto \boldsymbol{\Sigma}^{-1}\)) and the worst case (\(\boldsymbol{D}\propto \boldsymbol{\Sigma}\)), and the geometric mean of the two is precisely \(\boldsymbol{D}\propto \boldsymbol{I}\), i.e., uniform singular values.

## Product of the Spectrum

So is the product of the singular values of \(\boldsymbol{\Phi}\) constant? Again, we discuss the Kronecker-product case first and the diagonal approximation second. First, for the "Kronecker product + square matrix" case, it is indeed constant:
\[
\prod_{i=1}^m d_i = |\det(\boldsymbol{\Phi})| = |\det(\boldsymbol{X}^{\top}\boldsymbol{P}\boldsymbol{X}\boldsymbol{G}\boldsymbol{Y}^{\top}\boldsymbol{Q}\boldsymbol{Y})| = |\det(\boldsymbol{P})\det(\boldsymbol{G})\det(\boldsymbol{Q})|
\tag{12}
\]
That is, under transformations by the orthogonal matrices \(\boldsymbol{X},\boldsymbol{Y}\), the absolute value of the determinant of \(\boldsymbol{\Phi}\) stays unchanged, so the product of its singular values is unchanged, and the conclusion of the previous section applies. For non-square matrices, however, the above identity no longer holds. For this case we have two options: one is to discuss it carefully following the method of the previous post; the other is to assume, without proof, that the above equation also holds **approximately** for non-square matrices.

The diagonal-approximation case is trickier. Although the Hadamard product is simple, it is not an intrinsic matrix operation, and introducing it makes all kinds of matrix analysis exceptionally difficult. So, whether or not the matrix is square, we cannot make an effective estimate of the singular values/determinant of \(\boldsymbol{\Phi} = \boldsymbol{X}^{\top}(\boldsymbol{N}\odot\boldsymbol{X}\boldsymbol{G}\boldsymbol{Y}^{\top})\boldsymbol{Y}\).

However, since the only variable parts of \(\boldsymbol{\Phi}\) are orthogonal matrices, and orthogonal transformations often do not change the determinant, we conjecture that "the determinant of \(\boldsymbol{\Phi}\) will not change significantly either," from which it follows that \(\boldsymbol{D}\) also tends toward uniformity. Note that this conjecture is quite "weak": it certainly does not hold identically, and there is no way to judge how good the approximation is. It is more of a "faith" we hold about determinants and orthogonal matrices.

## Summary

This post extended "orthogonal slowest descent" to diagonal preconditioning (the RMSProp and Adam family). By distilling an abstract form common to the two objectives, we arrived at the unified conclusion that "whether with a Kronecker product or a diagonal approximation, orthogonal slowest descent eliminates the original basis and changes at most the singular value spectrum of the gradient or momentum." Furthermore, we briefly discussed whether the singular values tend toward balance.

<hr class="section-divider">

*Citation: Su, J. (2026, October 15). 慢即是快：2. Adam的尽头还是Muon？ [Slow Is Fast: 2. Does Adam Also End in Muon?]. Scientific Spaces. [https://kexue.fm/archives/11964](https://kexue.fm/archives/11964)*

*Original content licensed under [CC BY-NC-ND 2.5 CN](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/). This translation is published with the author's permission for educational use.*