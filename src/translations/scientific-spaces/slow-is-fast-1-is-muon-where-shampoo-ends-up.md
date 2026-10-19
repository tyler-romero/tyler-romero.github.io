---
title: "Slow Is Fast: 1. Is Muon Where Shampoo Ends Up?"
subtitle: "Translated from [慢即是快：1. Shampoo的尽头是Muon？](https://kexue.fm/archives/11917) by Jianlin Su (苏剑林)"
date: 2026-09-28T17:00:00+08:00
blurb: "Starting from the average convergence bound of convex optimization, we propose the principle of orthogonal slowest descent and show that, for Kronecker-product preconditioners such as the Shampoo family, it washes out the preconditioner's eigenvectors and equalizes the gradient's singular values, pointing straight to Muon."
tags: ["translation", "muon", "shampoo", "optimization", "preconditioning"]
math: true
---

*Translator's note (Opus 5.5): This is an English translation of [慢即是快：1. Shampoo的尽头是Muon？](https://kexue.fm/archives/11917) by Jianlin Su (苏剑林), originally published on September 28, 2026 on [Scientific Spaces (科学空间)](https://kexue.fm). It is the first article in the "Slow Is Fast" (慢即是快) series. The translation preserves the author's first-person voice and is published with permission for educational use.*

<hr class="section-divider">

As everyone knows, the [Muon](https://kexue.fm/tag/muon/) optimizer has been popular for quite a while now. Although it can't be said to have completely replaced Adam, it has already become a highly competitive optimizer that can no longer be ignored. Meanwhile, over the past two years the Shampoo family of optimizers has also refused to be left behind (不甘落后): besides the original Shampoo, it has evolved into variants such as [KL-Shampoo](https://papers.cool/arxiv/2509.03378), [OKLS](https://blog.tilderesearch.com/blog/online-kl-shampoo), [PSGD](https://github.com/lixilinx/psgd_torch), and [SOAP](https://papers.cool/arxiv/2409.11321), all of which claim to achieve better results than Muon.

However, this post is going to pour a bucket of cold water on that: for all Shampoo-like, Kronecker-product-based preconditioned optimizers, the ideal limit may well be Muon — the preconditioning matrices we pay such a high price to build may all be "wasted effort" (无用功).

## Average Convergence

We start from the average convergence result obtained in [*Making Alchemy More Scientific (Part 10): Dismantling and Patching the Monotonicity Assumption*](https://kexue.fm/archives/11885): under the convexity assumption, the update rule \(\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \boldsymbol{H}_t^{-1}\boldsymbol{g}_t\) satisfies the following inequality
\[
\frac{\sum_{t=1}^T \eta_t \mathbb{E}[L(\boldsymbol{\theta}_t) - L(\boldsymbol{\theta}^*)]}{\sum_{t=1}^T \eta_t} \leq \mathbb{E}\left[\frac{R^2 \Lambda_T}{2\sum_{t=1}^T \eta_t} + \frac{\sum_{t=1}^T \eta_t^2 \Vert\boldsymbol{g}_t\Vert_{\boldsymbol{H}_t^{-1}}^2}{2\sum_{t=1}^T \eta_t}\right]
\tag{1}
\]
where
\[
\Lambda_T \triangleq \lambda_{\max}(\boldsymbol{H}_1) + \sum_{t=2}^T [\lambda_{\max}(\boldsymbol{H}_t - \boldsymbol{H}_{t-1})]_+
\tag{2}
\]
I won't go into the meaning of the other notation here; please refer to the original post. Intuitively, this convergence upper bound consists of two terms: the first represents how fast we descend, and the second represents how violently the convergence process jitters. In actual training, we need to choose appropriate \(\eta_t\) and \(\boldsymbol{H}_t\) to balance the sizes of the two terms and make the bound as small as possible.

In other words, we assume: **a better optimizer can be obtained by minimizing the convergence upper bound.** There is already a precedent for this approach: the post [*Making Alchemy More Scientific (Part 9): Classic Adaptive Gradient Algorithms*](https://kexue.fm/archives/11882), based on the assumptions that \(\boldsymbol{H}_t\succeq \boldsymbol{H}_{t-1}\) and that \(\eta_t\) is constant, replaced \(\Lambda_T\) with its upper bound \(\operatorname{tr}(\boldsymbol{H}_T)\), then minimized the bound to obtain \(\boldsymbol{H}^* = \left(\sum_{t=1}^T  \boldsymbol{g}_t\boldsymbol{g}_t^{\top}\right)^{1/2}\), thereby leading to the AdaGrad optimizer.

This assumption is also the core starting point that runs through all the derivations in this post.

## Slowest Descent

Now let's switch perspectives: suppose we have a ready-made, fixed matrix \(\boldsymbol{H}_t\), and we transform it into \(\boldsymbol{O}_t^{\top}\boldsymbol{H}_t\boldsymbol{O}_t\), where \(\boldsymbol{O}_t\) is an orthogonal matrix. We hope to choose an appropriate \(\boldsymbol{O}_t\) such that
\[
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t (\boldsymbol{O}_t^{\top}\boldsymbol{H}_t\boldsymbol{O}_t)^{-1}\boldsymbol{g}_t
\tag{3}
\]
becomes a better optimizer. How do we judge whether it is better? Following the viewpoint of the previous section, we want it to lower the convergence upper bound. Note that orthogonal transformations do not change eigenvalues, so \(\lambda_{\max}(\boldsymbol{O}_1^{\top}\boldsymbol{H}_1\boldsymbol{O}_1)=\lambda_{\max}(\boldsymbol{H}_1)\). Then, assuming that \(\boldsymbol{O}_t\) varies slowly with \(t\), we have
\[
\lambda_{\max}(\boldsymbol{O}_t^{\top}\boldsymbol{H}_t\boldsymbol{O}_t - \boldsymbol{O}_{t-1}^{\top}\boldsymbol{H}_{t-1}\boldsymbol{O}_{t-1}) \approx \lambda_{\max}(\boldsymbol{O}_t^{\top}\boldsymbol{H}_t\boldsymbol{O}_t - \boldsymbol{O}_t^{\top}\boldsymbol{H}_{t-1}\boldsymbol{O}_t) = \lambda_{\max}(\boldsymbol{H}_t - \boldsymbol{H}_{t-1})
\tag{4}
\]
As a result, \(\Lambda_T\) is approximately invariant under the substitution \(\boldsymbol{H}_t\to \boldsymbol{O}_t^{\top}\boldsymbol{H}_t\boldsymbol{O}_t\). So if we only want to optimize \(\boldsymbol{O}_t\), the objective function reduces to just the second term
\[
\frac{\sum_{t=1}^T \eta_t^2 \Vert\boldsymbol{g}_t\Vert_{(\boldsymbol{O}_t^{\top}\boldsymbol{H}_t\boldsymbol{O}_t)^{-1}}^2}{2\sum_{t=1}^T \eta_t}
\tag{5}
\]
Since the \(\boldsymbol{O}_t\) for each \(t\) can be different, we can perfectly well minimize \(\Vert\boldsymbol{g}_t\Vert_{(\boldsymbol{O}_t^{\top}\boldsymbol{H}_t\boldsymbol{O}_t)^{-1}}^2\) pointwise, which can in turn be written as
\[
\min_{\boldsymbol{O}_t}\langle\boldsymbol{g}_t, (\boldsymbol{O}_t^{\top}\boldsymbol{H}_t\boldsymbol{O}_t)^{-1}\boldsymbol{g}_t\rangle\qquad\text{s.t.}\qquad \boldsymbol{O}_t^{\top}\boldsymbol{O}_t = \boldsymbol{I}
\tag{6}
\]
Here \(\boldsymbol{\varphi}_t =(\boldsymbol{O}_t^{\top}\boldsymbol{H}_t\boldsymbol{O}_t)^{-1}\boldsymbol{g}_t\) is the optimizer's update direction. We know that \(\mathcal{L}(\boldsymbol{\theta} - \eta_t \boldsymbol{\varphi}_t)\approx \mathcal{L}(\boldsymbol{\theta}) - \eta_t\langle\boldsymbol{g}_t,\boldsymbol{\varphi}_t\rangle\), so another meaning of the above problem is: under the first-order approximation, minimize the amount by which the loss decreases at each step — that is, slowest descent.

## Fast vs. Slow

Readers who have read posts such as [*Muon Sequel: Why We Chose to Try Muon?*](https://kexue.fm/archives/10739), [*Steepest Descent on Manifolds: 1. SGD + Hypersphere*](https://kexue.fm/archives/11196), and [*Beyond MuP: 2. Linear Layers and Steepest Descent*](/translations/scientific-spaces/beyond-mup-2-linear-layers-and-steepest-descent/) should know that many optimizers are derived from the principle of steepest descent (i.e., fastest descent). And now along comes slowest descent — don't the two conflict?

In fact, they represent precisely two completely different ways of looking at optimizers. Steepest descent is a local, greedy solution under a first-order approximation: it hopes that taking the optimal step at each moment will yield a globally good solution, and it represents an optimistic assumption. Minimizing the convergence upper bound, on the other hand, represents a global, pessimistic assumption: it hopes that even in the worst case, we can still achieve some kind of convergence guarantee.

The reason the latter happens to take the counterintuitive form of slowest descent is that we have restricted the optimization space to orthogonal transformations, so the first term of the convergence upper bound is approximately negligible, leaving only the second term, whose meaning is slowest descent. This also reminds us that slowest descent cannot be considered for every optimization space; it is merely a special case for the orthogonal space, and its feasibility in other situations needs to be analyzed case by case.

Here, the role of slowest descent is to make the optimization trajectory smoother while still guaranteeing descent, so that a larger learning rate can be used, or so that at the same learning rate we advance faster and more accurately. In the language of steepest descent, we would instead need to build our assumptions about stability into the constraint on the update (for example, Muon chooses the spectral norm).

In addition, in stochastic optimization this term is also related to noise — if we decompose \(\boldsymbol{g}_t\) into "true value + random noise", then \(\mathbb{E}[\Vert\boldsymbol{g}_t\Vert_{\boldsymbol{H}_t^{-1}}^2]\) will contain a variance term — so slowest descent also means reducing the influence of noise on the optimization process, making the optimization more accurate.

## Matrix Optimization

The preceding sections discussed general principles; now let's formally carry out the derivation for matrix optimization. Consider a matrix parameter \(\boldsymbol{W}\in\mathbb{R}^{n\times m}\) with gradient \(\boldsymbol{G}\), whose flattened vector is \(\boldsymbol{g}=\operatorname{vec}(\boldsymbol{G})\in\mathbb{R}^{nm}\). Here \(\boldsymbol{H}\) is an \(nm\times nm\) matrix, which is rather laborious to think about.

For the matrix setting, we consider preconditioners that can be expressed as Kronecker products. However, we won't introduce the language of Kronecker products here; instead, we understand things directly from the perspective of linear operators, i.e., we directly interpret \(\boldsymbol{H}^{-1}\boldsymbol{g}\) as
\[
\boldsymbol{H}^{-1}\boldsymbol{g} = \operatorname{vec}(\boldsymbol{P}\boldsymbol{G}\boldsymbol{Q})
\tag{7}
\]
That is, the optimizer's update rule is \(\boldsymbol{W} - \eta\boldsymbol{P}\boldsymbol{G}\boldsymbol{Q}\) (written with Kronecker products, \(\boldsymbol{H}^{-1} = \boldsymbol{Q}\otimes\boldsymbol{P}\)), where \(\boldsymbol{P}\in\mathbb{R}^{n\times n},\boldsymbol{Q}\in\mathbb{R}^{m\times m}\) are two positive semidefinite preconditioning matrices. \(\boldsymbol{P},\boldsymbol{Q}\) may depend on the current gradient or even on historical gradients, so this covers Muon and the various Shampoo variants (in their momentum-free versions).

Now fix the matrices \(\boldsymbol{P},\boldsymbol{G},\boldsymbol{Q}\). We hope to choose appropriate orthogonal matrices \(\boldsymbol{X}\in\mathbb{R}^{n\times n}\) and \(\boldsymbol{Y}\in\mathbb{R}^{m\times m}\) such that the new update \(\boldsymbol{X}^{\top}\boldsymbol{P}\boldsymbol{X}\boldsymbol{G}\boldsymbol{Y}^{\top}\boldsymbol{Q}\boldsymbol{Y}\) becomes a better optimizer (written with Kronecker products, \(\boldsymbol{O} = \boldsymbol{Y}\otimes\boldsymbol{X}\)). The guiding principle is exactly the "orthogonal slowest descent" discussed in the preceding sections, namely
\[
\min_{\boldsymbol{X},\boldsymbol{Y}} \operatorname{tr}(\boldsymbol{G}^{\top}\boldsymbol{X}^{\top}\boldsymbol{P}\boldsymbol{X}\boldsymbol{G}\boldsymbol{Y}^{\top}\boldsymbol{Q}\boldsymbol{Y})\qquad\text{s.t.}\qquad \boldsymbol{X}^{\top}\boldsymbol{X} = \boldsymbol{I}_n,\quad \boldsymbol{Y}^{\top}\boldsymbol{Y} = \boldsymbol{I}_m
\tag{8}
\]

## A First Solution

The full problem is a bit complicated. For simplicity, let's first solve the special case \(\boldsymbol{Q}=\boldsymbol{I}_m\) to get a feel for it. The problem then simplifies to
\[
\min_{\boldsymbol{X}} \operatorname{tr}(\boldsymbol{G}^{\top}\boldsymbol{X}^{\top}\boldsymbol{P}\boldsymbol{X}\boldsymbol{G})\qquad\text{s.t.}\qquad \boldsymbol{X}^{\top}\boldsymbol{X} = \boldsymbol{I}_n
\tag{9}
\]
Note that \(\operatorname{tr}(\boldsymbol{G}^{\top}\boldsymbol{X}^{\top}\boldsymbol{P}\boldsymbol{X}\boldsymbol{G})=\operatorname{tr}(\boldsymbol{G}\boldsymbol{G}^{\top}\boldsymbol{X}^{\top}\boldsymbol{P}\boldsymbol{X})\), so this is exactly the von Neumann trace inequality problem we derived in [*The Rearrangement Inequality and Its Generalizations*](https://kexue.fm/archives/11910). Its minimum is
\[
\sum_{i=1}^n\lambda_i(\boldsymbol{G}\boldsymbol{G}^{\top}) \lambda_{n+1-i}(\boldsymbol{P})
\tag{10}
\]
where \(\lambda_i(\cdot)\) denotes the \(i\)-th largest eigenvalue of the corresponding matrix. Let the SVD of \(\boldsymbol{G}\) be \(\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}\) (the full version, i.e., \(\boldsymbol{U},\boldsymbol{V}\) are both square orthogonal matrices, with the singular values sorted from largest to smallest). Then the eigendecomposition of \(\boldsymbol{G}\boldsymbol{G}^{\top}\) is \(\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{\Sigma}^{\top}\boldsymbol{U}^{\top}\), and the minimizer \(\boldsymbol{X}_*\) satisfies \(\boldsymbol{X}_*^{\top}\boldsymbol{P}\boldsymbol{X}_* = \boldsymbol{U} \boldsymbol{\Lambda}^{\uparrow}_{\boldsymbol{P}}\boldsymbol{U}^{\top}\), where \(\boldsymbol{\Lambda}^{\uparrow}_{\boldsymbol{P}}\) is the diagonal matrix formed by the eigenvalues of \(\boldsymbol{P}\) sorted from smallest to largest. The optimal update is then
\[
\boldsymbol{X}_*^{\top}\boldsymbol{P}\boldsymbol{X}_*\boldsymbol{G} = (\boldsymbol{U} \boldsymbol{\Lambda}^{\uparrow}_{\boldsymbol{P}}\boldsymbol{U}^{\top})( \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}) = \boldsymbol{U} (\boldsymbol{\Lambda}^{\uparrow}_{\boldsymbol{P}}\boldsymbol{\Sigma})\boldsymbol{V}^{\top}
\tag{11}
\]
When I first saw this result, I was quite shocked. It tells us that no matter what the original \(\boldsymbol{P}\) is, after orthogonal slowest descent it can retain at most its eigenvalues, while its eigenvector matrix disappears completely. The final update is the left and right singular matrices of \(\boldsymbol{G}\) sandwiching a diagonal matrix \(\boldsymbol{\Lambda}^{\uparrow}_{\boldsymbol{P}}\boldsymbol{\Sigma}\). This diagonal matrix multiplies the eigenvalues of \(\boldsymbol{P}\) with the singular values of \(\boldsymbol{G}\) in reverse order, which makes all the singular values more uniform — consistent with the idea behind Muon.

This result hints that no matter how expensive our preconditioning is, the best outcome is nothing more than "heading for Muon" (奔赴Muon). Next, we will see that this conclusion still holds in the general case \(\boldsymbol{Q}\neq\boldsymbol{I}_m\).

## The General Case

Let \(\boldsymbol{A} = \boldsymbol{G}^{\top}\boldsymbol{X}^{\top}\boldsymbol{P}\boldsymbol{X}\boldsymbol{G}, \boldsymbol{B} = \boldsymbol{G}\boldsymbol{Y}^{\top}\boldsymbol{Q}\boldsymbol{Y}\boldsymbol{G}^{\top}\). By the invariance properties of the trace, we have
\[
\operatorname{tr}(\boldsymbol{G}^{\top}\boldsymbol{X}^{\top}\boldsymbol{P}\boldsymbol{X}\boldsymbol{G}\boldsymbol{Y}^{\top}\boldsymbol{Q}\boldsymbol{Y}) = \operatorname{tr}(\boldsymbol{A}\boldsymbol{Y}^{\top}\boldsymbol{Q}\boldsymbol{Y}) = \operatorname{tr}(\boldsymbol{B}\boldsymbol{X}^{\top}\boldsymbol{P}\boldsymbol{X})
\tag{12}
\]
Let the optimal solution be \(\boldsymbol{X}_*, \boldsymbol{Y}_*\), with the corresponding \(\boldsymbol{A},\boldsymbol{B}\) denoted \(\boldsymbol{A}_*,\boldsymbol{B}_*\). Fixing \(\boldsymbol{X}_*\) and solving for \(\boldsymbol{Y}_*\), and fixing \(\boldsymbol{Y}_*\) and solving for \(\boldsymbol{X}_*\), respectively, we can use the result of the previous section to write down the stationary-point conditions
\[
\boldsymbol{X}_*^{\top}\boldsymbol{P}\boldsymbol{X}_* = \boldsymbol{U}_{\boldsymbol{B}} \boldsymbol{\Lambda}_{\boldsymbol{P}}^{\uparrow} \boldsymbol{U}_{\boldsymbol{B}}^{\top},\qquad \boldsymbol{Y}_*^{\top}\boldsymbol{Q}\boldsymbol{Y}_* = \boldsymbol{U}_{\boldsymbol{A}} \boldsymbol{\Lambda}_{\boldsymbol{Q}}^{\uparrow} \boldsymbol{U}_{\boldsymbol{A}}^{\top}
\tag{13}
\]
where \(\boldsymbol{U}_{\boldsymbol{A}}\boldsymbol{\Lambda}_{\boldsymbol{A}}^{\downarrow}\boldsymbol{U}_{\boldsymbol{A}}^{\top},\boldsymbol{U}_{\boldsymbol{B}}\boldsymbol{\Lambda}_{\boldsymbol{B}}^{\downarrow}\boldsymbol{U}_{\boldsymbol{B}}^{\top}\) are the eigendecompositions of \(\boldsymbol{A}_*, \boldsymbol{B}_*\), respectively ("\(\downarrow\)" indicates eigenvalues sorted from largest to smallest). Our task is to analyze the properties of \(\boldsymbol{\Phi}=\boldsymbol{X}_*^{\top}\boldsymbol{P}\boldsymbol{X}_*\boldsymbol{G}\boldsymbol{Y}_*^{\top}\boldsymbol{Q}\boldsymbol{Y}_*\). The first conclusion is exactly the same as in the previous section: **\(\boldsymbol{\Phi}\) has the form \(\boldsymbol{U}\boldsymbol{D}\boldsymbol{V}^{\top}\), where \(\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}\) is the SVD of \(\boldsymbol{G}\) and \(\boldsymbol{D}\) is some diagonal matrix.** The proof is not difficult. From the stationary-point conditions, direct computation gives
\[
\boldsymbol{G}^{\top}\boldsymbol{\Phi} = \boldsymbol{U}_{\boldsymbol{A}}(\boldsymbol{\Lambda}_{\boldsymbol{A}}^{\downarrow}\boldsymbol{\Lambda}_{\boldsymbol{Q}}^{\uparrow})\boldsymbol{U}_{\boldsymbol{A}}^{\top},\qquad \boldsymbol{\Phi}\boldsymbol{G}^{\top} = \boldsymbol{U}_{\boldsymbol{B}}(\boldsymbol{\Lambda}_{\boldsymbol{B}}^{\downarrow}\boldsymbol{\Lambda}_{\boldsymbol{P}}^{\uparrow})\boldsymbol{U}_{\boldsymbol{B}}^{\top}
\tag{14}
\]
This shows that \(\boldsymbol{G}^{\top}\boldsymbol{\Phi}\) and \(\boldsymbol{\Phi}\boldsymbol{G}^{\top}\) are both symmetric matrices. Since \(\boldsymbol{U},\boldsymbol{V}\) are both square orthogonal matrices, any \(\boldsymbol{\Phi}\) can be uniquely written as \(\boldsymbol{\Phi}=\boldsymbol{U}\boldsymbol{M}\boldsymbol{V}^{\top}\) (i.e., \(\boldsymbol{M}=\boldsymbol{U}^{\top}\boldsymbol{\Phi}\boldsymbol{V}\)). Substituting gives \(\boldsymbol{G}^{\top}\boldsymbol{\Phi}=\boldsymbol{V}\boldsymbol{\Sigma}^{\top}\boldsymbol{M}\boldsymbol{V}^{\top}\) and \(\boldsymbol{\Phi}\boldsymbol{G}^{\top}=\boldsymbol{U}\boldsymbol{M}\boldsymbol{\Sigma}^{\top}\boldsymbol{U}^{\top}\); their symmetry is equivalent to the symmetry of \(\boldsymbol{\Sigma}^{\top}\boldsymbol{M}\) and \(\boldsymbol{M}\boldsymbol{\Sigma}^{\top}\), respectively. In component form, this reads
\[
\sigma_i M_{i,j} = \sigma_j M_{j,i},\qquad \sigma_j M_{i,j} = \sigma_i M_{j,i}
\tag{15}
\]
From this we get \((\sigma_i^2-\sigma_j^2)M_{i,j}=0\). Assuming the singular values of \(\boldsymbol{G}\) are pairwise distinct (if not, interpret it as the limit of a sequence of matrices), it follows that \(M_{i,j}=0\) for \(i\neq j\), i.e., \(\boldsymbol{M}\) is a diagonal matrix \(\boldsymbol{D}\). This result shows once again that, even in the general case, the eigenvector matrices of the preconditioners are still eliminated by orthogonal slowest descent, leaving at most their eigenvalues.

## Tending Toward Balance

Unfortunately, we cannot simply write down a closed-form solution for \(\boldsymbol{D}\), but we can qualitatively deduce that **the diagonal of \(\boldsymbol{D}\) tends toward balance**. Specifically, from \(\boldsymbol{\Phi} = \boldsymbol{U}\boldsymbol{D}\boldsymbol{V}^{\top}\) we can write
\[
\boldsymbol{G}^{\top}\boldsymbol{\Phi} = \boldsymbol{V}(\boldsymbol{\Sigma}^{\top}\boldsymbol{D})\boldsymbol{V}^{\top},\qquad \boldsymbol{\Phi}\boldsymbol{G}^{\top} = \boldsymbol{U}(\boldsymbol{D}\boldsymbol{\Sigma}^{\top})\boldsymbol{U}^{\top}
\tag{16}
\]
Comparing with Eq. (14), by the uniqueness of eigendecomposition (note: the uniqueness of eigendecomposition relies on the eigenvalues being pairwise distinct; otherwise we could construct counterexamples that make the problem more complicated, but since we are mainly doing a heuristic derivation here, we won't nitpick the details), we find that \(\boldsymbol{U}_{\boldsymbol{A}},\boldsymbol{U}_{\boldsymbol{B}}\) differ from \(\boldsymbol{V},\boldsymbol{U}\) by at most a permutation matrix, so we can write
\[
\boldsymbol{U}_{\boldsymbol{A}} = \boldsymbol{V}\boldsymbol{\Pi}_{\boldsymbol{Q}},\qquad \boldsymbol{U}_{\boldsymbol{B}} = \boldsymbol{U}\boldsymbol{\Pi}_{\boldsymbol{P}}
\tag{17}
\]
where \(\boldsymbol{\Pi}_{\boldsymbol{Q}},\boldsymbol{\Pi}_{\boldsymbol{P}}\) are two permutation matrices. Then
\[
\begin{aligned}
\boldsymbol{\Phi} = \boldsymbol{X}_*^{\top}\boldsymbol{P}\boldsymbol{X}_*\boldsymbol{G}\boldsymbol{Y}_*^{\top}\boldsymbol{Q}\boldsymbol{Y}_* =&\, (\boldsymbol{U}_{\boldsymbol{B}} \boldsymbol{\Lambda}_{\boldsymbol{P}}^{\uparrow} \boldsymbol{U}_{\boldsymbol{B}}^{\top}) \boldsymbol{G}(\boldsymbol{U}_{\boldsymbol{A}} \boldsymbol{\Lambda}_{\boldsymbol{Q}}^{\uparrow} \boldsymbol{U}_{\boldsymbol{A}}^{\top}) \\
=&\, (\boldsymbol{U}\boldsymbol{\Pi}_{\boldsymbol{P}} \boldsymbol{\Lambda}_{\boldsymbol{P}}^{\uparrow} \boldsymbol{\Pi}_{\boldsymbol{P}}^{\top}\boldsymbol{U}^{\top}) (\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}) (\boldsymbol{V}\boldsymbol{\Pi}_{\boldsymbol{Q}} \boldsymbol{\Lambda}_{\boldsymbol{Q}}^{\uparrow} \boldsymbol{\Pi}_{\boldsymbol{Q}}^{\top}\boldsymbol{V}^{\top}) \\
=&\, \boldsymbol{U}\Big((\boldsymbol{\Pi}_{\boldsymbol{P}} \boldsymbol{\Lambda}_{\boldsymbol{P}}^{\uparrow} \boldsymbol{\Pi}_{\boldsymbol{P}}^{\top})\boldsymbol{\Sigma}(\boldsymbol{\Pi}_{\boldsymbol{Q}} \boldsymbol{\Lambda}_{\boldsymbol{Q}}^{\uparrow} \boldsymbol{\Pi}_{\boldsymbol{Q}}^{\top})\Big)\boldsymbol{V}^{\top}
\end{aligned}
\tag{18}
\]
Thus \(\operatorname{tr}(\boldsymbol{G}^{\top}\boldsymbol{\Phi})=\operatorname{tr}\big(\boldsymbol{\Sigma}^{\top}(\boldsymbol{\Pi}_{\boldsymbol{P}} \boldsymbol{\Lambda}_{\boldsymbol{P}}^{\uparrow} \boldsymbol{\Pi}_{\boldsymbol{P}}^{\top})\boldsymbol{\Sigma}(\boldsymbol{\Pi}_{\boldsymbol{Q}} \boldsymbol{\Lambda}_{\boldsymbol{Q}}^{\uparrow} \boldsymbol{\Pi}_{\boldsymbol{Q}}^{\top})\big)\). Note that every factor inside the parentheses is a diagonal matrix. Finding the minimum amounts to finding a pairing among the eigenvalues of \(\boldsymbol{P},\boldsymbol{Q},\boldsymbol{G}^{\top}\boldsymbol{G}\) that minimizes the sum of their products, which is essentially a three-sequence generalization of the rearrangement inequality. From [*The Rearrangement Inequality and Its Generalizations*](https://kexue.fm/archives/11910), we know that the ideal outcome is for the component-wise products to be equal after pairing, which corresponds to \(\boldsymbol{D}\propto\boldsymbol{\Sigma}^{-1}\) (taking the reciprocal of each diagonal entry while keeping the shape unchanged).

Of course, since we are only finding the minimum via pairing, it is hard to make all the products equal. In general, we can only expect it to be an intermediate state between \(\boldsymbol{\Sigma}\) (the worst case) and \(\boldsymbol{\Sigma}^{-1}\) (the best case) — and the geometric mean of the two is precisely the identity matrix! So once again, this points to the orthogonalization operation that makes all singular values uniform!

## Further Remarks

I should make clear that I have no intention of praising Muon at Shampoo's expense (捧Muon踩Shampoo). On the contrary, some time ago I spent quite a lot of time testing various Shampoo variants, with the original intention of finding an optimizer better than Muon. Unfortunately, after trying for a long time, I found that none of the Shampoo variants could beat Muon. As it happened, I had recently been deriving convergence results for adaptive optimizers, so I thought about the two together, which eventually led to the conclusions of this post.

To be frank, the conclusions of this post rely on the "orthogonal slowest descent" framework, and that framework in turn relies on a series of assumptions, every link of which is open to question. For example, using a convergence upper bound to guide optimizer design is itself a pessimistic, worst-case criterion; a smaller upper bound does not guarantee faster convergence in practice. As another example, the assumption that \(\boldsymbol{O}_t\) varies slowly, as well as the first-order approximations used throughout, are rather crude and cannot be made more precisely quantitative.

More importantly, although minimizing the convergence upper bound looks more global, it is essentially also a greedy solution — and, strictly speaking, the problem isn't even well-posed! This is because it takes the sequence of \(\boldsymbol{g}_t\) as given and derives the remaining parameters from it. However, once the learning rate or other parameters are changed, the subsequent gradients change accordingly, so there is simply no way for the entire gradient sequence to be given in advance.

Of course, if we accept these non-rigorous aspects and accept orthogonal slowest descent, then we arrive at the heuristic conclusions above, which tell us: we only need to focus on adjusting the spectrum (singular values) of the gradient or momentum, without paying a high price to build complex preconditioning matrices. This is at least engineering-friendly. As for the Shampoo route, setting aside whether it actually works better, the extra state variables and computation it introduces certainly carry a large engineering cost.

## Summary

Starting from the average convergence upper bound of convex optimization, this post proposed the principle of "orthogonal slowest descent" and applied it to Kronecker-product-based preconditioned optimizers (Muon and the Shampoo family). We found that, regardless of the original preconditioning matrix, orthogonal slowest descent "washes out" its eigenvector matrix, leaving at most its eigenvalues to correct the singular value spectrum of the gradient (or momentum) and equalize it. All of this is highly homogeneous with Muon — just as the title suggests.

<hr class="section-divider">

*Citation: Su, J. (2026, September 28). 慢即是快：1. Shampoo的尽头是Muon？ [Slow Is Fast: 1. Is Muon Where Shampoo Ends Up?]. Scientific Spaces. [https://kexue.fm/archives/11917](https://kexue.fm/archives/11917)*

*Original content licensed under [CC BY-NC-ND 2.5 CN](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/). This translation is published with the author's permission for educational use.*