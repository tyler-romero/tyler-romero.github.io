---
title: "Why Does the Official Muon Have an Extra max(1, ·) Compared to the MuP Version?"
subtitle: "Translated from [为什么官方版Muon比MuP版多出一个max(1, ⋅)？](https://kexue.fm/archives/11772) by Jianlin Su (苏剑林)"
date: 2026-06-03T00:00:00+08:00
blurb: "An analysis of where the max(1, ·) clipping in the KellerJordan version of Muon comes from, derived by requiring uniform feature-increment RMS across input and output dimensions."
tags: ["translation", "muon", "optimization", "mup"]
math: true
---

*Translator's note (Opus 4.7): This is an English translation of [为什么官方版Muon比MuP版多出一个max(1, ⋅)？](https://kexue.fm/archives/11772) by Jianlin Su (苏剑林), originally published on June 3, 2026 on [Scientific Spaces (科学空间)](https://kexue.fm). The translation preserves the author's first-person voice.*

<hr class="section-divider">

In the post [*Muon Optimizer Guide: Quick Start and Key Details*](/translations/scientific-spaces/muon-optimizer-guide-quick-start-and-key-details/), we listed several versions of Muon. They differ only in the matrix-shape-dependent scaling factor of the learning rate, where the "official version (KellerJordan version)" is just the "MuP version" plus an extra \(\max(1, \cdot)\) truncation. This post discusses where that truncation actually comes from.

## The Variants

The Muon update rule can be written uniformly as
\[
\begin{aligned}
\boldsymbol{M}_t =&\, \beta \boldsymbol{M}_{t-1} + \boldsymbol{G}_t \\[5pt]
\boldsymbol{W}_t =&\, \boldsymbol{W}_{t-1} - \eta_t (\alpha \operatorname{msign}(\boldsymbol{M}_t) + \lambda \boldsymbol{W}_{t-1})
\end{aligned}
\tag{1}
\]
The different versions differ in the choice of \(\alpha\):
\[
\alpha = \left\{
\begin{aligned}
& 1 & (\text{naïve version}) \\[5pt]
& \sqrt{\max(1, d_{out}/d_{in})} & (\text{KellerJordan version}) \\[5pt]
& \sqrt{d_{out}/d_{in}} & (\text{MuP version}) \\[5pt]
& 0.2\times\sqrt{\max(d_{out},d_{in})} & (\text{Moonlight version})
\end{aligned}\right.
\tag{2}
\]

Here the matrix \(\boldsymbol{W}\in\mathbb{R}^{d_{in}\times d_{out}}\) represents the trainable parameters of a linear layer \(\boldsymbol{y}=\boldsymbol{x}\boldsymbol{W}\), where the input \(\boldsymbol{x}\in\mathbb{R}^{d_{in}}\) is a row vector.

This post focuses mainly on the "KellerJordan version" and the "MuP version". The former adds an extra \(\max(1, \cdot)\) on top of the latter. Based on the analyses in [*High-Order MuP: A More Concise but More Insightful Spectral-Condition Scaling*](https://kexue.fm/archives/10795) and [*Beyond MuP 2: Linear Layers and Steepest Descent*](/translations/scientific-spaces/beyond-mup-2-linear-layers-and-steepest-descent/), under the spectral-condition constraints of MuP, the steepest descent direction should give exactly the MuP version of Muon. So how should we explain the extra \(\max(1, \cdot)\)?

## Feature Increment

For simplicity, in what follows we drop the subscript \(t\). Without loss of generality, we assume the momentum \(\boldsymbol{M}\) is full rank, so that the singular values of \(\boldsymbol{\Phi} = \operatorname{msign}(\boldsymbol{M})\) are all 1. Then when \(d_{in} \leq d_{out}\), \(\boldsymbol{\Phi} \boldsymbol{\Phi}^{\top} = \boldsymbol{I}_{d_{in}}\), and when \(d_{in} > d_{out}\), \(\boldsymbol{\Phi}^{\top} \boldsymbol{\Phi} = \boldsymbol{I}_{d_{out}}\).

Let \(\Delta \boldsymbol{W} = \eta\alpha \boldsymbol{\Phi}\). What we want is to find the relationship between \(\alpha\) and \(d_{in}, d_{out}\). From [*Why Do We Prefer Isotropy? An Understanding from Steepest Descent*](https://kexue.fm/archives/11549) we know that parameters are really just a by-product of the model — the changes at the feature level may be more fundamental. Translating \(\Delta \boldsymbol{W}\) to the feature level gives \(\Delta \boldsymbol{y} = \boldsymbol{x} \Delta\boldsymbol{W} = \eta\alpha \boldsymbol{x}\boldsymbol{\Phi}\), so \(\Vert\Delta \boldsymbol{y}\Vert_{RMS} = \alpha \Vert\boldsymbol{x}\boldsymbol{\Phi}\Vert_{RMS}\).

We need to consider two cases. First, when \(d_{in} \leq d_{out}\), \(\boldsymbol{\Phi}\) can be written in the form \(\boldsymbol{U}[\boldsymbol{I}_{d_{in}}, \boldsymbol{0}_{d_{in}\times (d_{out}-d_{in})}]\boldsymbol{V}^{\top}\), where \(\boldsymbol{U}\in\mathbb{R}^{d_{in}\times d_{in}}\) and \(\boldsymbol{V}\in\mathbb{R}^{d_{out}\times d_{out}}\) are both orthogonal matrices. Then
\[
\begin{aligned}
\Vert\Delta \boldsymbol{y}\Vert_{RMS} =&\, \eta\alpha\big\Vert\boldsymbol{x}\boldsymbol{U}[\boldsymbol{I}_{d_{in}}, \boldsymbol{0}_{d_{in}\times (d_{out}-d_{in})}]\boldsymbol{V}^{\top}\big\Vert_{RMS} \\[4pt]
=&\, \eta\alpha\big\Vert\boldsymbol{x}\boldsymbol{U}[\boldsymbol{I}_{d_{in}}, \boldsymbol{0}_{d_{in}\times (d_{out}-d_{in})}]\big\Vert_{RMS} \\[4pt]
=&\, \eta\alpha\big\Vert[\boldsymbol{x}\boldsymbol{U}, \boldsymbol{0}_{d_{out}-d_{in}}]\big\Vert_{RMS} \\[4pt]
=&\, \eta\alpha\sqrt{\frac{d_{in}}{d_{out}}}\Vert\boldsymbol{x}\boldsymbol{U}\Vert_{RMS} \\[4pt]
=&\, \eta\alpha\sqrt{\frac{d_{in}}{d_{out}}}\Vert\boldsymbol{x}\Vert_{RMS} \\
\end{aligned}
\tag{3}
\]
Note that every step is an equality, so we only need to set \(\alpha = \sqrt{d_{out}/d_{in}}\) to make the RMS of "every" \(\Delta \boldsymbol{y}\) equal to \(\eta\Vert\boldsymbol{x}\Vert_{RMS}\) — i.e., the relative update magnitude is the same for all tokens.

## The Isotropic Case

Unfortunately, in the second case \(d_{in} > d_{out}\), the goal of "complete uniformity" cannot be achieved. Specifically, in this case the SVD of \(\boldsymbol{\Phi}\) takes the form \(\boldsymbol{U}\begin{bmatrix}\boldsymbol{I}_{d_{out}} \\ \boldsymbol{0}_{(d_{in}-d_{out})\times d_{out}}\end{bmatrix}\boldsymbol{V}^{\top}\), so
\[
\begin{aligned}
\Vert\Delta \boldsymbol{y}\Vert_{RMS} =&\, \eta\alpha\left\Vert\boldsymbol{x}\boldsymbol{U}\begin{bmatrix}\boldsymbol{I}_{d_{out}} \\ \boldsymbol{0}_{(d_{in}-d_{out})\times d_{out}}\end{bmatrix}\boldsymbol{V}^{\top}\right\Vert_{RMS} \\[5pt]
=&\, \eta\alpha\left\Vert\boldsymbol{x}\boldsymbol{U}\begin{bmatrix}\boldsymbol{I}_{d_{out}} \\ \boldsymbol{0}_{(d_{in}-d_{out})\times d_{out}}\end{bmatrix}\right\Vert_{RMS} \\[5pt]
=&\, \eta\alpha\big\Vert(\boldsymbol{x}\boldsymbol{U})_{[:d_{out}]}\big\Vert_{RMS}
\end{aligned}
\tag{4}
\]
Here \(\boldsymbol{x}\boldsymbol{U}\) is a \(d_{in}\)-dimensional vector, and since \(d_{in} > d_{out}\), \((\boldsymbol{x}\boldsymbol{U})_{[:d_{out}]}\) just takes the first \(d_{out}\) components of \(\boldsymbol{x}\boldsymbol{U}\) before computing the RMS. Its RMS is then indeterminate: at most it can reach \(\sqrt{d_{in}/d_{out}}\Vert\boldsymbol{x}\Vert_{RMS}\) (the worst case), and at least it can be 0.

We know that orthogonal matrices do not change the RMS, so \(\Vert\boldsymbol{x}\boldsymbol{U}\Vert_{RMS}=\Vert\boldsymbol{x}\Vert_{RMS}\). When the distribution of \(\boldsymbol{x}\) is sufficiently isotropic, we may take this to mean that each component of \(\boldsymbol{x}\boldsymbol{U}\) has average scale \(\Vert\boldsymbol{x}\Vert_{RMS}\). Then taking the first \(d_{out}\) components and computing the RMS, on average it is also approximately \(\Vert\boldsymbol{x}\Vert_{RMS}\), i.e. \(\Vert\Delta \boldsymbol{y}\Vert_{RMS}\approx \alpha\Vert\boldsymbol{x}\Vert_{RMS}\). Therefore we only need to take \(\alpha = 1\) to achieve an effect similar to the previous section.

## The Anisotropic Case

Combining the results of the previous two sections, we obtain
\[
\alpha = \sqrt{\max\left(1, \frac{d_{out}}{d_{in}}\right)}
\tag{5}
\]
which is exactly the \(\max(1, \cdot)\) appearing in the KellerJordan version of Muon.

However, the conclusion of the previous section relied on the assumption that the input \(\boldsymbol{x}\) is sufficiently isotropic. This may approximately hold early in training, but as training progresses, the feature distribution gradually becomes anisotropic and concentrates in the directions that maximize \(\Vert\Delta \boldsymbol{y}\Vert_{RMS}\) — the "worst case" — at which point the average approximation \(\Vert\Delta \boldsymbol{y}\Vert_{RMS}\approx \eta\alpha\Vert\boldsymbol{x}\Vert_{RMS}\) is no longer accurate. Instead, the maximum \(\eta\alpha\sqrt{d_{in}/d_{out}}\Vert\boldsymbol{x}\Vert_{RMS}\) becomes the more accurate estimate.

In this case, the \(\alpha\) that makes \(\Vert\Delta \boldsymbol{y}\Vert_{RMS}\approx \eta\Vert\boldsymbol{x}\Vert_{RMS}\) is \(\sqrt{d_{out}/d_{in}}\), which agrees with the conclusion in the \(d_{in} \leq d_{out}\) case and recovers the MuP version. In other words, in the middle and later stages of training, the MuP version of Muon is more principled. To address this inconsistency, we have two strategies. The first is to always use the MuP version of Muon: this slightly slows down convergence in the early stage, but after all the middle and later stages are the "main act" of training. The second is to change the scaling factor to
\[
\alpha = \sqrt{\max\left(\tau_t, \frac{d_{out}}{d_{in}}\right)}
\tag{6}
\]
where \(\tau_t\) decays monotonically from 1 to 0. This produces a smooth transition from the KellerJordan version to the MuP version, at the cost of one more schedule to tune.

## Summary

This post mainly explains the origin of the \(\max(1, \cdot)\) in the KellerJordan version from the perspective of uniformity of "feature increments".

<hr class="section-divider">

*Citation: Su, J. (2026, June 3). 为什么官方版Muon比MuP版多出一个max(1, ⋅)？ [Why Does the Official Muon Have an Extra max(1, ·) Compared to the MuP Version?]. Scientific Spaces. [https://kexue.fm/archives/11772](https://kexue.fm/archives/11772)*

*Original content licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). This translation is shared under the same license.*
