---
title: "Beyond MuP 4: Upholding Parameter Stability"
subtitle: "Translated from [MuP之上：4. 坚守参数的稳定性](https://kexue.fm/archives/11729) by Jianlin Su (苏剑林)"
date: 2026-04-24T00:00:00+08:00
blurb: "A framework for maintaining parameter stability throughout training via minimal-intervention clipping operators, yielding singular value clipping (Post Clip) and spectral weight decay (Pre Decay) as special cases for the Muon optimizer."
tags: ["translation", "mup", "muon", "optimizer", "stability"]
math: true
---

*Translator's note (Opus 4.6): This is an English translation of [MuP之上：4. 坚守参数的稳定性](https://kexue.fm/archives/11729) by Jianlin Su (苏剑林), originally published on April 24, 2026 on [Scientific Spaces (科学空间)](https://kexue.fm). The translation preserves the author's first-person voice.*

<hr class="section-divider">

Through the derivations and computations of the previous articles, we can observe that the three stability indicators proposed in the first article [*Beyond MuP 1: Three Characteristics of Good Models*](/translations/scientific-spaces/beyond-mup-1-three-characteristics-of-good-models/) can generally be divided into "parameter stability" and "increment stability." In [*Beyond MuP 2: Linear Layers and Steepest Descent*](/translations/scientific-spaces/beyond-mup-2-linear-layers-and-steepest-descent/) and [*Beyond MuP 3: Special Cases, Special Treatment*](/translations/scientific-spaces/beyond-mup-3-special-cases-special-treatment/), we demonstrated the process of combining increment stability with steepest descent to derive new update rules (optimizers).

However, for parameter stability, we previously only addressed initialization. The task of this article is precisely to explore how to maintain parameter stability throughout the entire training process, completing the theoretical framework in practice.

## Problem Background

Taking [*Beyond MuP 2: Linear Layers and Steepest Descent*](/translations/scientific-spaces/beyond-mup-2-linear-layers-and-steepest-descent/) as an example, the three stability indicators are:

\[
\begin{aligned}
&\text{Forward stability:}\quad\max_{\Vert\boldsymbol{x}\Vert_{RMS}=1} \Vert \boldsymbol{x}\boldsymbol{W}\Vert_{RMS} = \sqrt{\frac{d_{in}}{d_{out}}}\Vert\boldsymbol{W}\Vert_2 \\[5pt]
&\text{Dependence stability:}\quad\max_{\Vert\boldsymbol{x}_1\Vert_{RMS}=\Vert\boldsymbol{x}_2\Vert_{RMS}=1} \frac{\Vert \boldsymbol{x}_1\boldsymbol{W} - \boldsymbol{x}_2\boldsymbol{W}\Vert_{RMS}}{\Vert \boldsymbol{x}_1 - \boldsymbol{x}_2\Vert_{RMS}} = \sqrt{\frac{d_{in}}{d_{out}}}\Vert\boldsymbol{W}\Vert_2 \\[5pt]
&\text{Update stability:}\quad\max_{\Vert\boldsymbol{x}\Vert_{RMS}=1} \Vert \boldsymbol{x}(\boldsymbol{W} + \Delta\boldsymbol{W}) - \boldsymbol{x}\boldsymbol{W}\Vert_{RMS} = \sqrt{\frac{d_{in}}{d_{out}}}\Vert\Delta\boldsymbol{W}\Vert_2
\end{aligned}
\]

where \(\boldsymbol{W}\in\mathbb{R}^{d_{in}\times d_{out}}\) is the linear layer's parameter. We want all three indicators to be \(\Theta(1)\), which means we want the parameter and its increment to satisfy \(\Vert\boldsymbol{W}\Vert_2 = \Theta(\sqrt{d_{out}/d_{in}})\) and \(\Vert\Delta\boldsymbol{W}\Vert_2 = \Theta(\sqrt{d_{out}/d_{in}})\), respectively. In [*Beyond MuP 3: Special Cases, Special Treatment*](/translations/scientific-spaces/beyond-mup-3-special-cases-special-treatment/), we performed computations for layers like Embedding and LM Head, and the conclusions were similar — only the corresponding norms differed.

For the increment condition, we treat it as a stability indicator, and based on the "stability first, speed second" steepest descent principle, we derive the theoretically optimal update rule. For example, the linear layer corresponds to the Muon optimizer:

\[
\operatorname*{argmin}_{\Vert\Delta\boldsymbol{W}\Vert_2\leq\eta\sqrt{\frac{d_{out}}{d_{in}}}} \operatorname{tr}(\boldsymbol{G}^{\top}\Delta\boldsymbol{W}) \qquad \Rightarrow \qquad \Delta\boldsymbol{W} = -\eta\sqrt{\frac{d_{out}}{d_{in}}}\operatorname{msign}(\boldsymbol{G})
\]

As for the parameter stability part, we previously only required that the initialization satisfy \(\Vert\boldsymbol{W}\Vert_2 = \Theta(\sqrt{d_{out}/d_{in}})\). How to ensure the model maintains the same parameter stability throughout the entire training process remained unknown.

## General Framework

How can we ensure that \(\boldsymbol{W}\) maintains \(\Vert\boldsymbol{W}\Vert_2 = \Theta(\sqrt{d_{out}/d_{in}})\)? More generally, given a parameter \(\boldsymbol{\omega}\) — which could be a vector, matrix, or even a higher-order tensor — along with a norm \(\Vert\cdot\Vert\) (typically induced by forward stability or dependence stability) and a specified scale \(\tau\), the question is: how can we ensure that \(\boldsymbol{\omega}\) maintains \(\Vert\boldsymbol{\omega}\Vert=\Theta(\tau)\) throughout training?

### Initial Thoughts

A naive idea is to directly enforce \(\Vert\boldsymbol{\omega}\Vert=\tau\) (one could replace \(\tau\) with a constant multiple, but this does not affect the following discussion). The simplest implementation is to rescale the norm back to \(\tau\) via normalization after each optimization step (e.g., [Hyperball](https://whenwen.github.io/wd_blog/public/hyperball-part-1.html) and [Nemotron-Flash](https://papers.cool/arxiv/2511.18890)). Another approach is to reparameterize the original model using normalization, i.e., change \(\boldsymbol{f}(\boldsymbol{x};\boldsymbol{\omega})\) to \(\boldsymbol{f}(\boldsymbol{x};\tau\boldsymbol{\omega}/\Vert\boldsymbol{\omega}\Vert)\), which theoretically achieves a similar effect.

A more advanced approach combines the steepest descent idea to adjust the update rule, as discussed in [*Steepest Descent on Manifolds 1: SGD + Hypersphere*](https://kexue.fm/archives/11196), [*Steepest Descent on Manifolds 4: Muon + Spectral Sphere*](https://kexue.fm/archives/11241), and the paper [*Controlled LLM Training on Spectral Sphere*](https://papers.cool/arxiv/2601.08393). This approach is more elegant methodologically, but more complex in practice — it typically requires solving a nonlinear equation to obtain the exact update.

However, should we really constrain a parameter's norm strictly to some fixed value? Intuitively, the parameter's norm should be determined by the training process itself; at most, we set a prior range for it. Although some works show that fixing the parameter norm to a preset value does not affect performance when done properly, this still disrupts the original training dynamics and may require more effort to understand and adapt to.

Therefore, the viewpoint this article proposes is: we only need to ensure \(\Vert\boldsymbol{\omega}\Vert = \mathcal{O}(\tau)\). Specifically, we aim to guarantee that every step satisfies \(\Vert\boldsymbol{\omega}\Vert \leq \tau\), while the exact value — and whether it achieves \(\Theta(\tau)\) — is left for the training algorithm itself to decide, with no further intervention.

### Post Clip

The next natural question is: how do we implement \(\Vert\boldsymbol{\omega}\Vert \leq \tau\)? More specifically, suppose \(\boldsymbol{\omega}\)'s original update rule is

\[
\boldsymbol{\omega}_t = \boldsymbol{\omega}_{t-1} - \eta \boldsymbol{\phi}_t \tag{1}
\]

How should we modify it to ensure \(\boldsymbol{\omega}_t\) always satisfies \(\Vert\boldsymbol{\omega}_t\Vert\leq\tau\)? There are of course many methods — for instance, the normalization mentioned in the previous section is also a valid approach. Given this, we want to select the approach with the **least impact** on the optimization process. That is, given a parameter \(\boldsymbol{\omega}\) and a norm \(\Vert\cdot\Vert\), we want to reduce its norm to at most \(\tau\) with minimal modification, formally defined as:

\[
\lfloor\boldsymbol{\omega}\rfloor_{\Vert\cdot\Vert\leq\tau} = \operatorname*{argmin}_{\Vert\tilde{\boldsymbol{\omega}}\Vert\leq\tau} \Vert \boldsymbol{\omega} - \tilde{\boldsymbol{\omega}}\Vert_{RMS} \tag{2}
\]

Readers familiar with convex optimization will readily recognize this as the projection of \(\boldsymbol{\omega}\) onto the ball of radius \(\tau\) under a certain norm. The key point here is that we want to achieve the goal of norm not exceeding \(\tau\), while minimizing the impact on the original parameter \(\boldsymbol{\omega}\), hence we minimize the discrepancy metric \(\Vert \boldsymbol{\omega} - \tilde{\boldsymbol{\omega}}\Vert_{RMS}\), which induces a specific projection or clipping operation.

As for how to compute \(\lfloor\boldsymbol{\omega}\rfloor_{\Vert\cdot\Vert\leq\tau}\), this requires case-by-case analysis for specific norms, which we will expand on shortly. With this operation in hand, one approach we can consider is to clip the parameter norm after each update step, i.e., modify equation (1) to:

\[
\boldsymbol{\omega}_t = \lfloor\boldsymbol{\omega}_{t-1} - \eta \boldsymbol{\phi}_t\rfloor_{\Vert\cdot\Vert\leq\tau}
\]

We tentatively call this approach "Post Clip." Its characteristic is simplicity and intuitiveness, but it may give a sense of "non-smoothness." This is easy to understand: suppose we initialize with a radius smaller than \(\tau\), and as training begins the parameter radius slowly increases; once it reaches \(\tau\), clipping "suddenly" kicks in. Although this process is continuous, it is not smooth — similar to the \(\max(x,0)\) function.

### Pre Decay

If this non-smoothness is concerning, we can consider mimicking weight decay by distributing the penalty across each update step. Starting again from the update rule (1), suppose \(\boldsymbol{\phi}_t\) satisfies \(\Vert\boldsymbol{\phi}_t\Vert\leq\tau\). Then by the triangle inequality, \(\Vert\boldsymbol{\omega}_t\Vert = \Vert\boldsymbol{\omega}_{t-1} - \eta \boldsymbol{\phi}_t\Vert\leq \Vert\boldsymbol{\omega}_{t-1}\Vert + \eta \tau\). That is, in the extreme case the norm increases by \(\eta\tau\) per step, which accumulates over time and eventually "loses control."

To prevent this, we can preprocess \(\boldsymbol{\omega}_{t-1}\) before adding \(- \eta \boldsymbol{\phi}_t\), reducing its norm just enough to offset the growth from the update. Drawing on the experience of weight decay, we can consider:

\[
\boldsymbol{\omega}_t = \lfloor\boldsymbol{\omega}_{t-1}\rfloor_{\Vert\cdot\Vert\leq (1-\eta)\Vert\boldsymbol{\omega}_{t-1}\Vert} - \eta \boldsymbol{\phi}_t \tag{3}
\]

That is, first reduce the norm of \(\boldsymbol{\omega}_{t-1}\) to \(1-\eta\) times its original value, then perform the update. This gives:

\[
\Vert\boldsymbol{\omega}_t\Vert \leq (1-\eta)\Vert\boldsymbol{\omega}_{t-1}\Vert + \eta \tau \leq \max(\Vert\boldsymbol{\omega}_{t-1}\Vert,\tau)
\]

Propagating this forward: \(\Vert\boldsymbol{\omega}_t\Vert \leq \max(\Vert\boldsymbol{\omega}_{t-1}\Vert,\tau) \leq \cdots \leq \max(\Vert\boldsymbol{\omega}_0\Vert,\tau)\). As long as the initialization satisfies \(\Vert\boldsymbol{\omega}_0\Vert\leq\tau\), the entire update chain automatically satisfies \(\Vert\boldsymbol{\omega}_t\Vert\leq \tau\). This conclusion is independent of the specific norm — it relies only on the triangle inequality. And the minimal-modification operation for reducing the norm is precisely the clipping operator defined in equation (2), so using it to reduce the norm is a natural choice.

We call this approach "Pre Decay." The difference from Post Clip is that the latter's threshold is the static \(\tau\) (so clipping does not necessarily trigger), while the former's threshold is the dynamic \((1-\eta)\Vert\boldsymbol{\omega}_{t-1}\Vert\), and clipping always triggers. This process is smoother, which is why we call it "decay" rather than "clipping" — it is a generalization of weight decay.

## Basic Results

So far we have established a general framework for constraining parameter norms, with two approaches: Post Clip and Pre Decay. The core operation in both is the clipping operator \(\lfloor\boldsymbol{\omega}\rfloor_{\Vert\cdot\Vert\leq\tau}\) defined in equation (2), which currently only has a formal definition. In practice, it must be computed on a case-by-case basis for specific norms. Below we present some basic results.

### A Simple Example

In this section we first compute a simple example where the chosen norm is \(\Vert\cdot\Vert_{RMS}\), which for vectors is equivalent to the L2 norm and for matrices is equivalent to the Frobenius norm. It is not hard to obtain:

\[
\lfloor\boldsymbol{\omega}\rfloor_{\Vert\cdot\Vert_{RMS}\leq\tau} = \operatorname*{argmin}_{\Vert\tilde{\boldsymbol{\omega}}\Vert_{RMS}\leq\tau} \Vert \boldsymbol{\omega} - \tilde{\boldsymbol{\omega}}\Vert_{RMS} = \min\left(1,\,\frac{\tau}{\Vert\boldsymbol{\omega}\Vert_{RMS}}\right)\boldsymbol{\omega}
\]

The proof is left to the reader (if you really cannot figure it out, you can ask Kimi). In particular, substituting \(\tau = (1 - \eta) \Vert\omega\Vert_{RMS}\):

\[
\lfloor\boldsymbol{\omega}\rfloor_{\Vert\cdot\Vert_{RMS}\leq (1 - \eta) \Vert\omega\Vert_{RMS}} = \min\left(1,\,\frac{(1 - \eta) \Vert\omega\Vert_{RMS}}{\Vert\boldsymbol{\omega}\Vert_{RMS}}\right)\boldsymbol{\omega} = (1-\eta)\boldsymbol{\omega}
\]

Then substituting into equation (3):

\[
\boldsymbol{\omega}_t = (1-\eta)\boldsymbol{\omega}_{t-1} - \eta \boldsymbol{\phi}_t
\]

It is easy to see that this is simply conventional weight decay. In other words, Pre Decay under the RMS norm is our familiar weight decay — it is the Pre Decay scheme with minimal modification to the original parameter under the constraint of maintaining the RMS norm (equivalently, the L2 norm of a vector or the Frobenius norm of a matrix).

### Singular Value Clipping

Now we enter this article's "main event" — matrix parameters and Muon. Here we return to the notation \(\boldsymbol{W}\), and write Muon's original update rule as:

\[
\boldsymbol{W}_t = \boldsymbol{W}_{t-1} - \eta\lambda\boldsymbol{\Phi}_t,\quad \boldsymbol{\Phi}_t=\frac{1}{\lambda}\sqrt{\frac{d_{out}}{d_{in}}}\operatorname{msign}(\boldsymbol{G}_t)
\]

Let \(\tau = \frac{1}{\lambda}\sqrt{\frac{d_{out}}{d_{in}}}\), so that \(\Vert\boldsymbol{\Phi}_t\Vert_2=\tau\). The two approaches for ensuring \(\boldsymbol{W}_t\) satisfies \(\Vert\boldsymbol{W}_t\Vert_2\leq\tau\) are:

\[
\begin{aligned}
\text{Post Clip:}\quad\boldsymbol{W}_t =&\, \lfloor\boldsymbol{W}_{t-1} - \eta\lambda\boldsymbol{\Phi}_t\rfloor_{\Vert\cdot\Vert_2\leq\tau} \\[5pt]
\text{Pre Decay:}\quad\boldsymbol{W}_t =&\, \lfloor\boldsymbol{W}_{t-1}\rfloor_{\Vert\cdot\Vert_2\leq(1-\eta\lambda)\Vert\boldsymbol{W}_{t-1}\Vert_2}  - \eta\lambda\boldsymbol{\Phi}_t \\
\end{aligned}
\]

The next task is to compute \(\lfloor\boldsymbol{W}\rfloor_{\Vert\cdot\Vert_2\leq\tau}\). Using the equivalence between RMS and Frobenius norms, this equals:

\[
\lfloor\boldsymbol{W}\rfloor_{\Vert\cdot\Vert_2\leq\tau} = \operatorname*{argmin}_{\Vert\tilde{\boldsymbol{W}}\Vert_2\leq\tau} \Vert\boldsymbol{W} - \tilde{\boldsymbol{W}}\Vert_F \tag{4}
\]

The optimal solution to this problem should be familiar to some readers — it is the "Singular Value Clipping (SVC)" that we mentioned in [*Higher-Order MuP: A Simpler Yet More Sophisticated Spectral Scaling Condition*](https://kexue.fm/archives/10795), and which was called \(\operatorname{mclip}\) in [*Computing Singular Value Clipping mclip via msign (Part 1)*](https://kexue.fm/archives/11006) and [*Computing Singular Value Clipping mclip via msign (Part 2)*](https://kexue.fm/archives/11059):

\[
\lfloor\boldsymbol{W}\rfloor_{\Vert\cdot\Vert_2\leq\tau} = \operatorname{mclip}(\boldsymbol{W};\tau) = \boldsymbol{U}\min(\boldsymbol{\Sigma},\tau)\boldsymbol{V}^{\top} \tag{5}
\]

where \(\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}\) is the SVD of \(\boldsymbol{W}\), and \(\min(\boldsymbol{\Sigma},\tau)\) clips the singular values to not exceed \(\tau\). We will demonstrate the proof in the next section. With this notation, the two approaches can be written as:

\[
\begin{aligned}
\text{Post Clip:}\quad\boldsymbol{W}_t =&\, \operatorname{mclip}(\boldsymbol{W}_{t-1} - \eta\lambda\boldsymbol{\Phi}_t;\tau) \\[5pt]
\text{Pre Decay:}\quad\boldsymbol{W}_t =&\, \operatorname{mclip}(\boldsymbol{W}_{t-1};(1-\eta\lambda)\Vert\boldsymbol{W}_{t-1}\Vert_2)  - \eta\lambda\boldsymbol{\Phi}_t \\
\end{aligned}
\]

### Derivation

In this section we prove result (5). Let the SVD of \(\boldsymbol{W}\) be \(\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}\), where \(\boldsymbol{U}\in\mathbb{R}^{d_{in}\times d_{in}}\), \(\boldsymbol{\Sigma}\in\mathbb{R}^{d_{in}\times d_{out}}\), \(\boldsymbol{V}\in\mathbb{R}^{d_{out}\times d_{out}}\). Then:

\[
\Vert\boldsymbol{W} - \tilde{\boldsymbol{W}}\Vert_F = \Vert\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top} - \tilde{\boldsymbol{W}}\Vert_F = \Vert\boldsymbol{U}(\boldsymbol{\Sigma} - \boldsymbol{U}^{\top}\tilde{\boldsymbol{W}}\boldsymbol{V})\boldsymbol{V}^{\top}\Vert_F = \Vert\boldsymbol{\Sigma} - \boldsymbol{U}^{\top}\tilde{\boldsymbol{W}}\boldsymbol{V}\Vert_F
\]

The last equality holds because orthogonal matrices do not change the Frobenius norm. Similarly, orthogonal matrices do not change the spectral norm, so letting \(\tilde{\boldsymbol{\Sigma}}=\boldsymbol{U}^{\top}\tilde{\boldsymbol{W}}\boldsymbol{V}\), the objective (4) can be equivalently simplified to:

\[
\operatorname*{argmin}_{\Vert\tilde{\boldsymbol{\Sigma}}\Vert_2\leq\tau} \Vert\boldsymbol{\Sigma} - \tilde{\boldsymbol{\Sigma}}\Vert_F
\]

Note that \(\boldsymbol{\Sigma}\) here is a diagonal matrix with diagonal entries \(\sigma_1,\sigma_2,\cdots \geq 0\), but \(\tilde{\boldsymbol{\Sigma}}\) is initially undetermined — for the proof we treat it as a general matrix. Writing in component form:

\[
\Vert\boldsymbol{\Sigma} - \tilde{\boldsymbol{\Sigma}}\Vert_F^2 = \sum_i \sigma_i^2 + \sum_{i,j} \tilde{\Sigma}_{i,j}^2 - 2\sum_i \sigma_i \tilde{\Sigma}_{i,i} \geq \sum_i \sigma_i^2 + \sum_i (\tilde{\Sigma}_{i,i}^2 - 2 \sigma_i \tilde{\Sigma}_{i,i})
\]

Term by term, \(\tilde{\Sigma}_{i,i}^2 - 2 \sigma_i \tilde{\Sigma}_{i,i}\) is simply a quadratic function of \(\tilde{\Sigma}_{i,i}\), minimized at \(\tilde{\Sigma}_{i,i}=\sigma_i\). But we also have the constraint \(\Vert\tilde{\boldsymbol{\Sigma}}\Vert_2\leq\tau\). Since the spectral norm is at least as large as the absolute value of any matrix entry, we have at least the constraint \(\tilde{\Sigma}_{i,i}\leq\tau\). Under this constraint, \(\tilde{\Sigma}_{i,i}^2 - 2 \sigma_i \tilde{\Sigma}_{i,i}\) is minimized at \(\tilde{\Sigma}_{i,i}^* = \min(\sigma_i,\tau)\).

Considering that all equalities hold simultaneously, we get \(\tilde{\Sigma}_{i,j}^*=0\) for \(i\neq j\). Thus \(\tilde{\boldsymbol{\Sigma}}^*\) is also a diagonal matrix, which can be written succinctly as \(\tilde{\boldsymbol{\Sigma}}^*=\min(\boldsymbol{\Sigma},\tau)\), corresponding to \(\tilde{\boldsymbol{W}}^*=\boldsymbol{U}\min(\boldsymbol{\Sigma},\tau)\boldsymbol{V}^{\top}\). This completes the proof of result (5).

### Clipping the Leading Term

So how can \(\operatorname{mclip}\) be computed efficiently? Performing a full SVD at every training step is obviously too expensive. In the articles [*Computing Singular Value Clipping mclip via msign (Part 1)*](https://kexue.fm/archives/11006) and [*Computing Singular Value Clipping mclip via msign (Part 2)*](https://kexue.fm/archives/11059), we actually explored this problem systematically. The approach there was to use \(\operatorname{msign}\) to implement it, but this requires 2–3 applications of \(\operatorname{msign}\), which is costly. For example, one identity discovered in Part 2 is:

\[
\operatorname{mclip}(\boldsymbol{W};\tau)
=\frac{1}{2}\Bigl\{\boldsymbol{W}+\tau\operatorname{msign}(\boldsymbol{W})-(\tau\boldsymbol{I}-\boldsymbol{W}\operatorname{msign}(\boldsymbol{W})^{\top})\operatorname{msign}(\tau\operatorname{msign}(\boldsymbol{W})-\boldsymbol{W})\Bigr\}
\]

This requires two applications of \(\operatorname{msign}\). Since parameter computations are typically done in FP32, executing two \(\operatorname{msign}\) operations is quite expensive, making this not yet particularly practical.

Here we primarily consider the element-wise clipping approach discussed in [*Streaming Power Iteration for Muon 5: Extensions*](https://kexue.fm/archives/11719). Specifically, \(\operatorname{mclip}\) clips all singular values exceeding \(\tau\) down to \(\tau\). The necessary operation is to clip the leading singular value to \(\tau\) (if it exceeds \(\tau\)). After clipping the leading singular value, if there still exist singular values exceeding \(\tau\), the largest among them becomes the new leading singular value. So by repeatedly "clipping the leading singular value to \(\tau\)," we can implement \(\operatorname{mclip}\).

Since the leading singular value and singular vectors can be efficiently obtained via power iteration (denoted \(\mathop{\text{SVD1}}\)), clipping the leading singular value can be considered efficient. Furthermore, we assume training is sufficiently smooth so that each step only needs one leading singular value clip, which can approximately achieve the same effect. Based on this strategy, the two approaches for constraining singular values can be further written as:

\[
\begin{aligned}
\text{Post Clip:}\quad\boldsymbol{W}_t =&\, \tilde{\boldsymbol{W}}_t - \max(\sigma_1 - \tau, 0) \boldsymbol{u}_1 \boldsymbol{v}_1^{\top},\quad\sigma_1, \boldsymbol{u}_1, \boldsymbol{v}_1 = \mathop{\text{SVD1}}(\tilde{\boldsymbol{W}}_t),\quad\tilde{\boldsymbol{W}}_t = \boldsymbol{W}_{t-1} - \eta \boldsymbol{\Phi}_t \\[5pt]
\text{Pre Decay:}\quad\boldsymbol{W}_t =&\, \boldsymbol{W}_{t-1} - \lambda\eta\sigma_1 \boldsymbol{u}_1 \boldsymbol{v}_1^{\top} - \eta \boldsymbol{\Phi}_t,\quad\sigma_1, \boldsymbol{u}_1, \boldsymbol{v}_1 = \mathop{\text{SVD1}}(\boldsymbol{W}_{t-1})
\end{aligned}
\]

The "Pre Decay" version is precisely the spectral weight decay introduced in [*From Spectral Norm Gradients to a New Form of Weight Decay*](https://kexue.fm/archives/10648) — after more than a year, we have arrived at the same result via a different path. As for the "Post Clip" version, [@_arohan_](https://x.com/_arohan_/status/1929945590366122037) mentioned it on X, calling it "Wion" at the time. In practice, since only one singular value is clipped per step, there may be some particularly "ambitious" matrices whose spectral norm noticeably deviates from the set threshold — this is normal and will gradually decrease during the LR decay phase.

### Other Details

For more precise clipping, we can also use power iteration to simultaneously compute the top-\(k\) singular values and singular vectors, clipping at most \(k\) singular values per step. The cost is that the L2 normalization in power iteration must be replaced with QR decomposition, and QR decomposition also has acceleration techniques. The relevant principles can be found in the streaming power iteration series, such as [*Streaming Power Iteration for Muon 1: First Encounter*](https://kexue.fm/archives/11654).

Beyond the spectral norm of linear layer matrices, in [*Beyond MuP 3: Special Cases, Special Treatment*](/translations/scientific-spaces/beyond-mup-3-special-cases-special-treatment/) we encountered different norms for other layers. For example, Embedding and LM Head correspond to the maximum row and column RMS respectively, while the gamma parameter of RMS Norm corresponds to the maximum absolute value, also known as the infinity norm of a vector.

Fortunately, the clipping operator \(\lfloor\boldsymbol{\omega}\rfloor_{\Vert\cdot\Vert\leq\tau}\) under these norms is relatively easy to compute. For example, the Embedding layer's norm is the maximum row RMS, so the clipping operator simply clips each row vector's RMS to not exceed \(\tau\). The LM Head is analogous, just with rows replaced by columns. As for the gamma parameter, it is even simpler — it is directly the element-wise clip \(\mathop{\text{clip}}(\boldsymbol{\gamma};-\tau,\tau) = \max(\min(\boldsymbol{\gamma},\tau),-\tau)\).

These conclusions are all quite intuitive, and their proofs are relatively simple, so we will not expand on them — consider them exercises for the reader.

## The Necessity of Guarantees

Some readers may wonder: does it really need to be this complicated? Can't we just use ordinary weight decay like in [*Training Deep Learning Models with Norm-Constrained LMOs*](https://papers.cool/arxiv/2502.07529)? For example:

\[
\boldsymbol{W}_t = (1-\eta\lambda)\boldsymbol{W}_{t-1}  - \eta\sqrt{\frac{d_{out}}{d_{in}}}\operatorname{msign}(\boldsymbol{G}_t) \tag{6}
\]

This can also constrain the spectral norm to within \(\tau = \frac{1}{\lambda}\sqrt{\frac{d_{out}}{d_{in}}}\) — so why not use this simpler form?

The answer is: to avoid excessive intervention. From definition (2), our clipping operator is the operation that achieves the same effect with minimal modification to the original parameter. For the spectral norm, directly multiplying by \(1-\eta\lambda\) can also reduce the spectral norm of \(\boldsymbol{W}_{t-1}\) to at most \((1-\eta\lambda)\Vert\boldsymbol{W}_{t-1}\Vert_2\), but since it differs from the minimal-modification \(\operatorname{mclip}\), it inevitably involves some degree of "excessive intervention."

The consequences of excessive intervention come in two forms: either we choose a small \(\lambda\) to preserve performance, in which case \(\tau\) is too large and we cannot guarantee the spectral norm stays within our desired range; or we choose a large \(\lambda\) to control the spectral norm, but this significantly degrades performance. For instance, in the case \(d_{in}=d_{out}\) where we want the spectral norm not to exceed 5, the formula gives \(\lambda=0.2\). For Muon with equation (6), a weight decay coefficient of 0.2 is extremely large (the typical value is around 0.01).

Note that we have repeatedly emphasized "guarantee" — this is crucial. Suppose we use weight decay with a coefficient of 0.01; theoretically the spectral norm could reach up to 100, but experiments on small models might show it never even reaches 5. This is very common. However, safety on small models does not mean safety on large models. As we have said before, large models are powerful — powerful enough to amplify any subtle bug. If the theoretical upper bound is 100, a small model may never reach it, but a large model truly can.

Therefore, ensuring that critical parameter norms have a reasonable theoretical bound is absolutely necessary — this is the embodiment of "stability" in the "stability first, speed second" principle. And the clipping operator defined in equation (2) is the most "lightweight" operation that guarantees boundedness. In other words, it is likely the operation that minimizes performance loss while guaranteeing the same bounds.

## Summary

This article, based on the minimal-modification principle, proposes a general framework for maintaining parameter stability throughout training, comprising two approaches: Post Clip and Pre Decay. Under the spectral norm, these further specialize to singular value clipping and spectral weight decay. These operations aim to guarantee that critical parameter norms remain bounded while minimizing intervention in the training dynamics.

<hr class="section-divider">

*Citation: Su, J. (2026, April 24). MuP之上：4. 坚守参数的稳定性 [Beyond MuP 4: Upholding Parameter Stability]. Scientific Spaces. [https://kexue.fm/archives/11729](https://kexue.fm/archives/11729)*

*Original content licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). This translation is shared under the same license.*