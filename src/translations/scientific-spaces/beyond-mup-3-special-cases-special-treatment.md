---
title: "Beyond MuP: 3. Special Cases, Special Treatment"
subtitle: "Translated from [MuP之上：3. 特殊情况特殊处理](https://kexue.fm/archives/11647) by Jianlin Su (苏剑林)"
date: 2026-03-02T00:00:00+08:00
blurb: "Embedding layers, LM Heads, and RMS Norm parameters each need their own stability analysis. Starting from three stability metrics, we derive the right initialization and steepest descent optimizer for each — and explain why Muon doesn't apply to all matrix parameters."
tags: ["translation", "mup", "muon", "optimization", "scaling"]
math: true
---

*Translator's note (Opus 4.6): This is an English translation of [MuP之上：3. 特殊情况特殊处理](https://kexue.fm/archives/11647) by Jianlin Su (苏剑林), originally published on March 2, 2026 on [Scientific Spaces (科学空间)](https://kexue.fm). It is the third article in the "Beyond MuP" series. The translation preserves the author's first-person voice.*

<hr class="section-divider">

After so many related blog posts, most readers should be no stranger to the Muon optimizer — even without knowing the theoretical details, you have probably come away with the impression that it is "an optimizer custom-built for matrix parameters." However, this characterization is not entirely correct — for instance, the input-side Embedding layer and the output-side LM Head both have matrix parameters, yet neither is suited for Muon (see [*Muon Optimizer Guide: Quick Start and Key Details*](https://kexue.fm/archives/11416)).

Why must they be "treated differently"? This article will continue using the three stability metrics proposed in the first installment, exploring the initialization rules and corresponding steepest descent directions for different types of layers, thereby answering this question.

## Recap

In the first article [*Beyond MuP: 1. Three Characteristics of Good Models*](/translations/scientific-spaces/beyond-mup-1-three-characteristics-of-good-models/), we proposed three stability metrics:

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

The three metrics share a unified format: compute the RMS over the output, then take the \(\max\) over the input. Here \(\boldsymbol{x}\) denotes the input, \(\boldsymbol{\omega}\) denotes the parameters, and \(\boldsymbol{f}(\boldsymbol{x};\boldsymbol{\omega})\) can represent a single layer, a block, or even the entire model — depending on our ability to compute the \(\max\).

Since we do not restrict the domain of \(\boldsymbol{x}\), the maximum may not always exist. To handle this, we sometimes need to augment the model with additional operations, which in turn guides model design. For example, in the previous article [*Beyond MuP: 2. Linear Layers and Steepest Descent*](/translations/scientific-spaces/beyond-mup-2-linear-layers-and-steepest-descent/), we added In Norm to the linear layer in order to compute its stability metrics. Furthermore, by combining this with the idea of steepest descent, we also rederived the Muon optimizer.

Steepest descent itself is not a new concept — it answers the question "given a stability metric, what optimizer should we use?" The core contribution of the "Beyond MuP" series is answering a different question: "what stability metric should we use?" It provides stability metric formulas that apply to arbitrary layers.

## The Embedding Layer

Now we consider the Embedding layer, arguably the simplest layer of all. The input is an index \(i\), and the output is the corresponding vector, i.e., \(\boldsymbol{f}(i;\boldsymbol{E}) = \boldsymbol{E}_i\), where \(\boldsymbol{E}\) is a \(|V|\times d\) matrix and \(\boldsymbol{E}_i \triangleq \boldsymbol{E}_{i,:}\) denotes the \(i\)-th row of \(\boldsymbol{E}\). It is easy to compute:

**Forward Stability:**

\[
\max_i \|\boldsymbol{E}_i\|_{\text{RMS}} = \Theta(1)
\]

**Dependency Stability:**

\[
\max_{i,j} \|\boldsymbol{E}_i - \boldsymbol{E}_j\|_{\text{RMS}} = \Theta(1)
\]

**Update Stability:**

\[
\max_i \| \Delta \boldsymbol{E}_i\|_{\text{RMS}} = \Theta(1) \tag{4}
\]

Note that \(\max_{i,j} \|\boldsymbol{E}_i - \boldsymbol{E}_j\|_{\text{RMS}} \leq 2 \max_i \|\boldsymbol{E}_i\|_{\text{RMS}}\), so all of these are essentially the maximum row norm (divided by \(\sqrt{d}\)) of \(\boldsymbol{E}\) or \(\Delta\boldsymbol{E}\). Forward stability and dependency stability serve only to guide initialization: they tell us to initialize \(\boldsymbol{E}\) with zero mean and \(\Theta(1)\) variance.

As for update stability, Equation \((4)\) tells us that although the Embedding layer's parameter is also a matrix, the right metric for measuring "stability" should not be the spectral norm but rather the maximum row norm. This means its steepest descent is not Muon. To find the steepest descent for the Embedding layer, we need to solve the optimization problem

\[
\min_{\Delta \boldsymbol{E}} \langle\boldsymbol{G},\Delta\boldsymbol{E}\rangle \qquad \text{s.t.}\qquad \max_i \underbrace{\|\Delta\boldsymbol{E}_i\|_{\text{RMS}}}_{\|\Delta\boldsymbol{E}_i\|_2/\sqrt{d}}\leq\eta
\]

This problem is not hard to solve — we just need the Cauchy-Schwarz inequality:

\[
\langle\boldsymbol{G},\Delta\boldsymbol{E}\rangle = \sum_{i=1}^{|V|}\langle\boldsymbol{G}_i,\Delta\boldsymbol{E}_i\rangle \geq -\sum_{i=1}^{|V|}\|\boldsymbol{G}_i\|_2 \times \|\Delta\boldsymbol{E}_i\|_2 \geq -\eta\sqrt{d}\sum_{i=1}^{|V|}\|\boldsymbol{G}_i\|_2
\]

Equality holds when \(\Delta\boldsymbol{E}_i = - \eta\boldsymbol{G}_i / \|\boldsymbol{G}_i\|_{\text{RMS}}\). In other words, the steepest descent for the Embedding layer is row-wise RMS Norm of the gradient (Normalized SGD).

## The Output Head

Next, let us look at the LM Head. On the surface, this is just another linear layer: the input is \(\boldsymbol{x}\in\mathbb{R}^d\), the weight is \(\boldsymbol{W}\in\mathbb{R}^{d\times |V|}\), the output is \(\boldsymbol{x}\boldsymbol{W}\in\mathbb{R}^{|V|}\), and \(\boldsymbol{x}\) typically comes with RMS Norm — everything looks like a linear layer. So why doesn't Muon apply?

### Accountable to the Loss

The answer is that **the LM Head must be accountable to the loss**.

Keep in mind that steepest descent serves training. From the inference perspective, a model takes in several tokens and predicts the next one. But from the training perspective, the true "model" takes in several tokens *and* the next token, and outputs the loss. In other words, both the data and the label are inputs, and the true output is the loss. For earlier layers, we can ignore the label and the loss, but the LM Head — being the final layer that "borders" the loss — has no such luxury.

So the LM Head's input becomes \(\boldsymbol{x}\) and the next token's index \(t\), and the output becomes the cross-entropy loss:

\[
\ell(\boldsymbol{x},t;\boldsymbol{W}) = \log\sum_{i=1}^{|V|} e^{\langle \boldsymbol{x},\boldsymbol{w}_i\rangle} - \langle \boldsymbol{x},\boldsymbol{w}_t\rangle = \log\sum_{i=1}^{|V|} e^{\langle \boldsymbol{x},\boldsymbol{w}_i - \boldsymbol{w}_t\rangle}
\]

where \(\boldsymbol{w}_i\triangleq \boldsymbol{W}_{:, i}\) is the \(i\)-th column of \(\boldsymbol{W}\). Since \(\ell\) is a complex nonlinear function of \(\boldsymbol{x}, t, \boldsymbol{W}\), its three metrics cannot be computed exactly. Our goal is to find bounds that are as tight as possible.

### Forward Stability

We begin with the simpler forward stability. A straightforward bound gives

\[
\begin{aligned}
\ell(\boldsymbol{x},t;\boldsymbol{W}) = \log\sum_{i=1}^{|V|} e^{\langle \boldsymbol{x},\boldsymbol{w}_i - \boldsymbol{w}_t\rangle} \leq&\, \log \left(|V| \max_i e^{\langle \boldsymbol{x},\boldsymbol{w}_i - \boldsymbol{w}_t\rangle}\right) \\
=&\, \log |V| + \max_i \langle \boldsymbol{x},\boldsymbol{w}_i - \boldsymbol{w}_t\rangle \\
\leq &\, \log |V| + \max_i \|\boldsymbol{x}\|_2 \|\boldsymbol{w}_i - \boldsymbol{w}_t\|_2
\end{aligned}
\]

Therefore

\[
\begin{aligned}
\text{Forward Stability:}\quad\max_{t, \|\boldsymbol{x}\|_{\text{RMS}}=1} \ell(\boldsymbol{x},t;\boldsymbol{W}) \leq&\, \log |V| + d\max_{i,t} \|\boldsymbol{w}_i - \boldsymbol{w}_t\|_{\text{RMS}} \\
\leq&\, \log |V| + 2d\max_i \|\boldsymbol{w}_i\|_{\text{RMS}}
\end{aligned}
\]

If we drop the constant \(\log|V|\), this becomes a lower bound, so the bound is fairly tight in describing the dependence on \(\boldsymbol{W}\). For this to be \(\Theta(1)\), the initialization variance of the LM Head should be \(\Theta(1/d^2)\).

### A Key Inequality

For the remaining two metrics, which involve differences, the computation is more involved. We first prove the following inequality:

\[
\left|\log\sum_{i=1}^n e^{a_i} - \log\sum_{i=1}^n e^{b_i}\right| \leq \max_i |a_i - b_i| \tag{5}
\]

The proof is not difficult. Let the right-hand side be \(M\). By the monotonicity of \(\log, \sum, \exp\), we readily get

\[
\log\sum_{i=1}^n e^{a_i} = \log\sum_{i=1}^n e^{(a_i - b_i)+b_i} \leq \log\sum_{i=1}^n e^{M + b_i} = M + \log\sum_{i=1}^n e^{b_i}
\]

This proves

\[
\log\sum_{i=1}^n e^{a_i} - \log\sum_{i=1}^n e^{b_i} \leq M
\]

By symmetry, swapping \(a_i\) and \(b_i\) also holds, which completes the proof of the original inequality.

### Dependency Stability

Using inequality \((5)\) and the Cauchy-Schwarz inequality, we get

\[
\begin{aligned}
|\ell(\boldsymbol{x}_1,t_1;\boldsymbol{W}) - \ell(\boldsymbol{x}_2,t_2;\boldsymbol{W})| \leq&\, \max_i |\langle \boldsymbol{x}_1,\boldsymbol{w}_i - \boldsymbol{w}_{t_1}\rangle - \langle \boldsymbol{x}_2,\boldsymbol{w}_i - \boldsymbol{w}_{t_2}\rangle| \\
\leq&\, \max_i (\|\boldsymbol{x}_1\|_2 \|\boldsymbol{w}_i - \boldsymbol{w}_{t_1}\|_2 + \|\boldsymbol{x}_2\|_2 \|\boldsymbol{w}_i - \boldsymbol{w}_{t_2}\|_2) \\
=&\, d\max_i (\|\boldsymbol{x}_1\|_{\text{RMS}} \|\boldsymbol{w}_i - \boldsymbol{w}_{t_1}\|_{\text{RMS}} + \|\boldsymbol{x}_2\|_{\text{RMS}} \|\boldsymbol{w}_i - \boldsymbol{w}_{t_2}\|_{\text{RMS}})
\end{aligned}
\]

Therefore

\[
\begin{aligned}
&\text{Dependency Stability:} \\
&\max_{\substack{t_1, t_2, \\ \|\boldsymbol{x}_1\|_{\text{RMS}}=1 \\ \|\boldsymbol{x}_2\|_{\text{RMS}}=1}} |\ell(\boldsymbol{x}_1,t_1;\boldsymbol{W}) - \ell(\boldsymbol{x}_2,t_2;\boldsymbol{W})| \leq d\max_{i,t_1,t_2} (\|\boldsymbol{w}_i - \boldsymbol{w}_{t_1}\|_{\text{RMS}} + \|\boldsymbol{w}_i - \boldsymbol{w}_{t_2}\|_{\text{RMS}}) \\
&\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\quad\;\;\leq 4d\max_i \|\boldsymbol{w}_i\|_{\text{RMS}}
\end{aligned}
\]

This is consistent with the forward stability result.

### Update Stability

Finally, for update stability, we again use inequality \((5)\) and Cauchy-Schwarz:

\[
\begin{aligned}
|\ell(\boldsymbol{x},t;\boldsymbol{W} + \Delta\boldsymbol{W}) - \ell(\boldsymbol{x},t;\boldsymbol{W})| \leq&\, \max_i |\langle \boldsymbol{x},\boldsymbol{w}_i + \Delta\boldsymbol{w}_i - \boldsymbol{w}_t - \Delta\boldsymbol{w}_t\rangle - \langle \boldsymbol{x},\boldsymbol{w}_i - \boldsymbol{w}_t\rangle| \\
=&\, \max_i |\langle \boldsymbol{x},\Delta\boldsymbol{w}_i - \Delta\boldsymbol{w}_t\rangle| \\
\leq &\, d \max_i  \|\boldsymbol{x}\|_{\text{RMS}} \|\Delta\boldsymbol{w}_i - \Delta\boldsymbol{w}_t\|_{\text{RMS}}
\end{aligned}
\]

Therefore

\[
\begin{aligned}
&\text{Update Stability:} \\
&\max_{t,\|\boldsymbol{x}\|_{\text{RMS}}=1} |\ell(\boldsymbol{x},t;\boldsymbol{W} + \Delta\boldsymbol{W}) - \ell(\boldsymbol{x},t;\boldsymbol{W})| \leq d \max_{i, t} \|\Delta\boldsymbol{w}_i - \Delta\boldsymbol{w}_t\|_{\text{RMS}} \\
&\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\quad\;\;\leq 2d\max_i \|\Delta\boldsymbol{w}_i\|_{\text{RMS}}
\end{aligned}
\]

It is easy to see that the three stability metrics for the LM Head are essentially the same as for the Embedding layer — both reduce to the maximum row/column norm of the parameter matrix or its increment. This means the steepest descent for the LM Head is also Normalized SGD, except that the LM Head normalizes column-wise. Additionally, the Embedding layer's initialization standard deviation and learning rate both scale as \(\Theta(1)\), whereas for the LM Head they are \(\Theta(1/d)\), which means they differ slightly when transferring across widths.

## Other Modules

Besides linear layers, Embedding, and the LM Head, a typical Transformer model has several other parameters or layers that need individual analysis. Let us go through them one by one.

### The Hadamard Product

We know that after RMS Norm there is usually a \(\boldsymbol{\gamma}\) vector multiplied element-wise (Hadamard product), i.e., \((\boldsymbol{x} / \|\boldsymbol{x}\|_{\text{RMS}})\odot\boldsymbol{\gamma}\), to control the output scale. This parameter is not a matrix, so neither Muon nor Normalized SGD as described above applies directly.

One could compute the three stability metrics for \(\boldsymbol{\gamma}\) from scratch, but there is a more elegant shortcut: notice that \(\boldsymbol{x}\odot\boldsymbol{\gamma}=\boldsymbol{x}\,\text{diag}(\boldsymbol{\gamma})\), i.e., the Hadamard product of \(\boldsymbol{x}\) with \(\boldsymbol{\gamma}\) equals the matrix product of \(\boldsymbol{x}\) with the diagonal matrix \(\text{diag}(\boldsymbol{\gamma})\). This turns it into a special linear layer with \(\boldsymbol{W}=\text{diag}(\boldsymbol{\gamma})\), and we can reuse the linear layer results.

From the previous article, the initial spectral norm of \(\boldsymbol{W}\) should be \(\Theta(\sqrt{d_{\text{out}}/d_{\text{in}}})\). Here \(\boldsymbol{W}\) is a square matrix, so this is simply \(\Theta(1)\). Since \(\boldsymbol{W}\) is also diagonal, we can simply initialize \(\boldsymbol{W}\) as the identity matrix, which corresponds to initializing \(\boldsymbol{\gamma}\) to all ones.

As for the optimizer, let \(\boldsymbol{g}\) be the gradient of \(\boldsymbol{\gamma}\). Then the gradient of \(\boldsymbol{W}\) is \(\boldsymbol{G}=\text{diag}(\boldsymbol{g})\). The steepest descent for linear layers is Muon, i.e., \(\Delta\boldsymbol{W}=-\eta\,\text{msign}(\boldsymbol{G})\). For a diagonal matrix, \(\text{msign}(\boldsymbol{G})=\text{sign}(\boldsymbol{G})=\text{diag}(\text{sign}(\boldsymbol{g}))\), so the steepest descent for the \(\boldsymbol{\gamma}\) parameter is SignSGD.

### Linear Bias Terms

Traditional linear layers usually have a bias vector \(\boldsymbol{b}\), i.e., the full linear operation is \(\boldsymbol{f}(\boldsymbol{x};\boldsymbol{W},\boldsymbol{b}) = \boldsymbol{x}\boldsymbol{W}+\boldsymbol{b}\). However, most recent open-source models have dropped the bias term, making it largely irrelevant in practice. For completeness, let us still discuss it here.

With the bias vector included, the three stability metrics become:

**Forward Stability:**

\[
\max_{\|\boldsymbol{x}\|_{\text{RMS}}=1} \| \boldsymbol{x}\boldsymbol{W} + \boldsymbol{b}\|_{\text{RMS}}
\]

**Dependency Stability:**

\[
\max_{\|\boldsymbol{x}_1\|_{\text{RMS}}=\|\boldsymbol{x}_2\|_{\text{RMS}}=1} \| \boldsymbol{x}_1\boldsymbol{W} - \boldsymbol{x}_2\boldsymbol{W}\|_{\text{RMS}}
\]

**Update Stability:**

\[
\max_{\|\boldsymbol{x}\|_{\text{RMS}}=1} \| \boldsymbol{x} \Delta\boldsymbol{W} + \Delta\boldsymbol{b}\|_{\text{RMS}}
\]

The dependency stability is the same as without bias, so we only need to examine forward stability and update stability. For simplicity, we use the inequality \(\| \boldsymbol{x}\boldsymbol{W} + \boldsymbol{b}\|_{\text{RMS}}\leq \| \boldsymbol{x}\boldsymbol{W}\|_{\text{RMS}} + \|\boldsymbol{b}\|_{\text{RMS}}\). Assuming \(\boldsymbol{W}\) uses its original initialization, the \(\| \boldsymbol{x}\boldsymbol{W}\|_{\text{RMS}}\) part already achieves \(\Theta(1)\), so we only need \(\|\boldsymbol{b}\|_{\text{RMS}}=\mathcal{O}(1)\). In practice this is typically achieved by zero-initializing \(\boldsymbol{b}\).

Similarly, \(\| \boldsymbol{x}\Delta\boldsymbol{W} + \Delta\boldsymbol{b}\|_{\text{RMS}}\leq \| \boldsymbol{x}\Delta\boldsymbol{W}\|_{\text{RMS}} + \|\Delta\boldsymbol{b}\|_{\text{RMS}}\). If we require \(\|\Delta\boldsymbol{b}\|_{\text{RMS}}=\mathcal{O}(1)\), then the \(\boldsymbol{b}\) parameter uses \(\|\Delta\boldsymbol{b}\|_{\text{RMS}}\) as its stability metric for steepest descent, yielding Normalized SGD as well.

### Attention Scaling

Using the forward stability metric, we can also rederive the scaling factor in the Attention mechanism. Let \(\boldsymbol{q}=\boldsymbol{x}\boldsymbol{W}_q\) and \(\boldsymbol{k}=\boldsymbol{x}\boldsymbol{W}_k\). If \(\boldsymbol{W}_q\) and \(\boldsymbol{W}_k\) are treated as linear layers, we can assume \(\|\boldsymbol{q}\|_{\text{RMS}}=\Theta(1)\) and \(\|\boldsymbol{k}\|_{\text{RMS}}=\Theta(1)\). Then by Cauchy-Schwarz:

\[
|\langle\boldsymbol{q},\boldsymbol{k}\rangle| \leq \|\boldsymbol{q}\|_2 \|\boldsymbol{k}\|_2 = d\|\boldsymbol{q}\|_{\text{RMS}} \|\boldsymbol{k}\|_{\text{RMS}}
\]

where \(d\) here is the dimension of \(\boldsymbol{q}\) and \(\boldsymbol{k}\), i.e., the head dimension. Clearly this is \(\Theta(d)\). To make it \(\Theta(1)\), we need to multiply \(\boldsymbol{q}\cdot\boldsymbol{k}\) by a scaling factor of order \(\Theta(1/d)\). This differs from the earlier \(1/\sqrt{d}\) (see [*A Brief Discussion of Transformer Initialization, Parametrization, and Normalization*](https://kexue.fm/archives/8620)).

So which is correct? Both are. The factor \(1/\sqrt{d}\) is the average result under random initialization, while \(\Theta(1/d)\) is the extreme-case bound valid throughout training. The latter does not mean we should directly change the scaling factor to \(1/d\), but rather that scaling inversely with \(d\) may yield better transferability. The two are compatible. For example, if \(1/\sqrt{128}\) works well at \(d=128\), then when transferring to \(d=256\), one might use \(1/(2\sqrt{128})\) rather than \(1/\sqrt{256}\).

In practice, constrained by Flash Attention, the head dimension has very limited choices — typically 128, at most 256 — so there is almost no practical need for cross-head-dim parameter transfer. This result is therefore mostly of theoretical interest.

## Summary

Finally, the main results from these two articles are summarized as follows:

**Linear layer** (\(\boldsymbol{W}\in\mathbb{R}^{d_{\text{in}}\times d_{\text{out}}}\), \(\boldsymbol{b}\in\mathbb{R}^{d_{\text{out}}}\)):

- Input: \(\boldsymbol{x}\). Output: \(\boldsymbol{x}\boldsymbol{W} + \boldsymbol{b}\).
- Init variance of \(\boldsymbol{W}\): \(\sqrt{\frac{d_{\text{out}}}{d_{\text{in}}}}\frac{1}{\sqrt{d_{\text{in}}} + \sqrt{d_{\text{out}}}}\). Init of \(\boldsymbol{b}\): \(0\).
- Steepest descent: \(\Delta\boldsymbol{W} = -\eta\sqrt{\frac{d_{\text{out}}}{d_{\text{in}}}}\,\text{msign}(\boldsymbol{G})\), \(\Delta\boldsymbol{b} = -\eta \frac{\boldsymbol{g}}{\lVert\boldsymbol{g}\rVert_{\text{RMS}}}\).

**Embedding** (\(\boldsymbol{E} \in\mathbb{R}^{\lvert V\rvert\times d}\)):

- Input: \(i\). Output: \(\boldsymbol{E}_{i,:}\).
- Init variance: \(1\).
- Steepest descent: \(\Delta\boldsymbol{E}_{i,:} = -\eta \frac{\boldsymbol{G}_{i,:}}{\lVert\boldsymbol{G}_{i,:}\rVert_{\text{RMS}}}\).

**LM Head** (\(\boldsymbol{W} \in\mathbb{R}^{d \times \lvert V\rvert}\)):

- Input: \(\boldsymbol{x}, t\). Output: \(\log\sum_{i=1}^{\lvert V\rvert} e^{\langle \boldsymbol{x},\boldsymbol{W}_{:,i} - \boldsymbol{W}_{:,t}\rangle}\).
- Init variance: \(\frac{1}{d^2}\).
- Steepest descent: \(\Delta\boldsymbol{W}_{:,i} = -\frac{\eta}{d} \frac{\boldsymbol{G}_{:,i}}{\lVert\boldsymbol{G}_{:,i}\rVert_{\text{RMS}}}\).

**RMS Norm** (\(\boldsymbol{\gamma} \in\mathbb{R}^d\)):

- Input: \(\boldsymbol{x}\). Output: \(\frac{\boldsymbol{x}}{\lVert\boldsymbol{x}\rVert_{\text{RMS}}}\odot\boldsymbol{\gamma}\).
- Init variance: \(1\).
- Steepest descent: \(\Delta\boldsymbol{\gamma} = -\eta\,\text{sign}(\boldsymbol{g})\).

The steepest descent directions for Embedding and LM Head are row-wise and column-wise Normalized SGD respectively, which is consistent with works such as [*Scion*](https://arxiv.org/abs/2502.07529). As for the variance and learning rate transfer rules, they are consistent with the conclusions of [MuP](https://kexue.fm/archives/10795). In these two articles, everything was derived from the "three stability metrics" we proposed, which shows that we have indeed found a unified form for stability measurement applicable to arbitrary layers.

<hr class="section-divider">

*Citation: Su, J. (2026, March 02). MuP之上：3. 特殊情况特殊处理 [Beyond MuP: 3. Special Cases, Special Treatment]. Scientific Spaces. [https://kexue.fm/archives/11647](https://kexue.fm/archives/11647)*
