---
title: "Attention Residuals"
subtitle: "Translated from [Attention Residuals 回忆录](https://kexue.fm/archives/11664) by Jianlin Su (苏剑林)"
date: 2026-03-19T00:00:00+08:00
blurb: "Replacing standard residual connections with inter-layer attention — from the initial idea through Hyper-Connections, Full AttnRes, and the practical Block AttnRes variant that achieves 25% gains for under 5% overhead."
tags: ["translation", "attention", "residual-connections", "model-architecture", "depth"]
math: true
---

*Translator's note (Opus 4.6): This is an English translation of [Attention Residuals 回忆录](https://kexue.fm/archives/11664) by Jianlin Su (苏剑林), originally published on March 19, 2026 on [Scientific Spaces (科学空间)](https://kexue.fm). The translation preserves the author's first-person voice.*

<hr class="section-divider">

This article introduces our latest work, [*Attention Residuals (AttnRes)*](https://arxiv.org/abs/2603.15031) — as the name suggests, this uses the idea of Attention to improve Residuals.

Many readers have probably heard of the Pre Norm vs. Post Norm debate, but at the end of the day that is just an "internal dispute" within Residuals themselves, and many subsequent Normalization variants are the same story. A more interesting development was [*HC (Hyper-Connections)*](https://arxiv.org/abs/2409.19606), which began exploring the route of expanding the residual stream, though perhaps due to unstable results it did not attract much attention. The later part of the story is probably well-known: at the end of last year, DeepSeek's [*mHC*](https://arxiv.org/abs/2512.24880) improved upon HC and validated its effectiveness at larger scales.

Rather than further expanding the residual stream, we chose a more radical route: directly performing Attention across layers to replace Residuals. Of course, making the full pipeline work still involved many details and much effort — here I will briefly recall the journey.

![AttnRes diagram](/assets/img/attnres-diagram.png)

## Inter-Layer Attention

As usual, let us start from [*Residuals*](https://arxiv.org/abs/1512.03385), which everyone should be very familiar with. They take the form:

\[
\boldsymbol{x}_t = \boldsymbol{x}_{t-1} + \boldsymbol{f}_t(\boldsymbol{x}_{t-1}) \tag{1}
\]

Here we adopt a different notation that reveals something deeper. Let \(\boldsymbol{y}_t = \boldsymbol{f}_t(\boldsymbol{x}_{t-1})\), so that \(\boldsymbol{x}_t = \boldsymbol{x}_{t-1} + \boldsymbol{y}_t\). Setting \(\boldsymbol{y}_0 = \boldsymbol{x}_0\), we easily get \(\boldsymbol{x}_t = \boldsymbol{y}_0 + \boldsymbol{y}_1 + \cdots + \boldsymbol{y}_t\), and so this can be equivalently written as:

\[
\boldsymbol{y}_{t+1} = \boldsymbol{f}_{t+1}(\boldsymbol{y}_0 + \boldsymbol{y}_1 + \cdots + \boldsymbol{y}_t) \tag{2}
\]

That is, from the \(\boldsymbol{y}\) perspective, Residuals compute the equal-weight sum of \(\boldsymbol{y}_0, \boldsymbol{y}_1, \ldots, \boldsymbol{y}_t\) as the input to \(\boldsymbol{f}_{t+1}\) to produce \(\boldsymbol{y}_{t+1}\). A natural generalization is to replace this with a weighted sum:

\[
\boldsymbol{y}_{t+1} = \boldsymbol{f}_{t+1}\!\left(\sum_{s=0}^{t} a_{t+1,s}\,\boldsymbol{y}_s\right) \qquad \text{where} \quad a_{t,s} \geq 0,\quad \sum_{s=0}^{t} a_{t+1,s} = 1 \tag{3}
\]

This is the seed of AttnRes. The formula adds two extra constraints on \(a_{t,s}\); let us discuss why they are necessary:

> 1. The constraint \(a_{t,s} \geq 0\) ensures that the same \(\boldsymbol{y}_s\) always contributes in the same direction across different layers, avoiding the inconsistency where one layer wants to increase \(\boldsymbol{y}_s\) while another wants to decrease it — intuitively more learning-friendly.
>
> 2. Our \(\boldsymbol{f}\) uses In Norm, which applies \(\text{RMSNorm}\) to the input first. Since \(\text{RMSNorm}(\boldsymbol{x}) = \text{RMSNorm}(c\boldsymbol{x})\) holds for all \(c > 0\), weighted averaging and weighted summation are entirely equivalent, so the constraint \(\sum_{s=0}^{t} a_{t+1,s} = 1\) does not reduce expressiveness.

## Hyper-Connections

Before diving into AttnRes, let us briefly review HC (Hyper-Connections) and prove that it too can be understood as inter-layer Attention — thereby showing that inter-layer Attention is indeed a more fundamental route. HC modifies Residuals to:

\[
\boldsymbol{X}_t = \boldsymbol{H}_t^{res}\boldsymbol{X}_{t-1} + \boldsymbol{H}_t^{post}\,\boldsymbol{f}_t(\boldsymbol{H}_t^{pre}\boldsymbol{X}_{t-1}) \tag{4}
\]

where \(\boldsymbol{X} \in \mathbb{R}^{k \times d}\), \(\boldsymbol{H}^{res} \in \mathbb{R}^{k \times k}\), \(\boldsymbol{H}^{pre} \in \mathbb{R}^{1 \times k}\), \(\boldsymbol{H}^{post} \in \mathbb{R}^{k \times 1}\), with the classic choice being \(k = 4\). In short, the state variable is expanded to \(k\) times its size; before feeding into \(\boldsymbol{f}_t\), a matrix \(\boldsymbol{H}_t^{pre}\) reduces it back to \(1\times\); after the output, \(\boldsymbol{H}_t^{post}\) expands it back to \(k\times\); finally it is added to \(\boldsymbol{x}_{t-1}\) adjusted by \(\boldsymbol{H}_t^{res}\). Without restricting the form of \(\boldsymbol{H}_t^{res}, \boldsymbol{H}_t^{pre}, \boldsymbol{H}_t^{post}\), both Post Norm and [*Highway Networks*](https://arxiv.org/abs/1505.00387) are special cases of HC.

Similarly, let \(\boldsymbol{y}_t = \boldsymbol{f}_t(\boldsymbol{H}_t^{pre}\boldsymbol{X}_{t-1})\), so \(\boldsymbol{X}_t = \boldsymbol{H}_t^{res}\boldsymbol{X}_{t-1} + \boldsymbol{H}_t^{post}\,\boldsymbol{y}_t\). Setting \(\boldsymbol{X}_0 = \boldsymbol{H}_0^{post}\boldsymbol{y}_0\), this can also be expanded as \(\boldsymbol{X}_t = \boldsymbol{H}_{t \leftarrow 1}^{res}\boldsymbol{H}_0^{post}\boldsymbol{y}_0 + \boldsymbol{H}_{t \leftarrow 2}^{res}\boldsymbol{H}_1^{post}\boldsymbol{y}_1 + \cdots + \boldsymbol{H}_{t \leftarrow t}^{res}\boldsymbol{H}_{t-1}^{post}\boldsymbol{y}_{t-1} + \boldsymbol{H}_t^{post}\boldsymbol{y}_t\), where \(\boldsymbol{H}_{t \leftarrow s}^{res}\) is defined as \(\boldsymbol{H}_t^{res}\boldsymbol{H}_{t-1}^{res}\cdots\boldsymbol{H}_{s+1}^{res}\boldsymbol{H}_s^{res}\). Further setting \(\boldsymbol{H}_{t \leftarrow t+1}^{res} = \boldsymbol{I}\), we can write:

\[
\boldsymbol{y}_{t+1} = \boldsymbol{f}_{t+1}(\boldsymbol{H}_{t+1}^{pre}\boldsymbol{x}_t) = \boldsymbol{f}_{t+1}\!\left(\sum_{s=0}^{t} \underbrace{\boldsymbol{H}_{t+1}^{pre}\boldsymbol{H}_{t \leftarrow s+1}^{res}\boldsymbol{H}_s^{post}}_{a_{t+1,s}}\,\boldsymbol{y}_s\right) \tag{5}
\]

Note that each \(\boldsymbol{H}_{t+1}^{pre}\boldsymbol{H}_{t \leftarrow s+1}^{res}\boldsymbol{H}_s^{post}\) is a \(1 \times 1\) matrix, i.e., a scalar, so this is also an inter-layer Attention form as in Equation (3). Readers familiar with [linear attention](https://kexue.fm/archives/11033) should quickly see the connection — HC is essentially DeltaNet "rotated 90 degrees." In practice, the three \(\boldsymbol{H}\) matrices are computed by simple linear layers with \(\tanh\) activation, which means the chained product \(\boldsymbol{H}_{t \leftarrow s}^{res}\) risks explosion or collapse, and non-negativity of \(a_{t+1,s}\) cannot be guaranteed.

Later, mHC made improvements: it switched all three \(\boldsymbol{H}\) matrices to Sigmoid activation, ensuring \(a_{t+1,s} \geq 0\); it then alternately normalized \(\boldsymbol{H}_t^{res}\) to be doubly stochastic, leveraging the closure of doubly stochastic matrices under multiplication to stabilize \(\boldsymbol{H}_{t \leftarrow s}^{res}\); and experiments validated the effectiveness of these changes. However, some newer experiments such as [*"Your DeepSeek mHC Might Not Need the 'm'"*](/translations/zhihu/your-deepseek-mhc-might-not-need-the-m/) show that simply setting \(\boldsymbol{H}_t^{res}\) to the identity matrix works well enough.

## Many Hands Make Light Work

Let us return to AttnRes. After recognizing its feasibility, the next question was: what form should \(a_{t+1,s}\) take? A natural idea is to follow standard [Scaled Dot-Product Attention](https://kexue.fm/archives/4765), but at the time I wanted to try something quickly, so I chose a simpler form:

\[
a_{t+1,s} \propto \exp(\boldsymbol{w}_{t+1} \cdot \boldsymbol{y}_s) \tag{6}
\]

where \(\boldsymbol{w}_t\) is a trainable vector parameter — that is, using a data-independent static vector as the Query, with both Keys and Values being \(\boldsymbol{y}_s\), and applying Softmax Attention. This was the first version of AttnRes. To our pleasant surprise, even this simple design already showed very significant improvements over standard Residuals!

After I shared the preliminary AttnRes results within the team, [@Zhang Yu](https://x.com/yzhang_cs) and [@Guangyu](https://x.com/nathancgy4) showed tremendous interest and joined in, beginning to validate on larger-scale models, with results that were consistently encouraging. Along the way, we also tried some more complex designs, finding that most of them actually performed worse than this simple version — only adding an extra \(\text{RMSNorm}\) to the Keys yielded stable gains. This gave us the final form of AttnRes:

\[
a_{t+1,s} \propto \exp(\boldsymbol{w}_{t+1} \cdot \text{RMSNorm}(\boldsymbol{y}_s)) \tag{7}
\]

However, AttnRes is after all a dense inter-layer connection scheme — is training and inference feasible at K2 scale or even larger? Encouragingly, [@V-ge](https://zhuanlan.zhihu.com/p/2017528295286133070) performed an elegant analysis and first confirmed feasibility for inference. The "stroke of genius" (点睛之笔) was precisely the static Query design that was originally chosen for convenience! This meant that once we computed \(\boldsymbol{y}_s\), we could precompute the attention \(a_{t,s}\) for all \(t > s\), giving the infrastructure team enough room to maneuver.

Unfortunately, the training engineers — such as [@Wang-ge](https://www.zhihu.com/question/2016993095078684011/answer/2017381145474508331) — after careful analysis, judged that under our current training environment, Full AttnRes was still not practical enough (to put it bluntly, we were resource-constrained). We needed a scheme that further reduced communication and memory costs, which led to the Block version below. The earlier version was accordingly renamed the Full version.

## The Block Version

Going from Full AttnRes to Block AttnRes is analogous to the classic process of linearizing quadratic Attention — all sorts of existing Efficient Attention ideas can be tried. The first thing we attempted was SWA (Sliding Window Attention), but it turned out to perform terribly in practice, even worse than plain Residuals.

After reflecting on this, I believe it can be understood as follows: Residuals are already a very strong baseline — they correspond to equal-weight summation of all state vectors. Any new design that wants to beat them must, at minimum, be able to subsume them in its formulation. Full AttnRes clearly satisfies this condition, but adding SWA does not: it discards some states and can no longer represent "equal-weight summation of all state vectors" as a special case.

This led us to realize that for AttnRes, "compression" is likely more effective than "sparsity," and the compression doesn't even need to be fine-grained — simple weighted summation may suffice. After some brainstorming and refinement, [@Zhang Yu](https://x.com/yzhang_cs) and [@Guangyu](https://x.com/nathancgy4) proposed the Block AttnRes design presented in the paper, combining block-wise processing with summation-based compression to achieve performance close to the Full version.

The idea behind Block AttnRes is roughly as follows: first, the Embedding layer is treated as its own block — because by observing the Full version's attention matrix (this is one benefit of the Attention concept: you can visualize attention patterns at any time), we found that the model tends to allocate substantial attention to the Embedding layer, so it makes sense to separate it out. The remaining layers are grouped into blocks of \(m\) layers each; within each block, states are compressed via summation, and inter-block Attention is computed over these compressed representations.

Experiments show that simply fixing the number of blocks at around 8 is enough to capture most of AttnRes's benefits. After evaluation, the training and inference engineers agreed that Block AttnRes's extra overhead is small, and entirely worthwhile relative to its gains (for detailed analysis, see the posts by [@Wang-ge](https://www.zhihu.com/question/2016993095078684011/answer/2017381145474508331) and [@V-ge](https://zhuanlan.zhihu.com/p/2017528295286133070) — in rough numbers, under 5% overhead for 25% gains). Everyone then pushed full steam ahead to integrate it into the mainline, which was another fulfilling and enjoyable experience that I won't go into here.

## The Matrix View

It is worth noting that we can use the attention matrix to unify Residuals, HC/mHC, Full AttnRes, and Block AttnRes — a rather interesting perspective. Below are examples, where \(\phi(\boldsymbol{q}, \boldsymbol{k}) = \exp(\boldsymbol{q} \cdot \text{RMSNorm}(\boldsymbol{k}))\), the Block AttnRes version corresponds to \(m = 3\), and \(\boldsymbol{y}_{s:t} = \sum_{i=s}^{t} \boldsymbol{y}_i\) (a notation we also used in [*Making Alchemy More Scientific (Part 4): New Identities, New Learning Rates*](https://kexue.fm/archives/11494)).

### Residuals

\[
\boldsymbol{A} = \begin{pmatrix} 1 \\ 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 & 1 & 1 \end{pmatrix}
\]

### HC/mHC

\[
\boldsymbol{A} = \begin{pmatrix}
\boldsymbol{H}_1^{pre}\boldsymbol{H}_0^{post} \\
\boldsymbol{H}_2^{pre}\boldsymbol{H}_{1\leftarrow 1}^{res}\boldsymbol{H}_0^{post} & \boldsymbol{H}_2^{pre}\boldsymbol{H}_1^{post} \\
\boldsymbol{H}_3^{pre}\boldsymbol{H}_{2\leftarrow 1}^{res}\boldsymbol{H}_0^{post} & \boldsymbol{H}_3^{pre}\boldsymbol{H}_{2\leftarrow 2}^{res}\boldsymbol{H}_1^{post} & \boldsymbol{H}_3^{pre}\boldsymbol{H}_2^{post} \\
\boldsymbol{H}_4^{pre}\boldsymbol{H}_{3\leftarrow 1}^{res}\boldsymbol{H}_0^{post} & \boldsymbol{H}_4^{pre}\boldsymbol{H}_{3\leftarrow 2}^{res}\boldsymbol{H}_1^{post} & \boldsymbol{H}_4^{pre}\boldsymbol{H}_{3\leftarrow 3}^{res}\boldsymbol{H}_2^{post} & \boldsymbol{H}_4^{pre}\boldsymbol{H}_3^{post} \\
\boldsymbol{H}_5^{pre}\boldsymbol{H}_{4\leftarrow 1}^{res}\boldsymbol{H}_0^{post} & \boldsymbol{H}_5^{pre}\boldsymbol{H}_{4\leftarrow 2}^{res}\boldsymbol{H}_1^{post} & \boldsymbol{H}_5^{pre}\boldsymbol{H}_{4\leftarrow 3}^{res}\boldsymbol{H}_2^{post} & \boldsymbol{H}_5^{pre}\boldsymbol{H}_{4\leftarrow 4}^{res}\boldsymbol{H}_3^{post} & \boldsymbol{H}_5^{pre}\boldsymbol{H}_4^{post} \\
\boldsymbol{H}_6^{pre}\boldsymbol{H}_{5\leftarrow 1}^{res}\boldsymbol{H}_0^{post} & \boldsymbol{H}_6^{pre}\boldsymbol{H}_{5\leftarrow 2}^{res}\boldsymbol{H}_1^{post} & \boldsymbol{H}_6^{pre}\boldsymbol{H}_{5\leftarrow 3}^{res}\boldsymbol{H}_2^{post} & \boldsymbol{H}_6^{pre}\boldsymbol{H}_{5\leftarrow 4}^{res}\boldsymbol{H}_3^{post} & \boldsymbol{H}_6^{pre}\boldsymbol{H}_{5\leftarrow 4}^{res}\boldsymbol{H}_4^{post} & \boldsymbol{H}_6^{pre}\boldsymbol{H}_5^{post} \\
\boldsymbol{H}_7^{pre}\boldsymbol{H}_{6\leftarrow 1}^{res}\boldsymbol{H}_0^{post} & \boldsymbol{H}_7^{pre}\boldsymbol{H}_{6\leftarrow 2}^{res}\boldsymbol{H}_1^{post} & \boldsymbol{H}_7^{pre}\boldsymbol{H}_{6\leftarrow 3}^{res}\boldsymbol{H}_2^{post} & \boldsymbol{H}_7^{pre}\boldsymbol{H}_{6\leftarrow 4}^{res}\boldsymbol{H}_3^{post} & \boldsymbol{H}_7^{pre}\boldsymbol{H}_{6\leftarrow 5}^{res}\boldsymbol{H}_4^{post} & \boldsymbol{H}_7^{pre}\boldsymbol{H}_{6\leftarrow 6}^{res}\boldsymbol{H}_5^{post} & \boldsymbol{H}_7^{pre}\boldsymbol{H}_6^{post}
\end{pmatrix}
\]

### Identity HC

Setting \(\boldsymbol{H}_t^{res} = \boldsymbol{I}\) for all layers[^idhc] gives \(\boldsymbol{H}_{t \leftarrow s}^{res} = \boldsymbol{I}\), and every entry simplifies to \(\boldsymbol{H}_{t+1}^{pre}\boldsymbol{H}_s^{post}\):

[^idhc]: This matrix is not from the original article — it is the special case proposed in [*"Your DeepSeek mHC Might Not Need the 'm'"*](/translations/zhihu/your-deepseek-mhc-might-not-need-the-m/).

\[
\boldsymbol{A} = \begin{pmatrix}
\boldsymbol{H}_1^{pre}\boldsymbol{H}_0^{post} \\
\boldsymbol{H}_2^{pre}\boldsymbol{H}_0^{post} & \boldsymbol{H}_2^{pre}\boldsymbol{H}_1^{post} \\
\boldsymbol{H}_3^{pre}\boldsymbol{H}_0^{post} & \boldsymbol{H}_3^{pre}\boldsymbol{H}_1^{post} & \boldsymbol{H}_3^{pre}\boldsymbol{H}_2^{post} \\
\boldsymbol{H}_4^{pre}\boldsymbol{H}_0^{post} & \boldsymbol{H}_4^{pre}\boldsymbol{H}_1^{post} & \boldsymbol{H}_4^{pre}\boldsymbol{H}_2^{post} & \boldsymbol{H}_4^{pre}\boldsymbol{H}_3^{post} \\
\boldsymbol{H}_5^{pre}\boldsymbol{H}_0^{post} & \boldsymbol{H}_5^{pre}\boldsymbol{H}_1^{post} & \boldsymbol{H}_5^{pre}\boldsymbol{H}_2^{post} & \boldsymbol{H}_5^{pre}\boldsymbol{H}_3^{post} & \boldsymbol{H}_5^{pre}\boldsymbol{H}_4^{post} \\
\boldsymbol{H}_6^{pre}\boldsymbol{H}_0^{post} & \boldsymbol{H}_6^{pre}\boldsymbol{H}_1^{post} & \boldsymbol{H}_6^{pre}\boldsymbol{H}_2^{post} & \boldsymbol{H}_6^{pre}\boldsymbol{H}_3^{post} & \boldsymbol{H}_6^{pre}\boldsymbol{H}_4^{post} & \boldsymbol{H}_6^{pre}\boldsymbol{H}_5^{post} \\
\boldsymbol{H}_7^{pre}\boldsymbol{H}_0^{post} & \boldsymbol{H}_7^{pre}\boldsymbol{H}_1^{post} & \boldsymbol{H}_7^{pre}\boldsymbol{H}_2^{post} & \boldsymbol{H}_7^{pre}\boldsymbol{H}_3^{post} & \boldsymbol{H}_7^{pre}\boldsymbol{H}_4^{post} & \boldsymbol{H}_7^{pre}\boldsymbol{H}_5^{post} & \boldsymbol{H}_7^{pre}\boldsymbol{H}_6^{post}
\end{pmatrix}
\]

### Full AttnRes

\[
\boldsymbol{A} = \begin{pmatrix}
\phi(\boldsymbol{w}_1, \boldsymbol{y}_0) \\
\phi(\boldsymbol{w}_2, \boldsymbol{y}_0) & \phi(\boldsymbol{w}_2, \boldsymbol{y}_1) \\
\phi(\boldsymbol{w}_3, \boldsymbol{y}_0) & \phi(\boldsymbol{w}_3, \boldsymbol{y}_1) & \phi(\boldsymbol{w}_3, \boldsymbol{y}_2) \\
\phi(\boldsymbol{w}_4, \boldsymbol{y}_0) & \phi(\boldsymbol{w}_4, \boldsymbol{y}_1) & \phi(\boldsymbol{w}_4, \boldsymbol{y}_2) & \phi(\boldsymbol{w}_4, \boldsymbol{y}_3) \\
\phi(\boldsymbol{w}_5, \boldsymbol{y}_0) & \phi(\boldsymbol{w}_5, \boldsymbol{y}_1) & \phi(\boldsymbol{w}_5, \boldsymbol{y}_2) & \phi(\boldsymbol{w}_5, \boldsymbol{y}_3) & \phi(\boldsymbol{w}_5, \boldsymbol{y}_4) \\
\phi(\boldsymbol{w}_6, \boldsymbol{y}_0) & \phi(\boldsymbol{w}_6, \boldsymbol{y}_1) & \phi(\boldsymbol{w}_6, \boldsymbol{y}_2) & \phi(\boldsymbol{w}_6, \boldsymbol{y}_3) & \phi(\boldsymbol{w}_6, \boldsymbol{y}_4) & \phi(\boldsymbol{w}_6, \boldsymbol{y}_5) \\
\phi(\boldsymbol{w}_7, \boldsymbol{y}_0) & \phi(\boldsymbol{w}_7, \boldsymbol{y}_1) & \phi(\boldsymbol{w}_7, \boldsymbol{y}_2) & \phi(\boldsymbol{w}_7, \boldsymbol{y}_3) & \phi(\boldsymbol{w}_7, \boldsymbol{y}_4) & \phi(\boldsymbol{w}_7, \boldsymbol{y}_5) & \phi(\boldsymbol{w}_7, \boldsymbol{y}_6)
\end{pmatrix}
\]

### Block AttnRes

In Block AttnRes (shown here with \(m = 3\)), the Embedding layer forms its own block, then every \(m\) layers are grouped together. Within each block, past layers are compressed into a single summed representation:

\[
\boldsymbol{A} = \left(\begin{array}{c:ccc:ccc}
\phi(\boldsymbol{w}_1, \boldsymbol{y}_0) \\
\hdashline
\phi(\boldsymbol{w}_2, \boldsymbol{y}_0) & \phi(\boldsymbol{w}_2, \boldsymbol{y}_1) \\
\phi(\boldsymbol{w}_3, \boldsymbol{y}_0) & \phi(\boldsymbol{w}_3, \boldsymbol{y}_{1:2}) & \phi(\boldsymbol{w}_3, \boldsymbol{y}_{1:2}) \\
\phi(\boldsymbol{w}_4, \boldsymbol{y}_0) & \phi(\boldsymbol{w}_4, \boldsymbol{y}_{1:3}) & \phi(\boldsymbol{w}_4, \boldsymbol{y}_{1:3}) & \phi(\boldsymbol{w}_4, \boldsymbol{y}_{1:3}) \\
\hdashline
\phi(\boldsymbol{w}_5, \boldsymbol{y}_0) & \phi(\boldsymbol{w}_5, \boldsymbol{y}_{1:3}) & \phi(\boldsymbol{w}_5, \boldsymbol{y}_{1:3}) & \phi(\boldsymbol{w}_5, \boldsymbol{y}_{1:3}) & \phi(\boldsymbol{w}_5, \boldsymbol{y}_4) \\
\phi(\boldsymbol{w}_6, \boldsymbol{y}_0) & \phi(\boldsymbol{w}_6, \boldsymbol{y}_{1:3}) & \phi(\boldsymbol{w}_6, \boldsymbol{y}_{1:3}) & \phi(\boldsymbol{w}_6, \boldsymbol{y}_{1:3}) & \phi(\boldsymbol{w}_6, \boldsymbol{y}_{4:5}) & \phi(\boldsymbol{w}_6, \boldsymbol{y}_{4:5}) \\
\phi(\boldsymbol{w}_7, \boldsymbol{y}_0) & \phi(\boldsymbol{w}_7, \boldsymbol{y}_{1:3}) & \phi(\boldsymbol{w}_7, \boldsymbol{y}_{1:3}) & \phi(\boldsymbol{w}_7, \boldsymbol{y}_{1:3}) & \phi(\boldsymbol{w}_7, \boldsymbol{y}_{4:6}) & \phi(\boldsymbol{w}_7, \boldsymbol{y}_{4:6}) & \phi(\boldsymbol{w}_7, \boldsymbol{y}_{4:6})
\end{array}\right)
\]

## Related Work

From the moment we decided to pursue AttnRes, my collaborators and I were immersed in the process of polishing, validating, and accelerating. Some readers may know that my research style is to first push as hard as I can on derivations and solutions, only searching for related literature once I hit a wall or finish completely. I happened to find a group of like-minded collaborators, and this time the AttnRes exploration went fairly smoothly overall, so it was not until all tests had basically passed and we began preparing the technical report that we started surveying the literature.

But precisely because of this, "you don't know what you don't know until you look" (不查不知道，一查吓一跳) — it turned out that there was already an enormous body of work on Dense Connections and Depth Attention. Beyond the classic [*DenseNet*](https://arxiv.org/abs/1608.06993), we found [*DenseFormer*](https://arxiv.org/abs/2402.02622), [*ANCRe*](https://arxiv.org/abs/2602.09009), [*MUDDFormer*](https://arxiv.org/abs/2502.12170), [*MRLA*](https://arxiv.org/abs/2302.03985), [*Dreamer*](https://arxiv.org/abs/2601.21582), and even [*ELMo*](https://arxiv.org/abs/1802.05365) from before BERT, which partially applied similar designs — all of which we included in our references.

After releasing the technical report, we gradually received reader comments pointing out additional related works we had not included, such as [*SKNets*](https://arxiv.org/abs/1903.06586), [*LIMe*](https://arxiv.org/abs/2502.09245), [*DCA*](https://arxiv.org/abs/2502.06785), etc. We apologize for the omissions and are grateful for the pointers, and we promise to add them in subsequent revisions. But whether reader or author, please remain reasonable about this — literature surveys are not easy, some omissions are inevitable, and we hold the highest respect for all related work.

At the same time, we encourage everyone to look beyond the "Depth Attention" concept and appreciate the engineering effort behind AttnRes. We fully agree that in 2026, "Depth Attention" or "Layer Attention" is by no means a novel idea. But making it work for sufficiently large models, as a strong enough replacement for Residuals while meeting training and inference efficiency requirements, is not a trivial undertaking. To the best of our knowledge, AttnRes is the first work to achieve this.

## Conclusion

This article introduced our latest result on model architecture: Attention Residuals (AttnRes). It replaces naive Residuals with inter-layer Attention, and through careful design ensures it meets training and inference efficiency requirements, ultimately scaling it successfully to sufficiently large models.

<hr class="section-divider">

*Citation: Su, J. (2026, March 19). Attention Residuals 回忆录 [Attention Residuals]. Scientific Spaces. [https://kexue.fm/archives/11664](https://kexue.fm/archives/11664)*

*Original content licensed under [CC BY-NC-ND 2.5 CN](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/). This translation is shared under the same license.*
