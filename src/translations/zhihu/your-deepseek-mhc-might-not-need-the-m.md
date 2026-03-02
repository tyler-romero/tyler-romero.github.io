---
title: "Your DeepSeek mHC Might Not Need the \"m\""
subtitle: "Translated from [你的deepseek mHC可能不需要\"m\"](https://zhuanlan.zhihu.com/p/2010852389670908320) by [涮月亮的谪仙人](https://www.zhihu.com/people/shuan-yue-liang-de-zhe-xian-ren)"
date: 2026-02-27T17:04:22.000Z
blurb: "Replacing the learned doubly stochastic H_res in DeepSeek's manifold Hyper-Connections with a plain identity matrix yields better results, while eliminating Sinkhorn-Knopp iterations entirely."
tags: ["translation", "deepseek", "hyper-connections", "residual-connections", "transformers"]
math: true
code: true
---

*Translator's note (Opus 4.6): This is an English translation of [你的deepseek mHC可能不需要"m"](https://zhuanlan.zhihu.com/p/2010852389670908320) by [涮月亮的谪仙人](https://www.zhihu.com/people/shuan-yue-liang-de-zhe-xian-ren), originally published on February 27, 2026 on [Zhihu (知乎)](https://www.zhihu.com). The translation preserves the author's informal, first-person voice.*

<hr class="section-divider">

Over the past few weeks I've been studying the advanced techniques from DeepSeek's [mHC](https://arxiv.org/abs/2512.24880) and stumbled upon an experimental conclusion that's both amusing and embarrassing:

> The paper's most critical algorithmic improvement is applying a manifold constraint to \(H^{res}\), using Sinkhorn-Knopp to constrain it to a doubly stochastic matrix so that forward and backward propagation remain stable.
>
> A natural question is: is \(H^{res}\) actually necessary?
>
> We found that simply replacing it with the identity works noticeably better — just \(H^{res} = I\), haha!

![Qwen 1.7B from scratch, 150B tokens — partial training curves comparing Identity HC (blue) vs mHC (red)](/assets/img/mhc-identity-vs-sinkhorn.jpg)

***Our current conclusion is Identity HC > mHC > mHC lite > mHC orthogonal (e.g., Cayley orthogonal).*** The experiments were run on Qwen3 1.7B and 8B dense models, 150B tokens. Hopefully there are no Megatron coding bugs...

The identity matrix (diagonal entries all 1) also has row and column sums of 1, spectral norm of 1, and is completely norm-preserving — the simplest possible "manifold constraint." Intuitively, an identity \(H^{res}\) means each residual stream preserves its own information without exchanging with other streams. In practice, we observed that the \(H^{res}\) learned by the original mHC approximately follows this pattern:

- **Single-layer** \(H^{res}\) (depth=1): close to identity (diagonal ~0.96, off-diagonal ~0.01)
- **Cumulative product** (depth ≥ 10): collapses to a uniform 0.25 matrix (uniform mixing matrix)

![Multi-layer H_res accumulation](/assets/img/mhc-cumulative-hres.jpg)

In other words, the single-layer \(H^{res}\) learned by Sinkhorn-Knopp is closer to the identity matrix, but after multiplying across many layers it becomes a uniform 0.25 matrix.

The mathematical reason behind this is: when doubly stochastic matrices satisfy a uniform positivity condition (i.e., all elements have a positive lower bound \(\delta > 0\)), their Dobrushin ergodicity coefficient \(\tau(P) \leq 1 - d\delta < 1\), and the ergodicity coefficient of the cumulative product decays geometrically: \(\tau(A_n) \leq (1 - d\delta)^n \to 0\), forcing all rows to converge, ultimately collapsing to the uniform matrix \((1/d) \cdot \mathbf{1}\mathbf{1}^T\). (A rigorous proof follows from the sub-multiplicativity of the Dobrushin ergodicity coefficient. Note that if the matrix sequence consists of pure permutation matrices or is reducible, the uniform positivity condition is not satisfied and collapse does not occur — but matrices output by Sinkhorn are typically strictly positive and do satisfy this condition.)

By directly setting it to identity, we lock in the single-layer behavior the model was already learning (close to permutation), while eliminating the drawback of doubly stochastic cumulative products collapsing to a rank-1 uniform mixture.

Some might argue: if \(H^{res}\) degenerates to a permutation matrix, is that good?

In practice, mHC's \(H^{res}\) at different layers may learn different approximate permutation matrices — for example, layer 1 maps (1,2,3,4) → (3,1,4,2), while layer 5 maps (1,2,3,4) → (2,4,1,3). When these different permutations are multiplied together, the result is yet another permutation. Each layer reshuffles the streams, disrupting semantic consistency across streams. Stream 1 ends up at stream 3's position after layer 1, then at stream 2's position after layer 5. \(H^{pre}\) and \(H^{post}\) need to "track" where each stream ends up after repeated reshuffling, which increases learning difficulty.

The advantage of identity may lie in using the same identity permutation across all layers:

- Stream 0 is always at position 0, stream 1 is always at position 1 — stream semantics are completely consistent across depth
- \(H^{pre}\) and \(H^{post}\) don't need to adapt to stream reshuffling; they directly learn "which stream to read from, which stream to write to"
- Cumulative product \(I^L = I\) — no collapse, no confusion

Identity should be the most intuitive, most straightforward \(H^{res}\) implementation. I'm not sure why neither the original HC paper nor the mHC paper included this ablation. Perhaps they did run it, but for such a critical ablation on the necessity of the \(H^{res}\) matrix, I feel it warrants being written up.

<hr class="section-divider">

## Background on Hyper-Connections and mHC

The standard residual connection in a Transformer is an identity mapping plus a transformation output (thank you ResNet):

\[
\mathbf{x}_{l+1} = \mathbf{x}_l + f(\mathbf{x}_l, \mathcal{W}_l)
\]

Hyper-Connections (HC) expand the residual stream from 1 to \(n\) streams (default \(n = 4\)), with the update formula becoming:

\[
\mathbf{x}_{l+1} = \mathcal{H}^{\text{res}}_l \cdot \mathbf{x}_l + \mathcal{H}^{\text{post}\,\top}_l \cdot f(\mathcal{H}^{\text{pre}}_l \cdot \mathbf{x}_l, \mathcal{W}_l)
\]

where \(\mathbf{x}_l \in \mathbb{R}^{n \times C}\) represents \(n\) parallel residual streams, and the three learnable mappings are:

| Mapping | Dimensions | Role |
|---|---|---|
| \(H^{pre}\) | \(n\) streams → 1 stream | Read from \(n\) streams |
| \(H^{post}\) | 1 stream → \(n\) streams | Write back to \(n\) streams |
| \(H^{res}\) | \(n\) streams → \(n\) streams | Inter-stream information mixing |

DeepSeek's core innovation is constraining \(\mathcal{H}^{\text{res}}\) to the doubly stochastic manifold — matrix row and column sums equal 1 — for norm preservation, so that multi-layer \(H\) products don't cause signal explosion. Though it doesn't guarantee against signal vanishing either, haha — although in my experiments mHC's spectral norm does stay roughly around 1. Personally, I think the best contribution of the mHC paper is that it wrote the HC formula clearly — the original HC paper was genuinely hard to follow. Also, switching the activation function from tanh to sigmoid so the range is non-negative is a good practice.

The Sinkhorn-Knopp iteration in the paper still looks fairly expensive. The key question is: how important is \(H^{res}\) really?

<hr class="section-divider">

## \(\phi\) Is the Real Key to Cross-Stream Mixing

It first flattens the \(n\) residual streams (\(nC\)), then projects everything into a fusion matrix \(H\) of dimension \(n^2 + 2n = 24\), which is then split into \(H^{pre}\), \(H^{post}\), and \(H^{res}\) (4, 4, and 4×4 matrices).

Even when \(H^{res} = I\), the \(H^{pre}\) generated by \(\phi\) is still **input-dependent**:

```
Input: x_l = [s, b, n*C]  (4 streams flattened)

Step 1: φ projection (cross-stream information fusion already happens here)
   x̂' · φ → [s, b, 2n]  (in identity mode, only project 2n=8 dimensions)
   φ's input contains information from all 4 streams

Step 2: Activation
   h_pre = sigmoid(α_pre · proj[:n] + b[:n])         ← dynamic aggregation weights
   h_post = 2·sigmoid(α_post · proj[n:2n] + b[n:2n]) ← dynamic expansion weights

Step 3: Aggregation (explicit manifestation of cross-stream fusion)
   aggregated = Σ h_pre_i · x_stream_i    ← [s, b, C]

Step 4: Transformation
   output = f(aggregated)    ← Attention or MLP

Step 5: Identity residual + dynamic write-back
   x_{l+1} = I · x_l + diag(h_post) · f(...)
           = x_l + diag(h_post) · f(...)  ← each stream independently maintains residual
```

In practice, the doubly stochastic constraint on \(H^{res}\) is not only unhelpful — it may be harmful, since the cumulative product of doubly stochastic matrices inevitably collapses.

By the Perron-Frobenius theorem, the eigenvalues of a doubly stochastic matrix \(H\) satisfy: the largest eigenvalue \(\lambda_1 = 1\), and the rest \(|\lambda_i| < 1\) (as long as \(H\) is not a reducible permutation matrix). The minimum singular value of the \(L\)-layer cumulative product:

\[
\sigma_{\min}\left(\prod_{l=1}^L H_l\right) \lesssim \prod_{l=1}^L |\lambda_{\min}(H_l)|
\]

When \(|\lambda_{\min}| < 1\), this product decays exponentially to zero. In experiments on Qwen3-1.7B (28 layers, 56 HC modules), the mean minimum eigenvalue of the Sinkhorn version's \(H^{res}\) was 0.49. Rough estimate:

\[
\sigma_{\min} \sim 0.49^{56} \approx 10^{-17}
\]

The measured value was **9.2 × 10⁻¹⁸** — matching the estimate. This means that after passing through 56 HC modules, shallow-layer signals have likely decayed to nothing in all directions except the mean. Though this might not necessarily be a bad thing...

You can also see this in the cumulative \(H^{res}\) plot at the top: after 10 layers, the information from all 4 streams has completely collapsed to a uniform mean of 0.25 — mixing ultimately becomes homogenization.

The 20-step Sinkhorn-Knopp iteration used to approximately project onto the doubly stochastic manifold doesn't guarantee convergence: we measured a row-sum standard deviation of 0.12, and this error accumulates across layers. mHC-lite also reports that approximately 27.9% of Sinkhorn inputs have relative range \(1/\nu \geq 10^{13}\), at which point column-sum deviation after 20 iterations can reach 100%.

So when we set \(H^{res} = I\):

| Metric | Sinkhorn | Identity |
|---|---|---|
| Cumulative product | rank-1 collapse (\(\kappa = 10^{17}\)) | \(I\) (\(\kappa = 1\)) |
| Approximation error | row-sum std = 0.12 | exact |
| Extra computation | 20-step iteration + backward recompute | zero |
| Extra parameters | \(nC \times n^2\) projection weights | none |
| Signal propagation | shallow signals decay exponentially | lossless propagation |

<hr class="section-divider">

(Okay, some of the text above was AI-generated — like that table, heh — but the experimental data is from runs we actually conducted.)

I'd originally been tinkering with various mHC-lite variants, and nothing I tried could beat the original mHC. I was quite frustrated. Then one afternoon, my advisor Wang Yang hit me with the obvious suggestion: "Why not just try identity?" We ran the experiment and, well... one look and you just go quiet, haha.

Failed attempts include: trying mHC-lite with convex combinations for exact doubly stochastic matrices, softmax weighting — it all looked very reasonable on paper. But on Qwen3-1.7B, nothing I tried beat the original mHC. Later, looking at internal metrics, I observed that as \(\alpha_{res}\) increased (initialized at 0.01, gradually growing to around 2), the softmax temperature dropped and the output trended toward one-hot — less mixing?

I also tried orthogonalization (Cayley transform, Givens rotations) to guarantee spectral norm of exactly 1 — no explosion, no vanishing. Experiments showed \(\alpha_{res}\) barely moved, staying near 0.01. The fatal problem with orthogonal matrices is that they allow negative values — some streams can get negated, causing capacity collapse and some ugly internal metrics.

<hr class="section-divider">

## Analysis Plots

Finally, let me share some analysis plots that Opus drew — they're quite interesting.

For example, the effective channel size between any two layers in mHC, measuring how much of layer \(i\)'s transformation output can propagate to layer \(j\)'s transformation input:

\[
c_{ij} = \mathbf{h}_j^{\text{pre}} \cdot \left(\prod_{k=i+1}^{j-1} H_k^{\text{res}}\right) \cdot \mathbf{h}_i^{\text{post}\,T}
\]

![Channel strength vs layer distance: maintains 0.3–0.5 for gaps 1–15, visibly decays beyond gap 20, approximately linear decay on log scale](/assets/img/mhc-channel-strength-decay.jpg)

![Outgoing and incoming channel strength by HC module: middle layers (20–40) are core information hubs, final layers (44–55) barely emit but absorb heavily, shallow layers (0–10) are weak](/assets/img/mhc-outgoing-incoming-channels.jpg)

When \(H^{res} = I\), the channel formula simplifies to:

\[
c_{ij} = \mathbf{h}_j^{\text{pre}} \cdot I \cdot \mathbf{h}_i^{\text{post}\,T} = \mathbf{h}_j^{\text{pre}} \cdot \mathbf{h}_i^{\text{post}\,T}
\]

The direct communication strength between any two layers depends only on their respective \(h^{pre}\) and \(h^{post}\), unaffected by cumulative decay from intermediate layers' \(H^{res}\).

Writing in haste, there are surely oversights — particularly where I borrowed AI-generated text without proofreading word by word. But the experimental observations, conclusions, and general reasoning should be roughly correct.

In short, throw away the "m" in DeepSeek's mHC and replace it with plain old identity. Think of it this way: you don't even need to trouble infra to develop some Sinkhorn-Knopp TileLang kernel. Everybody wins!

<hr class="section-divider">

*Citation: [涮月亮的谪仙人](https://www.zhihu.com/people/shuan-yue-liang-de-zhe-xian-ren). (2026, February 27). 你的deepseek mHC可能不需要"m" [Your DeepSeek mHC Might Not Need the "m"]. Zhihu. [https://zhuanlan.zhihu.com/p/2010852389670908320](https://zhuanlan.zhihu.com/p/2010852389670908320)*

*Translated and shared for educational purposes. Original content copyright belongs to the author.*
