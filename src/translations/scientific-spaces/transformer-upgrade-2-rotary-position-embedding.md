---
title: "Transformer Upgrade Path: 2. Rotary Position Embedding, the Best of Both Worlds"
subtitle: "Translated from [Transformer升级之路：2、博采众长的旋转式位置编码](https://kexue.fm/archives/8265) by Jianlin Su (苏剑林)"
date: 2021-03-23T00:00:00+08:00
blurb: "Deriving Rotary Position Embedding (RoPE) from first principles: an absolute encoding that achieves relative position awareness through complex-number rotation, with long-range decay and compatibility with linear attention."
tags: ["translation", "rope", "position-encoding", "attention", "transformers"]
math: true
code: true
---

*Translator's note (Opus 4.6): This is an English translation of [Transformer升级之路：2、博采众长的旋转式位置编码](https://kexue.fm/archives/8265) by Jianlin Su (苏剑林), originally published on March 23, 2021 on [Scientific Spaces (科学空间)](https://kexue.fm). It is the second article in the "Transformer Upgrade Path" series. The translation preserves the author's first-person voice.*

<hr class="section-divider">

In the [previous article](/translations/scientific-spaces/transformer-position-encodings-that-rack-researchers-brains/), we gave a fairly detailed derivation and analysis of the original Sinusoidal position encoding. The overall impression is that Sinusoidal position encoding is "an absolute position encoding that wants to be a relative position encoding." Generally speaking, absolute position encodings are simple to implement and fast to compute, while relative position encodings directly capture relative position signals, align with our intuition, and often perform better in practice. It follows that if one could implement relative position encoding *through* an absolute position encoding, that would be "taking the best of all worlds" (博采众长) — "having one's cake and eating it too" (鱼与熊掌兼得). Sinusoidal position encoding vaguely achieves this, but not well enough.

This article introduces our self-developed **Rotary Transformer (RoFormer)** model, whose main modification is the application of **Rotary Position Embedding (RoPE)**, which I conceived. This is a design that, in conjunction with the Attention mechanism, achieves "relative position encoding via the mechanism of absolute position encoding." Precisely because of this design, it is also currently the only relative position encoding that can be used with linear Attention.

> **RoFormer: [https://github.com/ZhuiyiTechnology/roformer](https://github.com/ZhuiyiTechnology/roformer)**

## Basic Idea

In the earlier article [*Transformer Position Encodings That Rack Researchers' Brains*](/translations/scientific-spaces/transformer-position-encodings-that-rack-researchers-brains/), we briefly introduced RoPE, calling it the "fusion-style" approach at the time. This article provides a more detailed account of its origin and properties. In RoPE, our starting point is "implementing relative position encoding through absolute position encoding." This approach has both theoretical elegance and practical utility — for instance, its extensibility to linear Attention is primarily due to this property.

To achieve this goal, we assume the following operations add absolute position information to \(\boldsymbol{q}\) and \(\boldsymbol{k}\):

\[
\tilde{\boldsymbol{q}}_m = \boldsymbol{f}(\boldsymbol{q}, m), \quad \tilde{\boldsymbol{k}}_n = \boldsymbol{f}(\boldsymbol{k}, n) \tag{1}
\]

That is, we design operations \(\boldsymbol{f}(\cdot, m)\) and \(\boldsymbol{f}(\cdot, n)\) for \(\boldsymbol{q}\) and \(\boldsymbol{k}\) respectively, such that after applying them, \(\tilde{\boldsymbol{q}}_m\) and \(\tilde{\boldsymbol{k}}_n\) carry absolute position information for positions \(m\) and \(n\). Since the core operation of Attention is the inner product, we want the inner product result to carry relative position information. We therefore posit the following identity:

\[
\langle \boldsymbol{f}(\boldsymbol{q}, m),\, \boldsymbol{f}(\boldsymbol{k}, n) \rangle = g(\boldsymbol{q}, \boldsymbol{k}, m - n) \tag{2}
\]

So we need to find a (preferably simple) solution to this identity. The solution process also requires some initial conditions; we can reasonably set \(\boldsymbol{f}(\boldsymbol{q}, 0) = \boldsymbol{q}\) and \(\boldsymbol{f}(\boldsymbol{k}, 0) = \boldsymbol{k}\).

## Derivation

Following the same approach as the previous article, we first consider the two-dimensional case and use complex numbers to solve it. In complex numbers, \(\langle \boldsymbol{q}, \boldsymbol{k} \rangle = \text{Re}[\boldsymbol{q}\boldsymbol{k}^*]\), where \(\text{Re}[\cdot]\) denotes the real part, so we have

\[
\text{Re}[\boldsymbol{f}(\boldsymbol{q}, m)\,\boldsymbol{f}^*(\boldsymbol{k}, n)] = g(\boldsymbol{q}, \boldsymbol{k}, m - n) \tag{3}
\]

For simplicity, we assume there exists a complex number \(\boldsymbol{g}(\boldsymbol{q}, \boldsymbol{k}, m - n)\) such that \(\boldsymbol{f}(\boldsymbol{q}, m)\,\boldsymbol{f}^*(\boldsymbol{k}, n) = \boldsymbol{g}(\boldsymbol{q}, \boldsymbol{k}, m - n)\). Then we write these in exponential form:

\[
\begin{aligned}
\boldsymbol{f}(\boldsymbol{q}, m) &= R_f(\boldsymbol{q}, m)\,e^{\text{i}\,\Theta_f(\boldsymbol{q}, m)} \\
\boldsymbol{f}(\boldsymbol{k}, n) &= R_f(\boldsymbol{k}, n)\,e^{\text{i}\,\Theta_f(\boldsymbol{k}, n)} \\
\boldsymbol{g}(\boldsymbol{q}, \boldsymbol{k}, m - n) &= R_g(\boldsymbol{q}, \boldsymbol{k}, m - n)\,e^{\text{i}\,\Theta_g(\boldsymbol{q}, \boldsymbol{k}, m - n)}
\end{aligned} \tag{4}
\]

Substituting into the equation yields the system:

\[
\begin{aligned}
R_f(\boldsymbol{q}, m)\, R_f(\boldsymbol{k}, n) &= R_g(\boldsymbol{q}, \boldsymbol{k}, m - n) \\
\Theta_f(\boldsymbol{q}, m) - \Theta_f(\boldsymbol{k}, n) &= \Theta_g(\boldsymbol{q}, \boldsymbol{k}, m - n)
\end{aligned} \tag{5}
\]

For the first equation, substituting \(m = n\) gives

\[
R_f(\boldsymbol{q}, m)\, R_f(\boldsymbol{k}, m) = R_g(\boldsymbol{q}, \boldsymbol{k}, 0) = R_f(\boldsymbol{q}, 0)\, R_f(\boldsymbol{k}, 0) = \|\boldsymbol{q}\|\,\|\boldsymbol{k}\| \tag{6}
\]

The last equality follows from the initial conditions \(\boldsymbol{f}(\boldsymbol{q}, 0) = \boldsymbol{q}\) and \(\boldsymbol{f}(\boldsymbol{k}, 0) = \boldsymbol{k}\). So we can simply set \(R_f(\boldsymbol{q}, m) = \|\boldsymbol{q}\|\) and \(R_f(\boldsymbol{k}, m) = \|\boldsymbol{k}\|\) — that is, the modulus does not depend on \(m\). For the second equation, again substituting \(m = n\) yields

\[
\Theta_f(\boldsymbol{q}, m) - \Theta_f(\boldsymbol{k}, m) = \Theta_g(\boldsymbol{q}, \boldsymbol{k}, 0) = \Theta_f(\boldsymbol{q}, 0) - \Theta_f(\boldsymbol{k}, 0) = \Theta(\boldsymbol{q}) - \Theta(\boldsymbol{k}) \tag{7}
\]

Here \(\Theta(\boldsymbol{q})\) and \(\Theta(\boldsymbol{k})\) are the arguments (angles) of \(\boldsymbol{q}\) and \(\boldsymbol{k}\) themselves, and the last equality again follows from the initial conditions. From the above we get \(\Theta_f(\boldsymbol{q}, m) - \Theta(\boldsymbol{q}) = \Theta_f(\boldsymbol{k}, m) - \Theta(\boldsymbol{k})\), so \(\Theta_f(\boldsymbol{q}, m) - \Theta(\boldsymbol{q})\) must be a function of \(m\) alone, independent of \(\boldsymbol{q}\). Denote it \(\varphi(m)\), i.e., \(\Theta_f(\boldsymbol{q}, m) = \Theta(\boldsymbol{q}) + \varphi(m)\). Next, substituting \(n = m - 1\) and rearranging gives

\[
\varphi(m) - \varphi(m - 1) = \Theta_g(\boldsymbol{q}, \boldsymbol{k}, 1) + \Theta(\boldsymbol{k}) - \Theta(\boldsymbol{q}) \tag{8}
\]

That is, \(\{\varphi(m)\}\) is an arithmetic sequence. Letting the right-hand side equal \(\theta\), we solve \(\varphi(m) = m\theta\).

## Encoding Form

In summary, we obtain the complex-number representation of RoPE in the two-dimensional case:

\[
\boldsymbol{f}(\boldsymbol{q}, m) = R_f(\boldsymbol{q}, m)\,e^{\text{i}\,\Theta_f(\boldsymbol{q}, m)} = \|\boldsymbol{q}\|\, e^{\text{i}(\Theta(\boldsymbol{q}) + m\theta)} = \boldsymbol{q}\, e^{\text{i}\,m\theta} \tag{9}
\]

By the geometric meaning of complex multiplication, this transformation corresponds to rotating the vector, which is why we call it "Rotary Position Embedding." It can also be written in matrix form:

\[
\boldsymbol{f}(\boldsymbol{q}, m) = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} q_0 \\ q_1 \end{pmatrix} \tag{10}
\]

Since the inner product satisfies linear additivity, RoPE for any even dimension can be expressed as a concatenation of two-dimensional cases:

\[
\underbrace{\begin{pmatrix}
\cos m\theta_0 & -\sin m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\
\sin m\theta_0 & \cos m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos m\theta_1 & -\sin m\theta_1 & \cdots & 0 & 0 \\
0 & 0 & \sin m\theta_1 & \cos m\theta_1 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2-1} & -\sin m\theta_{d/2-1} \\
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2-1} & \cos m\theta_{d/2-1}
\end{pmatrix}}_{\boldsymbol{\mathcal{R}}_m}
\begin{pmatrix} q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \\ q_{d-2} \\ q_{d-1} \end{pmatrix} \tag{11}
\]

In other words, multiply the vector \(\boldsymbol{q}\) at position \(m\) by the matrix \(\boldsymbol{\mathcal{R}}_m\), and the vector \(\boldsymbol{k}\) at position \(n\) by the matrix \(\boldsymbol{\mathcal{R}}_n\). Then perform Attention using the transformed \(\boldsymbol{Q}\) and \(\boldsymbol{K}\) sequences. The Attention automatically incorporates relative position information, because the following identity holds:

\[
(\boldsymbol{\mathcal{R}}_m \boldsymbol{q})^{\top}(\boldsymbol{\mathcal{R}}_n \boldsymbol{k}) = \boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_m^{\top} \boldsymbol{\mathcal{R}}_n \boldsymbol{k} = \boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_{n-m} \boldsymbol{k} \tag{12}
\]

It is worth noting that \(\boldsymbol{\mathcal{R}}_m\) is an orthogonal matrix — it does not change the magnitude of vectors, so it generally does not affect the stability of the original model.

Because \(\boldsymbol{\mathcal{R}}_m\) is sparse, implementing it directly via matrix multiplication wastes computation. The recommended implementation of RoPE is:

\[
\begin{pmatrix} q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \\ q_{d-2} \\ q_{d-1} \end{pmatrix} \otimes \begin{pmatrix} \cos m\theta_0 \\ \cos m\theta_0 \\ \cos m\theta_1 \\ \cos m\theta_1 \\ \vdots \\ \cos m\theta_{d/2-1} \\ \cos m\theta_{d/2-1} \end{pmatrix} + \begin{pmatrix} -q_1 \\ q_0 \\ -q_3 \\ q_2 \\ \vdots \\ -q_{d-1} \\ q_{d-2} \end{pmatrix} \otimes \begin{pmatrix} \sin m\theta_0 \\ \sin m\theta_0 \\ \sin m\theta_1 \\ \sin m\theta_1 \\ \vdots \\ \sin m\theta_{d/2-1} \\ \sin m\theta_{d/2-1} \end{pmatrix} \tag{13}
\]

where \(\otimes\) denotes element-wise multiplication, i.e., the `*` operation in NumPy, TensorFlow, and similar frameworks. From this implementation, we can also see that RoPE can be viewed as a variant of multiplicative position encoding.

## Long-Range Decay

We can see that RoPE is formally somewhat similar to Sinusoidal position encoding, except that Sinusoidal position encoding is additive while RoPE can be viewed as multiplicative. For the choice of \(\theta_i\), we follow the same scheme as Sinusoidal position encoding: \(\theta_i = 10000^{-2i/d}\), which provides a degree of long-range decay.

The specific proof is as follows. After pairing the dimensions of \(\boldsymbol{q}\) and \(\boldsymbol{k}\) in groups of two, the inner product after applying RoPE can be expressed via complex multiplication as:

\[
(\boldsymbol{\mathcal{R}}_m \boldsymbol{q})^{\top}(\boldsymbol{\mathcal{R}}_n \boldsymbol{k}) = \text{Re}\left[\sum_{i=0}^{d/2-1} \boldsymbol{q}_{[2i:2i+1]}\,\boldsymbol{k}_{[2i:2i+1]}^*\, e^{\text{i}(m-n)\theta_i}\right] \tag{14}
\]

Let \(h_i = \boldsymbol{q}_{[2i:2i+1]}\,\boldsymbol{k}_{[2i:2i+1]}^*\) and \(S_j = \sum_{i=0}^{j-1} e^{\text{i}(m-n)\theta_i}\), with the convention that \(h_{d/2} = 0\) and \(S_0 = 0\). Then by the [Abel transformation (summation by parts)](https://en.wikipedia.org/wiki/Abel%27s_summation_formula), we obtain:

\[
\sum_{i=0}^{d/2-1} \boldsymbol{q}_{[2i:2i+1]}\,\boldsymbol{k}_{[2i:2i+1]}^*\, e^{\text{i}(m-n)\theta_i} = \sum_{i=0}^{d/2-1} h_i(S_{i+1} - S_i) = -\sum_{i=0}^{d/2-1} S_{i+1}(h_{i+1} - h_i) \tag{15}
\]

Therefore:

\[
\begin{aligned}
\left|\sum_{i=0}^{d/2-1} \boldsymbol{q}_{[2i:2i+1]}\,\boldsymbol{k}_{[2i:2i+1]}^*\, e^{\text{i}(m-n)\theta_i}\right| &= \left|\sum_{i=0}^{d/2-1} S_{i+1}(h_{i+1} - h_i)\right| \\
&\leq \sum_{i=0}^{d/2-1} |S_{i+1}|\,|h_{i+1} - h_i| \\
&\leq \left(\max_i |h_{i+1} - h_i|\right) \sum_{i=0}^{d/2-1} |S_{i+1}|
\end{aligned} \tag{16}
\]

We can therefore examine how \(\frac{1}{d/2}\sum_{i=1}^{d/2} |S_i|\) varies with relative distance as a measure of the decay. The Mathematica code is:

```mathematica
d = 128;
\[Theta][t_] = 10000^(-2*t/d);
f[m_] = Sum[
    Norm[Sum[Exp[I*m*\[Theta][i]], {i, 0, j}]], {j, 0, d/2 - 1}]/(d/2);
Plot[f[m], {m, 0, 256}, AxesLabel -> {Relative Distance, Relative Magnitude}]
```

[^decay-plot]: Long-range decay of RoPE with \\(\theta_i = 10000^{-2i/d}\\) and \\(d = 128\\). ![RoPE long-range decay plot](/assets/img/rope-long-range-decay.png)

From the resulting plot[^decay-plot], we can see that as the relative distance increases, the inner product result exhibits a decaying trend. Therefore, the choice \(\theta_i = 10000^{-2i/d}\) does indeed provide a degree of long-range decay. Of course, as mentioned in the previous article, this is not the only choice that yields long-range decay — almost any smooth monotonic function will do. We simply follow the existing convention. I also tried initializing with \(\theta_i = 10000^{-2i/d}\) and treating \(\theta_i\) as trainable parameters, but after training for a while found that \(\theta_i\) did not update significantly, so I simply fixed \(\theta_i = 10000^{-2i/d}\).

## Linear Attention Setting

Finally, we point out that RoPE is currently the only relative position encoding that can be used with linear Attention. This is because other relative position encodings operate directly on the Attention matrix, but linear Attention does not compute the Attention matrix explicitly, so there is no way to manipulate it. Consequently, other approaches cannot be applied to linear Attention. RoPE, on the other hand, implements relative position encoding through the mechanism of absolute position encoding and does not need to operate on the Attention matrix, thus making it possible to apply to linear Attention.

For an introduction to linear Attention, we refer interested readers to [*Exploring Linear Attention: Does Attention Need a Softmax?*](https://kexue.fm/archives/7546). The common form of linear Attention is:

\[
\text{Attention}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V})_i = \frac{\sum_{j=1}^{n} \text{sim}(\boldsymbol{q}_i, \boldsymbol{k}_j)\,\boldsymbol{v}_j}{\sum_{j=1}^{n} \text{sim}(\boldsymbol{q}_i, \boldsymbol{k}_j)} = \frac{\sum_{j=1}^{n} \phi(\boldsymbol{q}_i)^{\top} \varphi(\boldsymbol{k}_j)\,\boldsymbol{v}_j}{\sum_{j=1}^{n} \phi(\boldsymbol{q}_i)^{\top} \varphi(\boldsymbol{k}_j)} \tag{17}
\]

where \(\phi\) and \(\varphi\) are activation functions with non-negative range. As we can see, linear Attention is also based on inner products, so a natural idea is to insert RoPE into the inner product:

\[
\frac{\sum_{j=1}^{n} [\boldsymbol{\mathcal{R}}_i \phi(\boldsymbol{q}_i)]^{\top} [\boldsymbol{\mathcal{R}}_j \varphi(\boldsymbol{k}_j)]\,\boldsymbol{v}_j}{\sum_{j=1}^{n} [\boldsymbol{\mathcal{R}}_i \phi(\boldsymbol{q}_i)]^{\top} [\boldsymbol{\mathcal{R}}_j \varphi(\boldsymbol{k}_j)]} \tag{18}
\]

However, the problem is that the inner product \([\boldsymbol{\mathcal{R}}_i \phi(\boldsymbol{q}_i)]^{\top} [\boldsymbol{\mathcal{R}}_j \varphi(\boldsymbol{k}_j)]\) may be negative, so it is no longer standard probabilistic attention, and the denominator risks being zero, which could cause optimization instability. Since \(\boldsymbol{\mathcal{R}}_i\) and \(\boldsymbol{\mathcal{R}}_j\) are both orthogonal matrices that do not change vector magnitudes, we can abandon the standard probabilistic normalization requirement and use the following as a new form of linear Attention:

\[
\frac{\sum_{j=1}^{n} [\boldsymbol{\mathcal{R}}_i \phi(\boldsymbol{q}_i)]^{\top} [\boldsymbol{\mathcal{R}}_j \varphi(\boldsymbol{k}_j)]\,\boldsymbol{v}_j}{\sum_{j=1}^{n} \phi(\boldsymbol{q}_i)^{\top} \varphi(\boldsymbol{k}_j)} \tag{19}
\]

That is, RoPE is inserted only in the numerator, while the denominator remains unchanged. This type of attention is no longer probability-based (the attention matrix no longer satisfies non-negative normalization), but in some sense it is still a normalization scheme, and there is no evidence that non-probabilistic attention is necessarily worse (for example, [*Nystromformer*](https://kexue.fm/archives/8180) also does not strictly construct attention according to probability distributions). So we include it as a candidate for experiments, and our preliminary results show that this form of linear Attention is indeed effective.

Additionally, in [*Exploring Linear Attention: Does Attention Need a Softmax?*](https://kexue.fm/archives/7546), I also proposed another linear Attention scheme: \(\text{sim}(\boldsymbol{q}_i, \boldsymbol{k}_j) = 1 + \left(\frac{\boldsymbol{q}_i}{\|\boldsymbol{q}_i\|}\right)^{\top}\left(\frac{\boldsymbol{k}_j}{\|\boldsymbol{k}_j\|}\right)\). It does not depend on the non-negativity of the range, and since RoPE does not change magnitudes, RoPE can be directly applied to this type of linear Attention without altering its probabilistic interpretation.

## Open-Source Model

The first version of the RoFormer model has been trained and open-sourced on Github:

> **RoFormer: [https://github.com/ZhuiyiTechnology/roformer](https://github.com/ZhuiyiTechnology/roformer)**

In brief, RoFormer is a [WoBERT](https://github.com/ZhuiyiTechnology/WoBERT) model with its absolute position encoding replaced by RoPE. Its structural comparison with other models is as follows:

|  | BERT | WoBERT | NEZHA | RoFormer |
|<hr class="section-divider">|<hr class="section-divider">|<hr class="section-divider">|<hr class="section-divider">|<hr class="section-divider">|
| Token unit | Character | Word | Character | Word |
| Position encoding | Absolute | Absolute | Classical relative | RoPE |

For pre-training, we used WoBERT Plus as the base and adopted an alternating training strategy with multiple sequence lengths and batch sizes, allowing the model to adapt to different training scenarios in advance:

| Stage | maxlen | batch size | Training steps | Final loss | Final acc |
|<hr class="section-divider">|<hr class="section-divider">|<hr class="section-divider">|<hr class="section-divider">|<hr class="section-divider">|<hr class="section-divider">|
| 1 | 512 | 256 | 200k | 1.73 | 65.0% |
| 2 | 1536 | 256 | 12.5k | 1.61 | 66.8% |
| 3 | 256 | 256 | 120k | 1.75 | 64.6% |
| 4 | 128 | 512 | 80k | 1.83 | 63.4% |
| 5 | 1536 | 256 | 10k | 1.58 | 67.4% |
| 6 | 512 | 512 | 30k | 1.66 | 66.2% |

From the table, we can also see that increasing the sequence length actually improves pre-training accuracy, which indirectly demonstrates RoFormer's effectiveness at handling long-text semantics and reflects the good extrapolation capability of RoPE. On short-text tasks, RoFormer performs similarly to WoBERT; the main strength of RoFormer is its ability to directly handle text of arbitrary length. Below are our experimental results on the [CAIL2019-SCM](https://arxiv.org/abs/1911.08962) task:

|  | Validation set | Test set |
|<hr class="section-divider">|<hr class="section-divider">|<hr class="section-divider">|
| BERT-512 | 64.13% | 67.77% |
| WoBERT-512 | 64.07% | 68.10% |
| RoFormer-512 | 64.13% | 68.29% |
| RoFormer-1024 | **66.07%** | **69.79%** |

The number after the hyphen is the maxlen used during fine-tuning. We can see that RoFormer does handle long-text semantics well. As for hardware requirements, on a 24GB GPU, running maxlen=1024, the batch_size can reach 8 or above. Among Chinese tasks, this is the only one I found that is suitable as a test of long-text capability, so long-text evaluation was limited to this task. Readers are welcome to run tests or recommend other evaluation tasks.

Of course, although RoFormer can theoretically handle sequences of arbitrary length, the current version still has quadratic complexity. We are also training a linear Attention-based RoFormer model, which will be open-sourced once the experiments are complete.

(Note: RoPE and RoFormer have been compiled into the paper [*RoFormer: Enhanced Transformer with Rotary Position Embedding*](https://arxiv.org/abs/2104.09864) and submitted to arXiv. Feel free to use and cite it!)

## Summary

This article introduced our self-developed Rotary Position Embedding (RoPE) and the corresponding pre-trained model RoFormer. From a theoretical perspective, RoPE shares some commonalities with Sinusoidal position encoding, but RoPE does not rely on Taylor expansion and is more rigorous and interpretable. From the results of the pre-trained RoFormer model, RoPE exhibits good extrapolation properties, and when applied to Transformers, it demonstrates strong long-text processing capability. Furthermore, RoPE is currently the only relative position encoding that can be used with linear Attention.

<hr class="section-divider">

*Citation: Su, J. (2021, March 23). Transformer升级之路：2、博采众长的旋转式位置编码 [Transformer Upgrade Path: 2. Rotary Position Embedding, the Best of Both Worlds]. Scientific Spaces. [https://kexue.fm/archives/8265](https://kexue.fm/archives/8265)*
