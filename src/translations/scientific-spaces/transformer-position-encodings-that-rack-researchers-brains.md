---
title: "Transformer Position Encodings That Rack Researchers' Brains"
subtitle: "Translated from [让研究人员绞尽脑汁的Transformer位置编码](https://kexue.fm/archives/8130) by Jianlin Su (苏剑林)"
date: 2021-02-03T00:00:00+08:00
blurb: "A survey of position encoding schemes for Transformers -- trainable, sinusoidal, recurrent, multiplicative, relative (classic, XLNet, T5, DeBERTa), CNN-based, complex-valued, and a fusion approach that previews RoPE."
tags: ["translation", "position-encoding", "attention", "transformers", "rope"]
math: true
---

*Translator's note (Opus 4.6): This is an English translation of [让研究人员绞尽脑汁的Transformer位置编码](https://kexue.fm/archives/8130) by Jianlin Su (苏剑林), originally published on February 3, 2021 on [Scientific Spaces (科学空间)](https://kexue.fm). This article is a precursor to Su's later work on Rotary Position Embedding (RoPE); the "fusion-style" encoding sketched in the final section of this post was formalized in the follow-up article [Transformer Upgrade Path: 2. Rotary Position Embedding](/translations/scientific-spaces/transformer-upgrade-2-rotary-position-embedding/). The translation preserves the author's first-person voice.*

---

Unlike RNN, CNN, and other models, for the Transformer model, adding position encoding is indispensable, because a pure Attention module cannot capture the order of its inputs -- it cannot distinguish Tokens at different positions. For this, we broadly have two choices: (1) find a way to incorporate position information into the input, which constitutes the general approach of **absolute position encoding**; (2) find a way to tweak the Attention structure so that it can distinguish Tokens at different positions, which constitutes the general approach of **relative position encoding**.

Although the main categories are just absolute and relative position encoding, each category can spawn all sorts of variants, and researchers have racked their brains (绞尽脑汁) over this. There are also some position encodings that don't play by the rules. In this article, let us appreciate the encoding schemes that researchers have constructed to better express position information -- a veritable showcase of "eight immortals crossing the sea, each displaying their own magic" (八仙过海，各显神通).

## Absolute Position Encoding

Formally, absolute position encoding is a relatively simple approach, but even so, it doesn't stop researchers from all directions from having their creative ideas, and there are quite a few variants. Generally, absolute position encoding is added to the input: to the \(k\)-th input vector \(\boldsymbol{x}_k\), a position vector \(\boldsymbol{p}_k\) is added to produce \(\boldsymbol{x}_k + \boldsymbol{p}_k\), where \(\boldsymbol{p}_k\) depends only on the position index \(k\).

### Trainable

Clearly, the most straightforward approach to absolute position encoding is not to design anything special, but to simply **treat the position encoding as a trainable parameter**. For example, with a maximum length of 512 and encoding dimension of 768, one initializes a \(512 \times 768\) matrix as the position vectors and lets them update during training. This is what BERT, GPT, and similar models use today. In fact, it can be traced back even further -- Facebook's 2017 paper [*Convolutional Sequence to Sequence Learning*](https://arxiv.org/abs/1705.03122) already used it.

For this trainable absolute position encoding, the common view is that its drawback is the lack of extrapolation ability: if the maximum pre-training length is 512, then it can only handle sentences of length 512 at most -- anything longer cannot be processed. Of course, one could randomly initialize position vectors beyond 512 and continue fine-tuning. However, my recent research shows that through hierarchical decomposition, absolute position encoding can extrapolate to sufficiently long ranges while maintaining decent performance. For details, see my earlier blog post [*Hierarchically Decomposed Position Encoding, Letting BERT Handle Ultra-Long Text*](https://kexue.fm/archives/7947). So in fact, lack of extrapolation is not really a clear drawback of absolute position encoding.

### Sinusoidal

Trigonometric position encoding, also commonly called **Sinusoidal position encoding**, is an explicit formula proposed in Google's paper [*Attention is All You Need*](https://arxiv.org/abs/1706.03762):

\[
\begin{cases}
\boldsymbol{p}_{k,2i} = \sin\Big(k / 10000^{2i/d}\Big) \\[5pt]
\boldsymbol{p}_{k,2i+1} = \cos\Big(k / 10000^{2i/d}\Big)
\end{cases} \tag{1}
\]

where \(\boldsymbol{p}_{k,2i}\) and \(\boldsymbol{p}_{k,2i+1}\) are the \(2i\)-th and \((2i+1)\)-th components of the encoding vector for position \(k\), and \(d\) is the dimension of the position vector.

Clearly, the hallmark of sinusoidal position encoding is its explicit generation rule, so one can hope it has some degree of extrapolation ability. Another reason to use it is the trigonometric identities \(\sin(\alpha + \beta) = \sin\alpha\cos\beta + \cos\alpha\sin\beta\) and \(\cos(\alpha + \beta) = \cos\alpha\cos\beta - \sin\alpha\sin\beta\), which show that the vector at position \(\alpha + \beta\) can be expressed as a combination of vectors at positions \(\alpha\) and \(\beta\), providing the possibility of expressing relative position information. Strangely though, we rarely see work that directly uses this form of absolute position encoding anymore, and the reason is unclear.

### Recurrent

In principle, RNN models don't need position encoding -- they inherently have the possibility of learning position information through their structure (because **recursion means we can train a "counting" model**). Therefore, if one prepends an RNN layer before the Transformer, then in theory no position encoding is needed. Similarly, we can use an RNN model to learn a form of absolute position encoding: starting from a vector \(\boldsymbol{p}_0\), use the recurrence \(\boldsymbol{p}_{k+1} = f(\boldsymbol{p}_k)\) to obtain encoding vectors for each position.

The ICML 2020 paper [*Learning to Encode Position for Transformer with Continuous Dynamical Model*](https://arxiv.org/abs/2003.09229) pushes this idea to its extreme: it proposes modeling position encoding via an ordinary differential equation (ODE) \(d\boldsymbol{p}_t / dt = \boldsymbol{h}(\boldsymbol{p}_t, t)\). This approach is called FLOATER. Clearly, FLOATER also belongs to the recurrent category. The function \(\boldsymbol{h}(\boldsymbol{p}_t, t)\) can be modeled by a neural network, making this a neural ODE, an area of increasing research activity.

Theoretically, recurrence-based position encodings also have good extrapolation properties, and they offer more flexibility than sinusoidal position encoding (for example, it is easy to show that sinusoidal position encoding is a particular solution of FLOATER). But clearly, recurrent position encoding sacrifices some parallelism and may introduce a speed bottleneck.

### Multiplicative

Earlier we noted that the typical way to combine the input \(\boldsymbol{x}_k\) with absolute position encoding \(\boldsymbol{p}_k\) is \(\boldsymbol{x}_k + \boldsymbol{p}_k\). But are there "atypical" combinations? For instance, \(\boldsymbol{x}_k \otimes \boldsymbol{p}_k\) (element-wise multiplication)? When building models, we have multiple ways to fuse two vectors -- addition, multiplication, and even concatenation are all worth considering. So why does everyone default to addition when doing absolute position encoding?

I'm afraid I don't know the answer either. Perhaps the default choice of addition is because vector addition has a clear geometric interpretation, but for deep learning models, this geometric interpretation doesn't really have much practical value. A recent experiment I came across suggests that switching from "add" to "multiply" -- that is, using \(\boldsymbol{x}_k \otimes \boldsymbol{p}_k\) -- may achieve better results than \(\boldsymbol{x}_k + \boldsymbol{p}_k\). I haven't done a thorough comparison myself; I'm just presenting this as a possibility. For the experimental source, see [*Chinese Language Model Research: (1) Multiplicative Position Encoding*](https://zhuanlan.zhihu.com/p/183234823).

## Relative Position Encoding

Relative position encoding doesn't fully model the position information of each input. Instead, it considers the relative distance between the current position and the attended position when computing Attention. Since natural language generally depends more on relative position, relative position encoding usually also performs well. Relative position encoding offers greater flexibility and further showcases researchers' "unbridled imagination" (天马行空).

### Classic

Relative position encoding originates from Google's paper [*Self-Attention with Relative Position Representations*](https://arxiv.org/abs/1803.02155). Huawei's open-source NEZHA model also uses this type of position encoding, and subsequent relative position encoding variants are basically simple modifications following the same pattern.

It is generally understood that relative position encoding was inspired by absolute position encoding. Consider the standard Attention with absolute position encoding:

\[
\begin{aligned}
\boldsymbol{q}_i &= (\boldsymbol{x}_i + \boldsymbol{p}_i)\boldsymbol{W}_Q \\
\boldsymbol{k}_j &= (\boldsymbol{x}_j + \boldsymbol{p}_j)\boldsymbol{W}_K \\
\boldsymbol{v}_j &= (\boldsymbol{x}_j + \boldsymbol{p}_j)\boldsymbol{W}_V \\
a_{i,j} &= \text{softmax}\left(\boldsymbol{q}_i \boldsymbol{k}_j^{\top}\right) \\
\boldsymbol{o}_i &= \sum_j a_{i,j} \boldsymbol{v}_j
\end{aligned} \tag{2}
\]

where \(\text{softmax}\) normalizes over the \(j\) dimension, and all vectors here are row vectors. We begin by expanding \(\boldsymbol{q}_i \boldsymbol{k}_j^{\top}\):

\[
\boldsymbol{q}_i \boldsymbol{k}_j^{\top} = (\boldsymbol{x}_i + \boldsymbol{p}_i)\boldsymbol{W}_Q \boldsymbol{W}_K^{\top}(\boldsymbol{x}_j + \boldsymbol{p}_j)^{\top} = (\boldsymbol{x}_i \boldsymbol{W}_Q + \boldsymbol{p}_i \boldsymbol{W}_Q)(\boldsymbol{W}_K^{\top}\boldsymbol{x}_j^{\top} + \boldsymbol{W}_K^{\top}\boldsymbol{p}_j^{\top}) \tag{3}
\]

To introduce relative position information, Google removed the first position term and changed the second term \(\boldsymbol{p}_j \boldsymbol{W}_K\) to a binary position vector \(\boldsymbol{R}_{i,j}^{K}\), yielding:

\[
a_{i,j} = \text{softmax}\left(\boldsymbol{x}_i \boldsymbol{W}_Q (\boldsymbol{x}_j \boldsymbol{W}_K + \boldsymbol{R}_{i,j}^{K})^{\top}\right) \tag{4}
\]

And in \(\boldsymbol{o}_i = \sum_j a_{i,j} \boldsymbol{v}_j = \sum_j a_{i,j}(\boldsymbol{x}_j \boldsymbol{W}_V + \boldsymbol{p}_j \boldsymbol{W}_V)\), they replaced \(\boldsymbol{p}_j \boldsymbol{W}_V\) with \(\boldsymbol{R}_{i,j}^{V}\):

\[
\boldsymbol{o}_i = \sum_j a_{i,j} \left(\boldsymbol{x}_j \boldsymbol{W}_V + \boldsymbol{R}_{i,j}^{V}\right) \tag{5}
\]

The "relative" part means changing the vectors \(\boldsymbol{R}_{i,j}^{K}\) and \(\boldsymbol{R}_{i,j}^{V}\), which originally depended on the pair of coordinates \((i, j)\), to depend only on the relative distance \(i - j\), and typically applying clipping to handle arbitrary distances:

\[
\begin{aligned}
\boldsymbol{R}_{i,j}^{K} &= \boldsymbol{p}_K[\text{clip}(i - j,\, p_{\min},\, p_{\max})] \\
\boldsymbol{R}_{i,j}^{V} &= \boldsymbol{p}_V[\text{clip}(i - j,\, p_{\min},\, p_{\max})]
\end{aligned} \tag{6}
\]

This way, **only a finite number of position encodings are needed to express arbitrary-length relative positions** (because of the clipping). Whether \(\boldsymbol{p}_K\) and \(\boldsymbol{p}_V\) use the trainable or sinusoidal approach, they can handle text of arbitrary length.

### XLNet-Style

The XLNet-style position encoding actually originates from the Transformer-XL paper [*Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context*](https://arxiv.org/abs/1901.02860). However, it was only after the [XLNet](https://arxiv.org/abs/1906.08237) model -- which adopted the Transformer-XL architecture -- surpassed BERT to some extent that Transformer-XL became widely known. Hence this position encoding is usually attributed to XLNet.

The XLNet-style position encoding stems from the full expansion of \(\boldsymbol{q}_i \boldsymbol{k}_j^{\top}\):

\[
\boldsymbol{q}_i \boldsymbol{k}_j^{\top} = \boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top} \boldsymbol{x}_j^{\top} + \boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top} \boldsymbol{p}_j^{\top} + \boldsymbol{p}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top} \boldsymbol{x}_j^{\top} + \boldsymbol{p}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top} \boldsymbol{p}_j^{\top} \tag{7}
\]

Transformer-XL's approach is straightforward: directly replace \(\boldsymbol{p}_j\) with a relative position vector \(\boldsymbol{R}_{i-j}\), and replace the two occurrences of \(\boldsymbol{p}_i\) with two trainable vectors \(\boldsymbol{u}\) and \(\boldsymbol{v}\):

\[
\boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top} \boldsymbol{x}_j^{\top} + \boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top} \boldsymbol{R}_{i-j}^{\top} + \boldsymbol{u}\, \boldsymbol{W}_Q \boldsymbol{W}_K^{\top} \boldsymbol{x}_j^{\top} + \boldsymbol{v}\, \boldsymbol{W}_Q \boldsymbol{W}_K^{\top} \boldsymbol{R}_{i-j}^{\top} \tag{8}
\]

The \(\boldsymbol{R}_{i-j}\) in this encoding is not clipped as in Eq. (6), but instead uses the Sinusoidal generation scheme directly. Since the encoding space of \(\boldsymbol{R}_{i-j}\) may not match that of \(\boldsymbol{x}_j\), the \(\boldsymbol{W}_K^{\top}\) preceding \(\boldsymbol{R}_{i-j}\) is replaced with a separate independent matrix \(\boldsymbol{W}_{K,R}^{\top}\). Additionally, \(\boldsymbol{u}\, \boldsymbol{W}_Q\) and \(\boldsymbol{v}\, \boldsymbol{W}_Q\) can be merged into single vectors \(\boldsymbol{u}\) and \(\boldsymbol{v}\), so the final formula is:

\[
\boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top} \boldsymbol{x}_j^{\top} + \boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_{K,R}^{\top} \boldsymbol{R}_{i-j}^{\top} + \boldsymbol{u}\, \boldsymbol{W}_K^{\top} \boldsymbol{x}_j^{\top} + \boldsymbol{v}\, \boldsymbol{W}_{K,R}^{\top} \boldsymbol{R}_{i-j}^{\top} \tag{9}
\]

Furthermore, the position bias on \(\boldsymbol{v}_j\) was simply dropped, i.e., \(\boldsymbol{o}_i = \sum_j a_{i,j} \boldsymbol{x}_j \boldsymbol{W}_V\). **It seems that starting from this work, subsequent relative position encodings have only been added to the Attention matrix, and no longer to \(\boldsymbol{v}_j\).**

### T5-Style

The T5 model comes from the paper [*Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*](https://arxiv.org/abs/1910.10683), which uses an even simpler relative position encoding. The reasoning again starts from the expansion in Eq. (7). If we insist on analyzing the meaning of each term, they can be understood as the combination of four types of attention: **"input-input"**, **"input-position"**, **"position-input"**, and **"position-position"**. If we believe that input information and position information should be independent (decoupled), then they shouldn't interact too much. So the "input-position" and "position-input" attention terms can be dropped. And \(\boldsymbol{p}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top} \boldsymbol{p}_j^{\top}\) is actually just a scalar that depends only on \((i, j)\), which we can directly train as a parameter. This simplifies to:

\[
\boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top} \boldsymbol{x}_j^{\top} + \boldsymbol{\beta}_{i,j} \tag{10}
\]

Put plainly, it simply **adds a trainable bias term** to the Attention matrix -- and like the XLNet-style, the position bias on \(\boldsymbol{v}_j\) is directly dropped. Microsoft's paper [*Rethinking Positional Encoding in Language Pre-training*](https://arxiv.org/abs/2006.15595) at ICLR 2021 contains the same idea with the TUPE position encoding.

What's rather distinctive is that, unlike the standard approach of treating \(\boldsymbol{\beta}_{i,j}\) as a function of \(i - j\) with clipping, T5 applies a **"bucketing"** operation to relative positions. That is, relative position \(i - j\) actually maps to position \(f(i - j)\), with the mapping as follows:

| \(i - j\) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| \(f(i-j)\) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 8 | 8 | 8 | 9 | 9 | 9 | 9 |

| \(i - j\) | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | ... |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| \(f(i-j)\) | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 11 | 11 | 11 | 11 | 11 | 11 | 11 | 11 | ... |

For the specific mapping code, readers can consult the source code. The design intuition is actually quite straightforward: for nearby positions (0--7), we need finer-grained distinctions, so each gets its own independent position encoding. For somewhat more distant positions (e.g., 8--11), we don't need to distinguish so precisely, so they can share a position encoding. The farther away, the larger the range that can share, until reaching the specified range where clipping is applied.

### DeBERTa-Style

DeBERTa was also developed by Microsoft; it was released in June of the previous year, with the paper [*DeBERTa: Decoding-enhanced BERT with Disentangled Attention*](https://arxiv.org/abs/2006.03654). It recently gained renewed attention -- first because it was officially accepted at ICLR 2021, and second because it reached the top of the [SuperGLUE](https://super.gluebenchmark.com/) leaderboard, slightly surpassing T5.

DeBERTa's main improvement is also in position encoding. Again starting from the expansion in Eq. (7): T5 simply dropped terms 2 and 3, keeping only term 4 and replacing it with relative position encoding. DeBERTa does the opposite -- it drops term 4 and keeps terms 2 and 3, replacing them with relative position encoding (indeed, research is about enumerating all permutations and combinations to see which is optimal):

\[
\boldsymbol{q}_i \boldsymbol{k}_j^{\top} = \boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top} \boldsymbol{x}_j^{\top} + \boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top} \boldsymbol{R}_{i,j}^{\top} + \boldsymbol{R}_{j,i}\, \boldsymbol{W}_Q \boldsymbol{W}_K^{\top} \boldsymbol{x}_j^{\top} \tag{11}
\]

The design of \(\boldsymbol{R}_{i,j}\) also uses clipping as in Eq. (6), with nothing particularly special.

However, what's interesting about DeBERTa is that it provides **a new perspective on using relative and absolute position encoding**. It argues that most NLP tasks probably only need relative position information, but there are indeed some scenarios where absolute position information helps more. So it divides the entire model into two parts. Taking the Base-version MLM pre-training model as an example: it has 13 layers total; the first 11 use only relative position encoding (called the "Encoder"), and the last 2 add absolute position information (called the "Decoder"), with a shorthand EMD (Enhanced Mask Decoder). For downstream task fine-tuning, the first 11 Encoder layers plus 1 Decoder layer are used.

The SuperGLUE results affirm DeBERTa's value, but its various naming choices in the paper are really rather uncomfortable. For instance, its self-proclaimed "Encoder" and "Decoder" easily mislead people into thinking it's a Seq2Seq model. The abbreviation EMD also conflicts with Earth Mover's Distance. While name collisions are sometimes unavoidable, the names it collides with are all well-known objects in the ML community, making misunderstanding quite likely. I really don't know what the authors were thinking...

## Other Position Encodings

Although absolute and relative position encodings come in many flavors, they still fall within the classical paradigm. From the above introduction, we can still feel the heavy "formulaic" flavor. Beyond these, there are some approaches that don't play by the conventional rules, yet they also express position encoding.

### CNN-Style

Although the classic work applying CNN to NLP, [*Convolutional Sequence to Sequence Learning*](https://arxiv.org/abs/1705.03122), added position encoding, we know that typical CNN models -- especially CNN models in computer vision -- don't add any separate position encoding. So how do CNN models actually capture position information?

If I were to answer, my guess would be that the anisotropy of convolution kernels enables them to distinguish relative positions in different directions. However, the ICLR 2020 paper [*How Much Position Information Do Convolutional Neural Networks Encode?*](https://arxiv.org/abs/2001.08248) gives a perhaps surprising answer: **CNN models' position information is leaked through zero padding!**

We know that to maintain a certain feature map size during convolution, we typically pad the input with zeros. This paper shows that this operation gives the model the ability to recognize position information. That is to say, while kernel anisotropy is important, the most fundamental factor is the presence of zero padding -- so one can imagine that what's actually being extracted is the relative distance from the current position to the padding boundary.

However, this ability relies on CNN's locality. Global, prior-free structures like Attention don't benefit from this. Readers who are only interested in Transformer position encoding schemes can treat this as a broadening of horizons.

### Complex-Valued

Complex-valued position encoding is perhaps the most iconoclastic position encoding scheme of all. It comes from the ICLR 2020 paper [*Encoding Word Order in Complex Embeddings*](https://arxiv.org/abs/1912.12333). The paper's main idea combines properties of complex numbers with some basic principles to derive a position encoding form (Complex Order):

\[
\left[r_{j,1}\, e^{\text{i}(\omega_{j,1} k + \theta_{j,1})},\; \ldots,\; r_{j,2}\, e^{\text{i}(\omega_{j,2} k + \theta_{j,2})},\; \cdots,\; r_{j,d}\, e^{\text{i}(\omega_{j,d} k + \theta_{j,d})}\right] \tag{12}
\]

Here \(\text{i}\) is the imaginary unit, \(j\) represents a particular word, \(k\) represents the position of that word, and

\[
\begin{aligned}
\boldsymbol{r}_j &= [r_{j,1}, r_{j,2}, \cdots, r_{j,d}] \\
\boldsymbol{\omega}_j &= [\omega_{j,1}, \omega_{j,2}, \cdots, \omega_{j,d}] \\
\boldsymbol{\theta}_j &= [\theta_{j,1}, \theta_{j,2}, \cdots, \theta_{j,d}]
\end{aligned} \tag{13}
\]

represent three sets of word vectors for word \(j\). You read that right -- it indeed assumes each word has three sets of position-independent word vectors (of course, parameter sharing can reduce them to two or even one set), and the position-dependent word vector for position \(k\) is computed according to the formula above.

You might think introducing multiple sets of word vectors is its most iconoclastic aspect? Not so! We see that Eq. (12) is still in complex form -- and guess what it does next? Convert it to real numbers? No! It uses it directly in a **complex-valued model**! That is, it takes the complex model route: not only is the input Embedding layer complex, but every Transformer layer inside is complex. It even implements and compares complex versions of FastText, LSTM, CNN, and other models! The first author of this paper is Benyou Wang, whose related work is basically all centered around complex models -- a die-hard fan of complex models indeed.

### Fusion-Style

By a fortunate coincidence, using the complex number formalism, I have also conceived a rather clever position encoding that can **fuse absolute and relative position encoding into one**. I share it here; interested readers are welcome to discuss and explore together.[^rope-followup]

[^rope-followup]: This "fusion-style" encoding was later formalized as Rotary Position Embedding (RoPE) in the follow-up article [Transformer Upgrade Path: 2. Rotary Position Embedding](/translations/scientific-spaces/transformer-upgrade-2-rotary-position-embedding/).

For simplicity, let us first assume that \(\boldsymbol{q}_m\) and \(\boldsymbol{k}_n\) are two-dimensional row vectors at positions \(m\) and \(n\) respectively. Since they are two-dimensional, we can treat them as complex numbers. We know that the key operation in Attention is the inner product, which in complex notation is:

\[
\langle \boldsymbol{q}_m, \boldsymbol{k}_n \rangle = \text{Re}\left[\boldsymbol{q}_m \boldsymbol{k}_n^*\right] \tag{14}
\]

where \({}^*\) denotes the complex conjugate, the multiplication on the right is ordinary complex multiplication, and \(\text{Re}[\cdot]\) denotes taking the real part. In other words:

> The inner product of two 2D vectors equals, when viewed as complex numbers, the real part of the product of one complex number with the conjugate of the other.

If we multiply \(\boldsymbol{q}_m\) and \(\boldsymbol{k}_n\) by \(e^{\text{i}m\theta}\) and \(e^{\text{i}n\theta}\) respectively to get \(\boldsymbol{q}_m e^{\text{i}m\theta}\) and \(\boldsymbol{k}_n e^{\text{i}n\theta}\), then we have effectively given them absolute position encoding (since they explicitly depend on absolute positions \(m\) and \(n\)). Plugging these into the inner product, we get:

\[
\langle \boldsymbol{q}_m e^{\text{i}m\theta},\, \boldsymbol{k}_n e^{\text{i}n\theta} \rangle = \text{Re}\left[(\boldsymbol{q}_m e^{\text{i}m\theta})(\boldsymbol{k}_n e^{\text{i}n\theta})^*\right] = \text{Re}\left[\boldsymbol{q}_m \boldsymbol{k}_n^*\, e^{\text{i}(m-n)\theta}\right] \tag{15}
\]

Remarkably, the inner product depends only on the relative position \(m - n\)! This elegantly fuses absolute and relative position into one.

Note that we are not being as "wild" as Complex Order -- the above computation is essentially still in the real-number domain; we merely used complex numbers to carry out certain derivations. From the above result, for a two-dimensional real vector \([x, y]\) at position \(n\), treating it as a complex number and multiplying by \(e^{\text{i}n\theta}\), we get the identity:

\[
(x + y\text{i})\,e^{\text{i}n\theta} = (x\cos n\theta - y\sin n\theta) + \text{i}\,(x\sin n\theta + y\cos n\theta) \tag{16}
\]

This means that through the transformation:

\[
\begin{pmatrix} x \\ y \end{pmatrix} \to \begin{pmatrix} x\cos n\theta - y\sin n\theta \\ x\sin n\theta + y\cos n\theta \end{pmatrix} = \begin{pmatrix} x \\ y \end{pmatrix} \cos n\theta + \begin{pmatrix} -y \\ x \end{pmatrix} \sin n\theta \tag{17}
\]

we endow \([x, y]\) with absolute position information that, when used in Attention, is equivalent to relative position encoding. For vectors of more than two dimensions, one can group them in pairs and apply the same operation to each pair, with a different \(\theta\) for each group.

This gives us a position encoding scheme that fuses absolute and relative position into one. Formally, it resembles multiplicative absolute position encoding. By applying this encoding to \(\boldsymbol{q}\) and \(\boldsymbol{k}\), the effect is equivalent to relative position encoding. If explicit absolute position information is also needed, one can additionally apply this encoding to \(\boldsymbol{v}\). In summary, through an absolute-position operation, we can achieve the effect of both absolute and relative position encoding. Preliminary experiments show that it works, but it has not been thoroughly validated. Readers are welcome to try it out and share their findings.

## Summary

This article surveys a collection of position encoding methods, broadly divided into absolute, relative, and unconventional categories. From these, we can see all sorts of creative approaches. Finally, I shared my own conceived scheme that fuses absolute and relative position encoding, for interested readers' reference.

---

*Citation: Su, J. (2021, February 03). 让研究人员绞尽脑汁的Transformer位置编码 [Transformer Position Encodings That Rack Researchers' Brains]. Scientific Spaces. [https://kexue.fm/archives/8130](https://kexue.fm/archives/8130)*
