---
title: "A Brief History of Linear Attention: From Imitation and Innovation to Feeding Back"
subtitle: "Translated from [线性注意力简史：从模仿、创新到反哺](https://kexue.fm/archives/11033) by Jianlin Su (苏剑林)"
date: 2025-06-20T00:00:00+08:00
blurb: "Tracing linear attention from its origins as an approximation of softmax attention, through forget gates and test-time training, to DeltaNet and its recent feedback into softmax attention via DeltaFormer and PaTH."
tags: ["translation", "linear-attention", "rnn", "attention"]
math: true
---

*Translator's note (Opus 4.6): This is an English translation of [线性注意力简史：从模仿、创新到反哺](https://kexue.fm/archives/11033) by Jianlin Su (苏剑林), originally published on June 20, 2025 on [Scientific Spaces (科学空间)](https://kexue.fm). The translation preserves the author's first-person voice.*

<hr class="section-divider">

In the Chinese-language community, this blog was probably among the earlier ones to pay attention to linear attention. When I wrote the first related post [*Exploring Linear Attention: Must Attention Have a Softmax?*](https://kexue.fm/archives/7546) in 2020, the community was still mainly discussing Softmax Attention in the context of BERT. In hindsight, thinking about linear attention during the BERT era was not particularly wise, because training lengths were short and models were primarily Encoders — linear attention offered essentially no advantage there. I also wrote [*The Linear Transformer Is Probably Not the Model You've Been Waiting For*](https://kexue.fm/archives/8610) to express this view.

It was not until the advent of ChatGPT, which pushed everyone toward decoder-only generative models, that the landscape changed. This paradigm is highly compatible with the RNN form of linear attention. At the same time, the pursuit of ever-longer training lengths made the quadratic complexity bottleneck of Softmax Attention increasingly painful. Under these new circumstances, linear attention has become ever more competitive, and has even begun to show signs of "feeding back" (反哺) into Softmax Attention.

## Quadratic Complexity

First, let us introduce some notation:

\[
\boldsymbol{q}_i, \boldsymbol{k}_i, \boldsymbol{v}_i, \boldsymbol{o}_i \in \mathbb{R}^{d \times 1}
\]

\[
\boldsymbol{Q} = [\boldsymbol{q}_1, \boldsymbol{q}_2, \cdots, \boldsymbol{q}_n]^{\top} \in \mathbb{R}^{n \times d}
\]

\[
\boldsymbol{K} = [\boldsymbol{k}_1, \boldsymbol{k}_2, \cdots, \boldsymbol{k}_n]^{\top} \in \mathbb{R}^{n \times d}
\]

\[
\boldsymbol{V} = [\boldsymbol{v}_1, \boldsymbol{v}_2, \cdots, \boldsymbol{v}_n]^{\top} \in \mathbb{R}^{n \times d}
\]

\[
\boldsymbol{O} = [\boldsymbol{o}_1, \boldsymbol{o}_2, \cdots, \boldsymbol{o}_n]^{\top} \in \mathbb{R}^{n \times d}
\]

An attention model is essentially a mapping \(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V} \to \boldsymbol{O}\). This article focuses primarily on the causal setting, which means \(\boldsymbol{o}_t\) depends on at most \(\boldsymbol{Q}_{[:t]}, \boldsymbol{K}_{[:t]}, \boldsymbol{V}_{[:t]}\). In principle, the \(d\) for \(\boldsymbol{Q}, \boldsymbol{K}\) and the \(d\) for \(\boldsymbol{V}, \boldsymbol{O}\) can differ — for example, [*GAU*](https://kexue.fm/archives/8934) and [*MLA*](https://kexue.fm/archives/10091) do exactly this — but simplifying them to the same value does not change the essence of the problem.

Standard Softmax Attention usually refers to the attention mechanism proposed in [*Attention is All You Need*](https://kexue.fm/archives/4765):

\[
\boldsymbol{O} = \text{softmax}(\boldsymbol{Q}\boldsymbol{K}^{\top} + \log \boldsymbol{M})\boldsymbol{V} \tag{2}
\]

Here we have omitted the scaling factor \(1/\sqrt{d}\), since it can always be absorbed into \(\boldsymbol{Q}, \boldsymbol{K}\). The \(\text{softmax}\) is applied along the second dimension (exponential normalization), and \(\boldsymbol{M} \in \mathbb{R}^{n \times n}\) is a lower-triangular matrix called the mask matrix, defined as:

\[
M_{i,j} = \begin{cases} 1, & i \geq j \\ 0, & i < j \end{cases} \tag{3}
\]

\(\log \boldsymbol{M}\) means taking the \(\log\) of each element, where \(\log 0 = -\infty\). Writing Softmax Attention in component form gives:

\[
\boldsymbol{o}_t = \frac{\sum_{j=1}^t \exp(\boldsymbol{q}_t^{\top}\boldsymbol{k}_j) \boldsymbol{v}_j}{\sum_{j=1}^t \exp(\boldsymbol{q}_t^{\top}\boldsymbol{k}_j)} \tag{4}
\]

The denominator's main role is to maintain numerical stability. Moreover, if we apply RMSNorm to \(\boldsymbol{O}\), the denominator cancels out automatically. So the core of Softmax Attention is the numerator:

\[
\boldsymbol{O} = \exp(\boldsymbol{Q}\boldsymbol{K}^{\top} + \log \boldsymbol{M})\boldsymbol{V} = (\exp(\boldsymbol{Q}\boldsymbol{K}^{\top}) \odot \boldsymbol{M})\boldsymbol{V} \tag{5}
\]

where \(\odot\) is the Hadamard product and \(\exp\) is applied element-wise. It is easy to see that the denominator is simply obtained by replacing \(\boldsymbol{V}\) with an \(n \times 1\) all-ones matrix — we can add it back if needed. The standard implementation of Softmax Attention requires computing the \(n \times n\) matrix \(\exp(\boldsymbol{Q}\boldsymbol{K}^{\top})\), so both space and time complexity are proportional to \(n^2\). The appearance of [*Flash Attention*](https://arxiv.org/abs/2205.14135) reduced the space requirement, but the quadratic time complexity remains unavoidable.

## The Original Form

The earliest approach to linear attention was mainly to imitate and approximate Softmax Attention. The simplest scheme is to directly remove the \(\exp\):

\[
\boldsymbol{O} = (\boldsymbol{Q}\boldsymbol{K}^{\top} \odot \boldsymbol{M})\boldsymbol{V} \tag{6}
\]

For simplicity, we adopt the convention that **matrix multiplication has higher precedence than the Hadamard product**, which saves us a pair of parentheses. Why is this form called "linear" attention? To quickly see this, consider the non-causal version without \(\odot \boldsymbol{M}\): then \(\boldsymbol{O} = (\boldsymbol{Q}\boldsymbol{K}^{\top})\boldsymbol{V} = \boldsymbol{Q}(\boldsymbol{K}^{\top}\boldsymbol{V})\). Note that computing \(\boldsymbol{K}^{\top}\boldsymbol{V}\) costs \(\mathcal{O}(nd^2)\), producing a \(d \times d\) matrix, and then multiplying by \(\boldsymbol{Q}\) also costs \(\mathcal{O}(nd^2)\), so the complexity depends linearly on \(n\).

For the causal version (6), we can understand it from the component form:

\[
\boldsymbol{o}_t = \sum_{j=1}^t \boldsymbol{v}_j (\boldsymbol{k}_j^{\top} \boldsymbol{q}_t) = \sum_{j=1}^t (\boldsymbol{v}_j \boldsymbol{k}_j^{\top}) \boldsymbol{q}_t = \left(\sum_{j=1}^t \boldsymbol{v}_j \boldsymbol{k}_j^{\top}\right) \boldsymbol{q}_t \tag{7}
\]

If we denote the bracketed part as \(\boldsymbol{S}_t\), then:

\[
\boldsymbol{o}_t = \boldsymbol{S}_t \boldsymbol{q}_t, \qquad \boldsymbol{S}_t = \boldsymbol{S}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^{\top} \tag{8}
\]

Thus, the causal form of attention can be written as a linear RNN with state \(\boldsymbol{S}_t\), where each step has constant complexity and total complexity is proportional to sequence length \(n\). Note the appearance of "linear RNN" here — it is a broader concept. Linear attention is one type of linear RNN, and linear RNNs have also developed independently for a period, such as [*LRU*](https://kexue.fm/archives/9554) and [*SSM*](https://kexue.fm/tag/ssm/) that I have previously introduced. However, the most competitive linear architectures recently all take the form of linear attention.

In its early days, linear attention had several conspicuous features that imitated Softmax Attention. For instance, a denominator was added to equation (6) for normalization, and to make that normalization work, \(\boldsymbol{k}_j^{\top} \boldsymbol{q}_t\) had to be non-negative, so non-negative activation functions were applied to \(\boldsymbol{Q}, \boldsymbol{K}\). A series of works represented by [*Performer*](https://kexue.fm/archives/7921) and [*RFA*](https://arxiv.org/abs/2103.02143) even took approximating \(\exp(\boldsymbol{Q}\boldsymbol{K}^{\top})\) as their starting point for model construction.

However, later research such as [*The Devil in Linear Transformer*](https://arxiv.org/abs/2210.10340) found that normalizing along the sequence length dimension cannot completely avoid numerical instability. It is better to simply normalize after the fact:

\[
\boldsymbol{O} = \text{RMSNorm}((\boldsymbol{Q}\boldsymbol{K}^{\top} \odot \boldsymbol{M})\boldsymbol{V}) \tag{9}
\]

Since normalization is no longer needed, applying non-negative activation functions to \(\boldsymbol{Q}, \boldsymbol{K}\) to ensure \(\boldsymbol{k}_j^{\top} \boldsymbol{q}_t \geq 0\) is no longer essential. Does adding (not necessarily non-negative) activation functions to \(\boldsymbol{Q}, \boldsymbol{K}\) still have value? My view is: adding activation functions is anyone's prerogative, and it is possible that some particular activation function might yield better results, but doing so does not change the form of linear attention and thus does not affect our description. Moreover, existing results show that not adding one already works well enough.

## Fancy Forget Gates

From equation (8) we can see that the current linear attention is essentially a cumulative sum (\(\text{cumsum}\)) — it adds all historical information with equal weight. It is not hard to imagine that when enough tokens have been accumulated, the information share of each token becomes tiny, and a fixed-size \(\boldsymbol{S}_t\) matrix alone cannot accurately reconstruct any individual token. The intuitive analogy is that every token's memory becomes blurry.

To mitigate this problem, [*RetNet*](https://arxiv.org/abs/2307.08621) introduced a forgetting effect into linear attention:

\[
\boldsymbol{o}_t = \boldsymbol{S}_t \boldsymbol{q}_t, \qquad \boldsymbol{S}_t = \gamma \boldsymbol{S}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^{\top} \tag{10}
\]

where the decay factor \(\gamma \in (0, 1)\). In RetNet it is set as a constant; others have made it a trainable parameter or replaced \(\gamma\) with a diagonal matrix, and so on. The linear attention used in [*MiniMax-01*](https://arxiv.org/abs/2501.08313) is also of this type. Note that decay factors existed before RetNet, but they mostly appeared in the form of linear RNNs, such as the [*LRU*](https://kexue.fm/archives/9554) and [*SSM*](https://kexue.fm/tag/ssm/) mentioned in the previous section. RetNet was likely the first to combine it with linear attention. After introducing the decay factor, the model tends to forget more distant historical information, at least preserving the resolution of recent tokens. In plain terms, this embodies the "recency bias" that aligns with the characteristics of language models, and therefore tends to work better.

Additionally, a noteworthy detail is that RetNet also added [*RoPE*](https://kexue.fm/archives/9403) to \(\boldsymbol{Q}, \boldsymbol{K}\), which amounts to extending the decay factor to the complex number \(\gamma e^{i\theta}\). From the [*LRU*](https://kexue.fm/archives/9554) perspective, this means considering complex eigenvalues. Although adding positional encoding to an RNN might seem somewhat incongruous, some experiments — for example, the recent [*TransXSSM*](https://arxiv.org/abs/2506.09507) — show that adding RoPE to linear attention can have a positive effect. Of course, this may depend on the specific model variant and experimental setup.

A simple generalization of equation (10) is to replace \(\gamma\) with a function of position \(t\), namely \(\gamma_t\), which was already reflected in [*SSM*](https://kexue.fm/tag/ssm/). Later, [*DFW*](https://arxiv.org/abs/2210.04243), [*Mamba*](https://arxiv.org/abs/2312.00752), [*Mamba2*](https://arxiv.org/abs/2405.21060), and other works generalized it to be input-dependent, forming a line of work on "data-dependent decay." This is actually very similar to the "forget gate" in classical nonlinear RNNs like GRU and LSTM, except that to maintain the model's linearity, the forget gate's dependence on the state (such as \(\boldsymbol{S}_t\)) is removed.

Why do we prefer linear RNNs? Because linear RNNs can almost always find some way to train in parallel, making them competitive with Softmax Attention — they are not inferior in either training efficiency or inference efficiency. The "universal solution" for parallelization is to convert the problem into a [Prefix Sum](https://en.wikipedia.org/wiki/Prefix_sum) problem and then apply Associative Scan; the general idea was briefly introduced in the "Parallelization" section of [*Google's New Work Attempts to "Revive" RNNs: Can RNNs Shine Again?*](https://kexue.fm/archives/9554).

However, the "universal solution" is not GPU-efficient. What GPUs are best at is matrix multiplication, so finding parallel algorithms that make heavy use of matrix multiplication is ideal. In fact, even without full parallelism, simply finding a chunk-by-chunk recurrence that makes heavy use of matrix multiplication can significantly improve training efficiency. This in turn imposes constraints on the model: only outer-product-form forget gates can achieve this goal. A typical counterexample is Mamba, which has a non-outer-product forget gate and cannot fully exploit GPU performance — hence the subsequent development of Mamba2 and [*GLA*](https://arxiv.org/abs/2312.06635).

## Test-Time Training

By this point, linear attention has evolved from a simple imitation of Softmax Attention to incorporating static decay factors and even "data-dependent decay," developing its own distinctive character and demonstrating value on many tasks. However, most of these advances were designed by hand based on experience. We cannot help but ask: **Is there a higher-level principle to guide the design of linear attention, or even general sequence models (token-mixers)?**

To this question, [*TTT (Test-Time Training)*](https://arxiv.org/abs/2407.04620) offers its own answer: it views the construction of sequence models as an "online learning" problem, and proposes using optimizers to build (not necessarily linear) RNNs. Specifically, it treats \(\boldsymbol{K}, \boldsymbol{V}\) as training data pairs \((\boldsymbol{k}_1, \boldsymbol{v}_1), (\boldsymbol{k}_2, \boldsymbol{v}_2), \cdots, (\boldsymbol{k}_t, \boldsymbol{v}_t)\), trains a model \(\boldsymbol{v} = \boldsymbol{f}(\boldsymbol{S}_t; \boldsymbol{k})\) on this data, and finally outputs \(\boldsymbol{o}_t = \boldsymbol{f}(\boldsymbol{S}_t; \boldsymbol{q}_t)\), where \(\boldsymbol{S}_t\) represents the model parameters and the model architecture is largely arbitrary.

What does this have to do with RNNs? Simple: optimizers like SGD and Adam are essentially RNNs over the model parameters! This observation is not new — as early as 2017, during the heyday of meta-learning, researchers had already proposed and exploited this connection, though the idea then was to use an RNN (LSTM) to simulate a better optimizer. See [*Optimization as a Model for Few-Shot Learning*](https://openreview.net/forum?id=rJY0-Kcll) for details.

As the saying goes, "what goes around comes around" (风水轮流转). Years later, TTT turned this around and proposed building RNNs through optimizers. The process is as follows: the current model parameters are \(\boldsymbol{S}_{t-1}\); the optimizer (SGD) receives new data \((\boldsymbol{k}_t, \boldsymbol{v}_t)\) and updates the model parameters to \(\boldsymbol{S}_t\); finally it returns the prediction for \(\boldsymbol{q}_t\): \(\boldsymbol{f}(\boldsymbol{S}_{t-1}; \boldsymbol{q}_t)\), and so on. The RNN realized by TTT can be written uniformly as:

\[
\boldsymbol{o}_t = \boldsymbol{f}(\boldsymbol{S}_t; \boldsymbol{q}_t), \qquad \boldsymbol{S}_t = \boldsymbol{S}_{t-1} - \eta_t \nabla_{\boldsymbol{S}_{t-1}} \mathcal{L}(\boldsymbol{f}(\boldsymbol{S}_{t-1}; \boldsymbol{k}_t), \boldsymbol{v}_t) \tag{11}
\]

where \(\mathcal{L}(\boldsymbol{f}(\boldsymbol{S}_{t-1}; \boldsymbol{k}_t), \boldsymbol{v}_t)\) is the loss function for the current data \((\boldsymbol{k}_t, \boldsymbol{v}_t)\) under the current parameters \(\boldsymbol{S}_{t-1}\), and \(\eta_t\) is the learning rate, which — following the "data-dependent decay" of the previous section — can also be made data-dependent. This form covers a very broad range of RNN models. For instance, equations (8) and (10) are both special cases:

| | RNN | \(\boldsymbol{o}_t\) | \(\boldsymbol{f}(\boldsymbol{S};\boldsymbol{k})\) | \(\mathcal{L}(\boldsymbol{f}(\boldsymbol{S};\boldsymbol{k}),\boldsymbol{v})\) | \(\eta_t\) |
|---|---|---|---|---|---|
| (8) | \(\boldsymbol{S}_t = \boldsymbol{S}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^{\top}\) | \(\boldsymbol{S}_t \boldsymbol{q}_t\) | \(\boldsymbol{S}\boldsymbol{k}\) | \(-\boldsymbol{v}^{\top}(\boldsymbol{S}\boldsymbol{k})\) | 1 |
| (10) | \(\boldsymbol{S}_t = \gamma\boldsymbol{S}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^{\top}\) | \(\boldsymbol{S}_t \boldsymbol{q}_t\) | \(\boldsymbol{S}\boldsymbol{k}\) | \(-\boldsymbol{v}^{\top}(\boldsymbol{S}\boldsymbol{k}) + \frac{1-\gamma}{2}\lVert\boldsymbol{S}\rVert_F^2\) | 1 |

The original TTT paper focused on exploring nonlinear RNNs with mini-batches. Later, [*Titans*](https://arxiv.org/abs/2501.00663) added momentum to TTT's SGD. After that, [*Test-Time Training Done Right*](https://arxiv.org/abs/2505.23884) explored large-batch TTT usage and even the "TTT + Muon" combination. Note that TTT only uses the optimizer to construct the RNN; the trainable parameters outside the RNN, such as those for \(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}\), are still trained by the overall optimizer after assembling the full model.

A more thought-provoking question is: why can TTT serve as a "guiding principle" for constructing RNNs? The core objective of an RNN is to effectively compress historical data into a fixed-size state, and model parameters are exactly fixed-size. Training a model is, in a sense, compressing the training data into the model weights. TTT exploits exactly this alignment with the RNN objective. Put bluntly: if we view the RNN as a compression task, TTT treats the model \(\boldsymbol{f}\) as the "decompressor," its weights as the "compressed archive," the compression algorithm as SGD, and the compression ratio as the loss \(\mathcal{L}\).

This way, we no longer need to painstakingly design recurrence relations. Instead, we design the model \(\boldsymbol{f}\) and the loss \(\mathcal{L}\), and we can gauge whether an RNN is strong or reliable simply by examining the corresponding \(\boldsymbol{f}\) and \(\mathcal{L}\).

Beyond this, TTT's construction of RNNs via online learning means the resulting RNN is naturally well-suited for ICL (In-Context Learning) tasks, which is another advantage of TTT as a "guiding principle." Earlier, [*Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers*](https://arxiv.org/abs/2212.10559) even went the other direction, stripping the Softmax from Softmax Attention to get linear attention in order to explain its ICL capability — viewed through today's lens, they essentially constructed the corresponding TTT.

## Out with the Old, In with the New

For example, the loss function corresponding to the earliest linear attention is \(-\boldsymbol{v}^{\top}(\boldsymbol{S}\boldsymbol{k})\) — one look and you can tell this is an unreliable objective, because it is unbounded below, which could cause \(\boldsymbol{S}\) to diverge to infinity. By contrast, RetNet added an L2 regularization term to the loss function, avoiding this risk and — from an optimization perspective — mitigating the risk of overfitting, thus yielding a better RNN.

However, while using the inner product as a loss function is concise and has some rationale, it does not directly encourage \(\boldsymbol{S}\boldsymbol{k} = \boldsymbol{v}\), so it is not an ideal regression loss. A better objective function is the squared loss, \(\frac{1}{2}\lVert\boldsymbol{S}\boldsymbol{k} - \boldsymbol{v}\rVert^2\). Substituting it into TTT's formula (11) gives:

\[
\boldsymbol{o}_t = \boldsymbol{S}_t \boldsymbol{q}_t, \qquad \boldsymbol{S}_t = \boldsymbol{S}_{t-1} - \eta_t \underbrace{(\boldsymbol{S}_{t-1} \boldsymbol{k}_t - \boldsymbol{v}_t)\boldsymbol{k}_t^{\top}}_{\nabla_{\boldsymbol{S}_{t-1}} \frac{1}{2}\lVert\boldsymbol{S}_{t-1}\boldsymbol{k}_t - \boldsymbol{v}_t\rVert^2} \tag{12}
\]

This is DeltaNet. The name comes from [*Parallelizing Linear Transformers with the Delta Rule over Sequence Length*](https://arxiv.org/abs/2406.06484); the idea was earlier proposed by [*Linear Transformers Are Secretly Fast Weight Programmers*](https://arxiv.org/abs/2102.11174). Notice that \(\eta_t (\boldsymbol{S}_{t-1} \boldsymbol{k}_t - \boldsymbol{v}_t)\boldsymbol{k}_t^{\top} = (\boldsymbol{S}_{t-1} (\sqrt{\eta_t}\boldsymbol{k}_t) - (\sqrt{\eta_t}\boldsymbol{v}_t))(\sqrt{\eta_t}\boldsymbol{k}_t)^{\top}\), which means \(\eta_t\) can always be absorbed into the definitions of \(\boldsymbol{k}_t, \boldsymbol{v}_t\). So for the rest of our analysis we consider only the case \(\eta_t = 1\):

\[
\boldsymbol{S}_t = \boldsymbol{S}_{t-1} - (\boldsymbol{S}_{t-1} \boldsymbol{k}_t - \boldsymbol{v}_t)\boldsymbol{k}_t^{\top} = \boldsymbol{S}_{t-1} - (\boldsymbol{S}_{t-1} \boldsymbol{k}_t)\boldsymbol{k}_t^{\top} + \boldsymbol{v}_t \boldsymbol{k}_t^{\top} = \boldsymbol{S}_{t-1}(\boldsymbol{I} - \boldsymbol{k}_t \boldsymbol{k}_t^{\top}) + \boldsymbol{v}_t \boldsymbol{k}_t^{\top} \tag{13}
\]

If needed, we can substitute \(\sqrt{\eta_t}\boldsymbol{k}_t, \sqrt{\eta_t}\boldsymbol{v}_t\) for \(\boldsymbol{k}_t, \boldsymbol{v}_t\) to recover \(\eta_t\). Comparing with the earliest form of linear attention (8), DeltaNet's difference is that before adding \(\boldsymbol{v}_t \boldsymbol{k}_t^{\top}\), it first subtracts \((\boldsymbol{S}_{t-1} \boldsymbol{k}_t)\boldsymbol{k}_t^{\top}\), where \(\boldsymbol{S}_{t-1} \boldsymbol{k}_t\) can be interpreted as the prediction of the new input \(\boldsymbol{k}_t\) under the old model \(\boldsymbol{S}_{t-1}\).

Intuitively, "subtract first, then add" means first removing the model's old knowledge about \(\boldsymbol{k}_t\), then supplementing it with new knowledge from \((\boldsymbol{k}_t, \boldsymbol{v}_t)\), achieving an "out with the old, in with the new" (除旧迎新) effect. This rule is called the "[Delta Rule](https://en.wikipedia.org/wiki/Delta_rule)" — the origin of the "Delta" in DeltaNet. The Delta Rule is not new; it is also known as [Least Mean Square](https://en.wikipedia.org/wiki/Least_mean_squares_filter) or the Widrow-Hoff Algorithm, dating back to the 1960s. In fact, truly novel ideas in this field are rare — many modifications can be traced back to some "ancient" work, and current efforts are mainly focused on identifying which parts can be made scalable.

It should also be pointed out that chronologically, DeltaNet came before TTT. Understanding RNNs from the online learning perspective had already appeared sporadically in earlier works, but TTT systematically proposed this "guiding principle" and used it to construct new RNN models. We placed TTT before DeltaNet in this presentation to make the exposition flow more naturally.

Some readers may wonder: is DeltaNet still a linear RNN? The answer is yes. When we say "linear RNN," we mean that the recurrence formula depends linearly on the state variable, though the dependence on inputs or \(\boldsymbol{q}, \boldsymbol{k}, \boldsymbol{v}\) can be nonlinear (of course, different forms of dependence affect parallelization efficiency). From equation (13), we can see that the right-hand side contains only the first power of \(\boldsymbol{S}_{t-1}\), so it satisfies the definition of linearity.

## Inversion and Generalizations

As mentioned earlier, the ideal (i.e., GPU-efficient) parallel algorithm for linear RNNs is one that makes heavy use of matrix multiplication. To achieve this, let us first rewrite DeltaNet as:

\[
\boldsymbol{S}_t = \boldsymbol{S}_{t-1} + (\boldsymbol{v}_t - \boldsymbol{S}_{t-1} \boldsymbol{k}_t)\boldsymbol{k}_t^{\top} \tag{14}
\]

Define \(\boldsymbol{u}_t = \boldsymbol{v}_t - \boldsymbol{S}_{t-1} \boldsymbol{k}_t\), so \(\boldsymbol{S}_t = \boldsymbol{S}_{t-1} + \boldsymbol{u}_t \boldsymbol{k}_t^{\top}\). That is, it simply replaces \(\boldsymbol{V}\) with \(\boldsymbol{U} = [\boldsymbol{u}_1, \boldsymbol{u}_2, \cdots, \boldsymbol{u}_n]^{\top}\) in the earliest linear attention. Iterating \(t-1\) times, we have:

\[
\boldsymbol{S}_{t-1} = \sum_{j=1}^{t-1} \boldsymbol{u}_j \boldsymbol{k}_j^{\top} \quad \Rightarrow \quad \boldsymbol{u}_t = \boldsymbol{v}_t - \left(\sum_{j=1}^{t-1} \boldsymbol{u}_j \boldsymbol{k}_j^{\top}\right)\boldsymbol{k}_t = \boldsymbol{v}_t - \sum_{j=1}^{t-1} \boldsymbol{u}_j (\boldsymbol{k}_j^{\top} \boldsymbol{k}_t) \tag{15}
\]

The last equality in matrix form is \(\boldsymbol{U} = \boldsymbol{V} - (\boldsymbol{K}\boldsymbol{K}^{\top} \odot \boldsymbol{M}^{-})\boldsymbol{U}\), where \(\boldsymbol{M}^{-} = \boldsymbol{M} - \boldsymbol{I}\). This is a system of linear equations whose solution can be expressed directly as:

\[
\boldsymbol{U} = (\boldsymbol{I} + \underbrace{\boldsymbol{K}\boldsymbol{K}^{\top} \odot \boldsymbol{M}^{-}}_{\text{denote as } \boldsymbol{B}})^{-1}\boldsymbol{V} \tag{16}
\]

Here we encounter \((\boldsymbol{I} + \boldsymbol{B})^{-1}\), the inverse of an \(n \times n\) matrix, with standard complexity \(\mathcal{O}(n^3)\) — even higher than Softmax Attention! Fortunately, we do not need the explicit inverse; we only need \(\boldsymbol{U}\), which can be obtained by solving the system \((\boldsymbol{I} + \boldsymbol{B})\boldsymbol{U} = \boldsymbol{V}\), reducing complexity to \(\mathcal{O}(n^2)\). Furthermore, exploiting the fact that \(\boldsymbol{I} + \boldsymbol{B}\) is lower-triangular and \(\boldsymbol{B}\) has low-rank structure, the complexity can be reduced to linear, and when written as block matrix multiplications, GPU utilization becomes efficient. These details can only be found in the original paper; this article focuses on the main mathematical principles.

After DeltaNet, [*Gated DeltaNet (GDN)*](https://arxiv.org/abs/2412.06464) further introduced forget gates into DeltaNet — a foreseeable development. Gated DeltaNet's original formulation is:

\[
\boldsymbol{S}_t = \alpha_t \boldsymbol{S}_{t-1}(\boldsymbol{I} - \beta_t \boldsymbol{k}_t \boldsymbol{k}_t^{\top}) + \beta_t \boldsymbol{v}_t \boldsymbol{k}_t^{\top} \tag{17}
\]

However, I personally think this formulation explicitly breaks the Delta Rule. A better formulation would follow [*Comba*](https://arxiv.org/abs/2506.02475), applying the decay only to the first \(\boldsymbol{S}_{t-1}\):

\[
\boldsymbol{S}_t = \gamma_t \boldsymbol{S}_{t-1} + \eta_t (\boldsymbol{v}_t - \boldsymbol{S}_{t-1} \boldsymbol{k}_t)\boldsymbol{k}_t^{\top} \tag{18}
\]

This corresponds to the loss function \(\frac{1}{2}\lVert\boldsymbol{S}\boldsymbol{k} - \boldsymbol{v}\rVert^2 + \frac{1-\gamma}{\eta}\lVert\boldsymbol{S}\rVert_F^2\). Of course, mathematically the two formulations are equivalent:

\[
\alpha_t \boldsymbol{S}_{t-1}(\boldsymbol{I} - \beta_t \boldsymbol{k}_t \boldsymbol{k}_t^{\top}) + \beta_t \boldsymbol{v}_t \boldsymbol{k}_t^{\top} = \alpha_t \boldsymbol{S}_{t-1} + \alpha_t \beta_t (\boldsymbol{v}_t / \alpha_t - \boldsymbol{S}_{t-1} \boldsymbol{k}_t)\boldsymbol{k}_t^{\top} \tag{19}
\]

That is, \(\gamma_t = \alpha_t\), \(\eta_t = \alpha_t \beta_t\), and absorbing \(1/\alpha_t\) into \(\boldsymbol{v}_t\) converts it to the latter form. So there is no mathematical difference between the two. Since \(\alpha_t\) will be close to 1 in most cases, there is probably no difference in capability either (Comba claims (18) is slightly better) — it is just that the latter more intuitively preserves the look of the Delta Rule.

Theoretically, Gated DeltaNet can also be written in DeltaNet form. Simply define \(\bar{\alpha}_t = \prod_{j=1}^t \alpha_j\), then divide both sides of equation (17) by \(\bar{\alpha}_t\):

\[
\bar{\alpha}_t^{-1}\boldsymbol{S}_t = \bar{\alpha}_{t-1}^{-1}\boldsymbol{S}_{t-1}(\boldsymbol{I} - \beta_t \boldsymbol{k}_t \boldsymbol{k}_t^{\top}) + \beta_t (\bar{\alpha}_t^{-1}\boldsymbol{v}_t)\boldsymbol{k}_t^{\top} \tag{20}
\]

Combined with \(\boldsymbol{o}_t = \boldsymbol{S}_t \boldsymbol{q}_t = (\bar{\alpha}_t^{-1}\boldsymbol{S}_t)(\bar{\alpha}_t \boldsymbol{q}_t)\), we see that by setting \(\bar{\alpha}_t \boldsymbol{q}_t\) and \(\bar{\alpha}_t^{-1}\boldsymbol{v}_t\) as the new \(\boldsymbol{q}_t\) and \(\boldsymbol{v}_t\), it simplifies to DeltaNet form. However, this result only has theoretical value in certain situations (such as deriving the attention matrix in the next section), because in practice, for sufficiently large \(t\), one of \(\bar{\alpha}_t\) or \(\bar{\alpha}_t^{-1}\) will inevitably overflow.

After DeltaNet there is another generalization, [*DeltaProduct*](https://arxiv.org/abs/2502.10297), which expands \(\boldsymbol{k}, \boldsymbol{v}\) by a constant factor before applying DeltaNet or Gated DeltaNet, attempting to enhance the model's state-tracking ability. However, in my aesthetic judgment, rather than expanding by a constant factor like DeltaProduct, it would be better to try quadratic-complexity RNNs as in [*The Spacetime Chapter: Viewing Attention as a Quadratic-Complexity RNN*](https://kexue.fm/archives/10017), to see if there is an opportunity to surpass Softmax Attention.

## Feeding Back in Progress

Speaking of surpassing Softmax Attention, as mentioned at the beginning, today's linear attention can not only compete with Softmax Attention but has even begun "feeding back" into it. This may seem incredible, but upon reflection it is not hard to understand. In a sense, Softmax Attention has been regressing in recent years: from MHA to GQA to MQA, all are subtractive changes made to compress the KV cache. Linear attention has no KV cache problem, so it has been steadily moving in a better direction.

To see this more clearly, let us write out all the attention mechanisms mentioned above in matrix form:

| | Formula |
|---|---|
| Softmax Attention | \((\exp(\boldsymbol{Q}\boldsymbol{K}^{\top}) \odot \boldsymbol{M})\boldsymbol{V}\) |
| Earliest Linear Attention | \((\boldsymbol{Q}\boldsymbol{K}^{\top} \odot \boldsymbol{M})\boldsymbol{V}\) |
| With Forget Gate | \((\boldsymbol{Q}\boldsymbol{K}^{\top} \odot \boldsymbol{\Gamma})\boldsymbol{V}\) |
| DeltaNet | \((\boldsymbol{Q}\boldsymbol{K}^{\top} \odot \boldsymbol{M})(\boldsymbol{I} + \boldsymbol{K}\boldsymbol{K}^{\top} \odot \boldsymbol{M}^{-})^{-1}\boldsymbol{V}\) |
| Gated DeltaNet | \((\boldsymbol{Q}\boldsymbol{K}^{\top} \odot \boldsymbol{\Gamma})(\boldsymbol{I} + \boldsymbol{K}\boldsymbol{K}^{\top} \odot \boldsymbol{\Gamma}^{-})^{-1}\boldsymbol{V}\) |

where:

\[
\Gamma_{i,j} = \begin{cases} \prod_{\tau=j+1}^{i} \gamma_{\tau}, & i > j \\ 1, & i = j \\ 0, & i < j \end{cases} \tag{21}
\]

and \(\boldsymbol{\Gamma}^{-} = \boldsymbol{\Gamma} - \boldsymbol{I}\). Seen this way, Softmax Attention's form has only stayed at the level of the earliest linear attention (which, of course, also attests to its strength). So how does the "feeding back" work? First, we need a method to convert Softmax Attention into linear attention. This is not difficult — as early as [*Transformer Upgrade Path 5: As Infinite-Dimensional Linear Attention*](https://kexue.fm/archives/8601) we summarized three approaches for converting Softmax Attention into *infinite-dimensional* linear attention.

In short, there exists a mapping \(\phi\) that maps \(\boldsymbol{Q}, \boldsymbol{K}\) from \(n \times d\) to \(n \times \infty\), satisfying \(\exp(\boldsymbol{Q}\boldsymbol{K}^{\top}) = \phi(\boldsymbol{Q})\phi(\boldsymbol{K})^{\top}\) — this is the "kernel trick." The rest is straightforward: substitute \(\phi(\boldsymbol{Q}), \phi(\boldsymbol{K})\) for \(\boldsymbol{Q}, \boldsymbol{K}\) in the linear attention formulas in the table above, then recover the \(\exp\) and normalize to obtain new Softmax Attention variants. For example, substituting into the forget-gate formula:

\[
(\phi(\boldsymbol{Q})\phi(\boldsymbol{K})^{\top} \odot \boldsymbol{\Gamma})\boldsymbol{V} = \exp(\boldsymbol{Q}\boldsymbol{K}^{\top} + \log \boldsymbol{\Gamma})\boldsymbol{V} \tag{22}
\]

If \(\gamma_t\) is a constant, this is in fact [*ALiBi*](https://arxiv.org/abs/2108.12409) from [*Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*](https://arxiv.org/abs/2108.12409). If \(\gamma_t\) is input-dependent, it becomes [*FoX*](https://arxiv.org/abs/2503.02130) from [*Forgetting Transformer: Softmax Attention with a Forget Gate*](https://arxiv.org/abs/2503.02130).

A more interesting result is DeltaFormer, proposed in [*Understanding Transformer from the Perspective of Associative Memory*](https://arxiv.org/abs/2505.19488). As the name suggests, it is the Softmax Attention version of DeltaNet. Substituting \(\phi(\boldsymbol{Q}), \phi(\boldsymbol{K})\) for \(\boldsymbol{Q}, \boldsymbol{K}\) in DeltaNet:

\[
(\phi(\boldsymbol{Q})\phi(\boldsymbol{K})^{\top} \odot \boldsymbol{M})(\boldsymbol{I} + \phi(\boldsymbol{K})\phi(\boldsymbol{K})^{\top} \odot \boldsymbol{M}^{-})^{-1}\boldsymbol{V} = \underbrace{\exp(\boldsymbol{Q}\boldsymbol{K}^{\top} + \log \boldsymbol{M})}_{\text{denote as } \boldsymbol{A}}(\boldsymbol{I} + \underbrace{\exp(\boldsymbol{K}\boldsymbol{K}^{\top} + \log \boldsymbol{M}^{-})}_{\text{denote as } \boldsymbol{B}})^{-1}\boldsymbol{V} \tag{23}
\]

To normalize, we simply replace \(\exp\) with \(\text{softmax}\). Compared to Softmax Attention, DeltaFormer changes \(\boldsymbol{A}\boldsymbol{V}\) to \(\boldsymbol{A}(\boldsymbol{I} + \boldsymbol{B})^{-1}\boldsymbol{V}\). Note that:

\[
\boldsymbol{A}(\boldsymbol{I} + \boldsymbol{B})^{-1}\boldsymbol{V} = \boldsymbol{A}(\boldsymbol{I} - \boldsymbol{B} + \boldsymbol{B}^2 - \boldsymbol{B}^3 + \cdots)\boldsymbol{V} = \boldsymbol{A}(\boldsymbol{V} - \boldsymbol{B}\boldsymbol{V} + \boldsymbol{B}^2\boldsymbol{V} - \boldsymbol{B}^3\boldsymbol{V} + \cdots) \tag{24}
\]

So DeltaFormer effectively first computes multiple rounds of attention using \(\boldsymbol{K}, \boldsymbol{K}, \boldsymbol{V}\), superimposes the results as a new \(\boldsymbol{V}\), and then does one round of attention with \(\boldsymbol{Q}, \boldsymbol{K}\). This property makes it especially effective for multi-hop tasks (such as code). Moreover, this characteristic means DeltaFormer pairs especially well with MQA, because the \((\boldsymbol{I} + \boldsymbol{B})^{-1}\boldsymbol{V}\) part involves only \(\boldsymbol{K}, \boldsymbol{V}\), and for MQA, \(\boldsymbol{K}, \boldsymbol{V}\) have only a single head, so computation is much reduced compared to MHA.

However, in my view, this fixed-coefficient superposition is likely a "no free lunch" situation. For instance, my experimental results show that DeltaFormer's language model loss does not change much, which means that if the loss on some tasks decreases noticeably, the loss on other tasks must increase.

## Hardcore Position Encoding

Another noteworthy "feeding back" work is PaTH Attention, from [*PaTH Attention: Position Encoding via Accumulating Householder Transformations*](https://arxiv.org/abs/2505.16381). It feeds DeltaNet back into Softmax Attention from the perspective of position encoding.

In [*Transformer Upgrade Path 6: Completeness Analysis of Rotary Position Embedding*](https://kexue.fm/archives/9403), we showed that for any orthogonal matrix \(\boldsymbol{\Omega}\), \(\boldsymbol{R}_m = \boldsymbol{\Omega}^m\) is a generalized RoPE. Besides rotation matrices, what other orthogonal matrices are easy to construct? PaTH uses [Householder matrices](https://en.wikipedia.org/wiki/Householder_transformation): if \(\boldsymbol{w}\) is any column vector with norm \(\sqrt{2}\), then \(\boldsymbol{I} - \boldsymbol{w}\boldsymbol{w}^{\top}\) is an orthogonal matrix. We derived this in [*The Orthogonal Matrix That Transforms One Unit Vector to Another*](https://kexue.fm/archives/8453) as well; its geometric meaning is mirror reflection.

It is easy to see that this is the same as the \(\boldsymbol{I} - \boldsymbol{k}_t \boldsymbol{k}_t^{\top}\) multiplied by \(\boldsymbol{S}_{t-1}\) in DeltaNet. So PaTH simply borrows this part directly — abandoning the \(\boldsymbol{\Omega}^m\) form and the constraint \(\lVert\boldsymbol{w}\rVert = \sqrt{2}\), instead using a chain of \(\boldsymbol{I} - \boldsymbol{w}\boldsymbol{w}^{\top}\) products to express positional information:

\[
\boldsymbol{q}_i^{\top}\boldsymbol{k}_j \;\to\; \boldsymbol{q}_i^{\top}\underbrace{(\boldsymbol{I} - \boldsymbol{w}_i \boldsymbol{w}_i^{\top})(\boldsymbol{I} - \boldsymbol{w}_{i-1}\boldsymbol{w}_{i-1}^{\top})\cdots(\boldsymbol{I} - \boldsymbol{w}_{j+1}\boldsymbol{w}_{j+1}^{\top})}_{\text{denote as } \boldsymbol{R}_{i,j}}\boldsymbol{k}_j \tag{25}
\]

Writing \(\boldsymbol{R}_{i,j}\) in recursive form: \(\boldsymbol{R}_{i,j} = (\boldsymbol{I} - \boldsymbol{w}_i \boldsymbol{w}_i^{\top})\boldsymbol{R}_{i-1,j}\), with \(\boldsymbol{R}_{j,j} = \boldsymbol{I}\). Comparing with DeltaNet's equation (13), the above amounts to setting \(\boldsymbol{v}_t\) identically to zero but with a non-zero initial value \(\boldsymbol{S}_0\). Using the same process from the "Inversion and Generalizations" section, we can obtain:

\[
\boldsymbol{R}_{i,j} = \boldsymbol{I} - \boldsymbol{W}_{[j:i]}^{\top}(\boldsymbol{I} + \boldsymbol{W}_{[j:i]}\boldsymbol{W}_{[j:i]}^{\top} \odot \boldsymbol{M}^{-})^{-1}\boldsymbol{W}_{[j:i]} \tag{26}
\]

where \(\boldsymbol{W} = [\boldsymbol{w}_1, \boldsymbol{w}_2, \cdots, \boldsymbol{w}_n]^{\top}\), slicing follows NumPy conventions (e.g., \(\boldsymbol{W}_{[j:i]} = [\boldsymbol{w}_{j+1}, \boldsymbol{w}_{j+2}, \cdots, \boldsymbol{w}_i]^{\top}\)), and slicing takes precedence over transposition. Note that the matrix to be inverted is lower-triangular. Triangular matrices have an important property: the diagonal elements of the inverse equal the reciprocals of the original's diagonal elements (and for block-triangular matrices, diagonal blocks satisfy this property as well). Hence we can write:

\[
(\boldsymbol{I} + \boldsymbol{W}_{[j:i]}\boldsymbol{W}_{[j:i]}^{\top} \odot \boldsymbol{M}^{-})^{-1} = (\underbrace{(\boldsymbol{I} + \boldsymbol{W}\boldsymbol{W}^{\top} \odot \boldsymbol{M}^{-})^{-1}}_{\text{denote as } \boldsymbol{J}})_{[j:i, j:i]} \tag{27}
\]

The subsequent transformation may be easier to understand in component form:

\[
A_{i,j} = \boldsymbol{q}_i^{\top}\boldsymbol{R}_{i,j}\boldsymbol{k}_j = \boldsymbol{q}_i^{\top}\boldsymbol{k}_j - \boldsymbol{q}_i^{\top}\boldsymbol{W}_{[j:i]}^{\top}\boldsymbol{J}_{[j:i,j:i]}\boldsymbol{W}_{[j:i]}\boldsymbol{k}_j \tag{28}
\]

Expanding using index notation and exploiting the fact that \(\boldsymbol{J}\) is lower-triangular (so \(J_{l,r} = 0\) when \(l < r\)), and using indicator functions \(\chi\), we find that the sums over \(p\) and \(s\) respectively produce \(\boldsymbol{Q}\boldsymbol{W}^{\top}\) and \(\boldsymbol{W}\boldsymbol{K}^{\top}\), where multiplying by \(\chi_{l \leq i}\) corresponds to keeping the lower-triangular part (including the diagonal) of \(\boldsymbol{Q}\boldsymbol{W}^{\top}\), and multiplying by \(\chi_{r \geq j+1}\) corresponds to keeping the strictly lower-triangular part of \(\boldsymbol{W}\boldsymbol{K}^{\top}\).

Thus, we can write out the entire (pre-Softmax) attention matrix:

\[
\boldsymbol{A} = \boldsymbol{Q}\boldsymbol{K}^{\top} \odot \boldsymbol{M} - (\boldsymbol{Q}\boldsymbol{W}^{\top} \odot \boldsymbol{M})(\boldsymbol{I} + \boldsymbol{W}\boldsymbol{W}^{\top} \odot \boldsymbol{M}^{-})^{-1}(\boldsymbol{W}\boldsymbol{K}^{\top} \odot \boldsymbol{M}^{-}) \tag{29}
\]

Impressive, right? But it does not end there. Direct inversion has \(\mathcal{O}(n^3)\) complexity, which is obviously unacceptable, so one must further exploit the low-rank structure of \(\boldsymbol{W}\boldsymbol{W}^{\top}\) to reduce it to \(\mathcal{O}(n^2)\), then derive backpropagation, and finally implement it as an efficient Flash Attention-style kernel. These details can only be found in the original paper — the whole process is extremely hardcore throughout.

From the position encoding perspective, PaTH is a type of [*CoPE (Contextual Position Encoding)*](https://arxiv.org/abs/2405.18719): positions are not the simple indices \(1, 2, 3, \cdots\), but rather positional signals automatically generated from the context. Similarly, FoX can be viewed as a contextual version of ALiBi. Context-dependent positional information is a main characteristic of current linear attention, and may well be the primary direction through which it feeds back into Softmax Attention.

## The Joy of Simplification

Let us delve a bit deeper into PaTH — this will help us understand not only PaTH itself but also DeltaNet, as the two are highly related. In this section we examine two special cases of PaTH that illuminate the connection between PaTH and DeltaNet.

**First special case: \(\boldsymbol{W} = \boldsymbol{K}\).** Substituting into equation (29):

\[
\boldsymbol{A} = (\boldsymbol{Q}\boldsymbol{K}^{\top} \odot \boldsymbol{M})(\boldsymbol{I} - (\boldsymbol{I} + \boldsymbol{K}\boldsymbol{K}^{\top} \odot \boldsymbol{M}^{-})^{-1}(\boldsymbol{K}\boldsymbol{K}^{\top} \odot \boldsymbol{M}^{-})) = (\boldsymbol{Q}\boldsymbol{K}^{\top} \odot \boldsymbol{M})(\boldsymbol{I} + \boldsymbol{K}\boldsymbol{K}^{\top} \odot \boldsymbol{M}^{-})^{-1} \tag{30}
\]

using the identity \(\boldsymbol{I} - (\boldsymbol{I} + \boldsymbol{A})^{-1}\boldsymbol{A} = (\boldsymbol{I} + \boldsymbol{A})^{-1}\). Does this look familiar? It is precisely the attention matrix of DeltaNet! So from this special case, the difference between PaTH and DeltaFormer is: DeltaFormer uses the kernel trick to separately apply \(\exp\) to DeltaNet's \(\boldsymbol{Q}\boldsymbol{K}^{\top}\) and \(\boldsymbol{K}\boldsymbol{K}^{\top}\), while PaTH directly applies \(\exp\) to DeltaNet's attention matrix.

**Second special case: reintroduce the constraint \(\lVert\boldsymbol{w}\rVert = \sqrt{2}\).** In this case, \(\boldsymbol{I} - \boldsymbol{w}\boldsymbol{w}^{\top}\) is an orthogonal matrix. Define:

\[
\boldsymbol{R}_i \triangleq (\boldsymbol{I} - \boldsymbol{w}_i \boldsymbol{w}_i^{\top})(\boldsymbol{I} - \boldsymbol{w}_{i-1}\boldsymbol{w}_{i-1}^{\top})\cdots(\boldsymbol{I} - \boldsymbol{w}_1 \boldsymbol{w}_1^{\top}) = \boldsymbol{I} - \boldsymbol{W}_{[:i]}^{\top}(\boldsymbol{I} + \boldsymbol{W}_{[:i]}\boldsymbol{W}_{[:i]}^{\top} \odot \boldsymbol{M}^{-})^{-1}\boldsymbol{W}_{[:i]} = \boldsymbol{R}_{i,0} \tag{31}
\]

Then \(\boldsymbol{R}_{i,j} = \boldsymbol{R}_i \boldsymbol{R}_j^{\top}\). This equation means we can implement relative-position PaTH in an absolute-position manner, just like RoPE: simply multiply each \(\boldsymbol{q}_i^{\top}\) and \(\boldsymbol{k}_i^{\top}\) by \(\boldsymbol{R}_i\), then use the standard Softmax Attention implementation. What operation does multiplying by \(\boldsymbol{R}_i\) correspond to? Repeating the expansion from the previous section, we get:

\[
\boldsymbol{Q}\boldsymbol{R} = \boldsymbol{Q} - (\boldsymbol{Q}\boldsymbol{W}^{\top} \odot \boldsymbol{M})(\boldsymbol{I} + \boldsymbol{W}\boldsymbol{W}^{\top} \odot \boldsymbol{M}^{-})^{-1}\boldsymbol{W} \tag{32}
\]

Does this look familiar again? The second part is precisely \(\text{DeltaNet}(\boldsymbol{Q}, \boldsymbol{W}, \boldsymbol{W})\)! So in this case, PaTH's effect is equivalent to:

\[
\text{SoftmaxAttention}(\underbrace{\boldsymbol{Q} - \text{DeltaNet}(\boldsymbol{Q}, \boldsymbol{W}, \boldsymbol{W})}_{\tilde{\boldsymbol{Q}}}, \underbrace{\boldsymbol{K} - \text{DeltaNet}(\boldsymbol{K}, \boldsymbol{W}, \boldsymbol{W})}_{\tilde{\boldsymbol{K}}}, \boldsymbol{V}) \tag{33}
\]

In other words, using DeltaNet to add position encoding to \(\boldsymbol{Q}, \boldsymbol{K}\). Seen this way, PaTH (under the constraint \(\lVert\boldsymbol{w}\rVert = \sqrt{2}\)) amounts to a kind of intra-layer hybrid of Softmax Attention and DeltaNet. Of course, we could also abandon the preceding derivation and use the formula above even when \(\lVert\boldsymbol{w}\rVert \neq \sqrt{2}\). This would be similar to the [*Canon Layers*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5240330) approach of using convolution to add positional information to \(\boldsymbol{Q}, \boldsymbol{K}\), except here the convolution is not a short convolution but a long convolution via DeltaNet.

## An Unconventional Approach

Finally, let us look at a recent linear attention model that is also worth noting — MesaNet (along with a very similar concurrent work, [*Atlas*](https://arxiv.org/abs/2505.23735)). TTT's online learning perspective tells us that DeltaNet is essentially using SGD to optimize \(\frac{1}{2}\lVert\boldsymbol{S}\boldsymbol{k} - \boldsymbol{v}\rVert^2\). On closer inspection, \(\boldsymbol{S}\boldsymbol{k}\) is just a linear function of \(\boldsymbol{k}\), so this is actually a linear regression problem — and linear regression has a closed-form solution!

\[
\boldsymbol{S}_t = \boldsymbol{G}_t \boldsymbol{H}_t^{-1}, \quad \boldsymbol{G}_t = \sum_{j=1}^t \boldsymbol{v}_j \boldsymbol{k}_j^{\top}, \quad \boldsymbol{H}_t = \sum_{j=1}^t \boldsymbol{k}_j \boldsymbol{k}_j^{\top} \tag{34}
\]

MesaNet uses this closed-form solution to build a sequence model. The idea originated from [*Uncovering Mesa-Optimization Algorithms in Transformers*](https://arxiv.org/abs/2309.05858), with efficient training implemented in [*MesaNet: Sequence Modeling by Locally Optimal Test-Time Training*](https://arxiv.org/abs/2506.05233). MesaNet adds forget gates to \(\boldsymbol{G}_t, \boldsymbol{H}_t\) on top of the above formula, and adds a diagonal matrix \(\boldsymbol{\Lambda}_t\) for invertibility. The full model is:

\[
\boldsymbol{o}_t = \boldsymbol{G}_t(\boldsymbol{H}_t + \boldsymbol{\Lambda}_t)^{-1}\boldsymbol{q}_t, \quad \boldsymbol{G}_t = \gamma_t \boldsymbol{G}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^{\top}, \quad \boldsymbol{H}_t = \gamma_t \boldsymbol{H}_{t-1} + \boldsymbol{k}_t \boldsymbol{k}_t^{\top} \tag{35}
\]

Clearly, \(\boldsymbol{G}_t\) and \(\boldsymbol{H}_t\) have linear complexity in sequence length, so computing \(\boldsymbol{o}_t\) is also linear. Thus MesaNet still belongs to the linear attention family, and thanks to the closed-form solution, it can essentially guarantee better performance than DeltaNet or even Gated DeltaNet in most situations. From the signal processing perspective, MesaNet and DeltaNet correspond to [Recursive Least Squares](https://en.wikipedia.org/wiki/Recursive_least_squares_filter) and [Least Mean Squares](https://en.wikipedia.org/wiki/Least_mean_squares_filter), respectively.

Sounds like all advantages — so why do I call it "unconventional" (剑走偏锋)? In my view, MesaNet "lives by the closed-form solution and dies by the closed-form solution." The closed-form solution makes it usually better than DeltaNet, but it also gives a feeling of "this is as far as it goes," because the slightest change almost eliminates any chance of obtaining a closed-form solution. Throughout the history of mathematics, virtually every branch that depended on closed-form solutions has declined, because closed-form solutions are simply too rare and unrepresentative.

From an implementation standpoint, the matrix \(\boldsymbol{H}_t + \boldsymbol{\Lambda}_t\) to be inverted is not triangular. Although \((\boldsymbol{H}_t + \boldsymbol{\Lambda}_t)^{-1}\boldsymbol{q}_t\) can still be obtained by solving a system rather than explicitly inverting, the non-triangular structure significantly increases solving complexity. How to efficiently compute all \((\boldsymbol{H}_t + \boldsymbol{\Lambda}_t)^{-1}\boldsymbol{q}_t\) in parallel will be a long-standing challenge for MesaNet. The current paper uses the [conjugate gradient method](https://en.wikipedia.org/wiki/Conjugate_gradient_method) for approximate solutions — it works, but is not perfect.

From a theoretical capability standpoint, MesaNet is also not strictly superior to DeltaNet. This is because the update rules for MesaNet's \(\boldsymbol{G}_t, \boldsymbol{H}_t\) are still simple moving averages, and the inversion does not involve interaction between tokens, so its capability ceiling is probably below that of DeltaNet with its Delta Rule. The intuitive understanding is: MesaNet tries to remember all \(\boldsymbol{k}, \boldsymbol{v}\) — "wanting it all" may lead to blurry memories — while DeltaNet's principle is "out with the old, in with the new," and because of the "out with the old" step, it can achieve long-term, precise memorization of certain content.

We can also understand this non-optimality from a special example: all attention mechanisms discussed so far except MesaNet allow the choice \(\boldsymbol{K} = \boldsymbol{V}\) — "allow" meaning not necessarily optimal, but at least capable of producing non-trivial results. However, MesaNet cannot do this, because if \(\boldsymbol{K} = \boldsymbol{V}\), MesaNet's \(\boldsymbol{S}_t\) becomes identically the identity matrix.

Overall, MesaNet is an aesthetically pleasing model, but the closed-form solution also increases its complexity and limits its flexibility, leaving much space still to be explored. Readers who want to learn more about building sequence models based on linear regression can also read [*TTR*](https://arxiv.org/abs/2501.12352), which provides a detailed discussion of sequence models under various linear regression objectives.

## A Path Still Unfolding

This article has briefly surveyed the development trajectory of linear attention and introduced the mathematical principles behind some of the models. Linear attention started by imitating Softmax Attention, gradually developed its own distinctive character, and has now become a highly competitive approach to sequence modeling — one that has even turned around to provide new ideas for the development of Softmax Attention. This process itself is full of both fascination and insight.

<hr class="section-divider">

*Citation: Su, J. (2025, June 20). 线性注意力简史：从模仿、创新到反哺 [A Brief History of Linear Attention: From Imitation and Innovation to Feeding Back]. Scientific Spaces. [https://kexue.fm/archives/11033](https://kexue.fm/archives/11033)*

*Original content licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). This translation is shared under the same license.*
