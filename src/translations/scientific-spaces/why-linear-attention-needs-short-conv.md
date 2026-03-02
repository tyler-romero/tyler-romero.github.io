---
title: "Why Does Linear Attention Need Short Conv?"
subtitle: "Translated from [为什么线性注意力要加Short Conv？](https://kexue.fm/archives/11320) by Jianlin Su (苏剑林)"
date: 2025-10-05T00:00:00+08:00
blurb: "Short convolutions on K in linear attention transform the TTT training objective from trivial self-prediction to next-token prediction, enabling meaningful memorization of the KV cache."
tags: ["translation", "linear-attention", "rnn", "attention"]
math: true
---

*Translator's note (Opus 4.6): This is an English translation of [为什么线性注意力要加Short Conv？](https://kexue.fm/archives/11320) by Jianlin Su (苏剑林), originally published on October 5, 2025 on [Scientific Spaces (科学空间)](https://kexue.fm). The translation preserves the author's first-person voice.*

<hr class="section-divider">

If you have been following developments in model architecture, you will have noticed that the newer linear attention models (see [*A Brief History of Linear Attention: From Imitation and Innovation to Feeding Back*](https://kexue.fm/archives/11033)) all add a short convolution to \(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}\). For example, here is the architecture of [*DeltaNet*](https://arxiv.org/abs/2406.06484):

![Short Conv in DeltaNet](/assets/img/deltanet-short-conv.png)

Why add this short convolution? An intuitive explanation might be that it increases model depth, enhances the model's token-mixing ability, and so on — in plain terms, it compensates for the drop in expressiveness caused by linearization. This explanation is roughly correct, but it is a "one-size-fits-all template" answer. We would like a more precise understanding of the mechanism by which it works.

In what follows, I will present my own understanding (or more accurately, my conjecture).

## Test-Time Training

From [*A Brief History of Linear Attention: From Imitation and Innovation to Feeding Back*](https://kexue.fm/archives/11033), we know that the core idea behind current new-style linear attention is [*TTT (Test-Time Training)*](https://arxiv.org/abs/2407.04620), or equivalently, online learning. TTT exploits the similarity between optimizer updates and RNN iterations, using optimizers to construct (not necessarily linear) RNN models. Models such as DeltaNet, GDN, and Comba can all be seen as special cases of this framework.

Specifically, TTT treats \(\boldsymbol{K}, \boldsymbol{V}\) as paired training data \((\boldsymbol{k}_1, \boldsymbol{v}_1), (\boldsymbol{k}_2, \boldsymbol{v}_2), \cdots, (\boldsymbol{k}_t, \boldsymbol{v}_t)\). We use this data to train a model \(\boldsymbol{v} = \boldsymbol{f}(\boldsymbol{S}_t; \boldsymbol{k})\), and then output \(\boldsymbol{o}_t = \boldsymbol{f}(\boldsymbol{S}_t; \boldsymbol{q}_t)\), where \(\boldsymbol{S}_t\) denotes the model parameters, updated via SGD:

\[
\boldsymbol{S}_t = \boldsymbol{S}_{t-1} - \eta_t \nabla_{\boldsymbol{S}_{t-1}} \mathcal{L}(\boldsymbol{f}(\boldsymbol{S}_{t-1}; \boldsymbol{k}_t), \boldsymbol{v}_t) \tag{1}
\]

Of course, if we wish, we can also consider other optimizers — for example, [*Test-Time Training Done Right*](https://arxiv.org/abs/2505.23884) experimented with the Muon optimizer. Beyond the choice of optimizer, other flexible design choices include the model architecture \(\boldsymbol{v} = \boldsymbol{f}(\boldsymbol{S}_t; \boldsymbol{k})\) and the loss function \(\mathcal{L}(\boldsymbol{f}(\boldsymbol{S}_{t-1}; \boldsymbol{k}_t), \boldsymbol{v}_t)\). Additionally, we can consider chunk-based mini-batch TTT.

It is easy to imagine that TTT is theoretically highly flexible and can construct arbitrarily complex RNN models. When the architecture is a linear model \(\boldsymbol{v} = \boldsymbol{S}_t \boldsymbol{k}\) and the loss function is squared error, the result corresponds to DeltaNet; if we add some regularization terms, we can derive variants such as GDN.

## The Hard Question

The reason for presenting TTT first is to make clear that the underlying logic of mainstream linear attention today is the same as TTT: at its core, it is online learning over the data pairs \((\boldsymbol{k}_1, \boldsymbol{v}_1), (\boldsymbol{k}_2, \boldsymbol{v}_2), \cdots, (\boldsymbol{k}_t, \boldsymbol{v}_t)\). This naturally raises a question: why do it this way? What does this actually learn?

To answer this, we first need to reflect on what we actually want. Following the characteristics of softmax attention, what we want is to compute an \(\boldsymbol{o}_t\) from \((\boldsymbol{k}_1, \boldsymbol{v}_1), (\boldsymbol{k}_2, \boldsymbol{v}_2), \cdots, (\boldsymbol{k}_t, \boldsymbol{v}_t)\) and \(\boldsymbol{q}_t\) — a process that should ideally depend on all the \((\boldsymbol{k}, \boldsymbol{v})\) pairs. At the same time, we want to achieve this in constant complexity, so a natural idea is to first compress the \((\boldsymbol{k}, \boldsymbol{v})\) pairs into a fixed-size state (independent of \(t\)), and then read from that state.

How do we achieve this compression? TTT's idea is: design a model \(\boldsymbol{v} = \boldsymbol{f}(\boldsymbol{S}_t; \boldsymbol{k})\), then "train" this model on the \((\boldsymbol{k}, \boldsymbol{v})\) pairs. After training, the model has in some sense "memorized" these \((\boldsymbol{k}, \boldsymbol{v})\) pairs — this is equivalent to compressing all the \((\boldsymbol{k}, \boldsymbol{v})\) pairs into the fixed-size model weights \(\boldsymbol{S}_t\). As for how \(\boldsymbol{q}_t\) uses \(\boldsymbol{S}_t\), directly substituting it into the model to get \(\boldsymbol{o}_t = \boldsymbol{f}(\boldsymbol{S}_t; \boldsymbol{q}_t)\) is a fairly natural choice, but in principle we could also design other ways to use it.

In other words, the core task of TTT is to leverage the fact that "training a model" approximately equals "memorizing the training set" in order to compress \(\boldsymbol{K}, \boldsymbol{V}\). However, the claim that "training a model" approximately equals "memorizing the training set" is not trivial — it has some preconditions.

## Keys and Values from the Same Source

Consider an example: if we set \(\boldsymbol{K} = \boldsymbol{V}\), the TTT framework theoretically breaks down. This is because the optimal solution for the model \(\boldsymbol{v} = \boldsymbol{f}(\boldsymbol{S}_t; \boldsymbol{k})\) would simply be the identity map — a trivial solution that amounts to memorizing nothing. Online-update methods like DeltaNet might still salvage the situation to some extent, while exact-solution methods like [*MesaNet*](https://arxiv.org/abs/2506.05233) would literally output the identity matrix \(\boldsymbol{I}\).

Some readers might object: why would we ever consider the unnatural choice \(\boldsymbol{K} = \boldsymbol{V}\)? Indeed, \(\boldsymbol{K} = \boldsymbol{V}\) is an extreme case — it serves here only as an example to show that "training a model" approximately equaling "memorizing the training set" does not hold unconditionally. Furthermore, we verified in [*Transformer Upgrade Path 20: What Makes MLA Good? (Part 1)*](https://kexue.fm/archives/10907) that for softmax attention, \(\boldsymbol{K} = \boldsymbol{V}\) can still produce decent results.

This tells us that \(\boldsymbol{K} = \boldsymbol{V}\) is not a fundamental obstacle for the attention mechanism, but within the TTT framework it can cause model failure. The reason is that if \(\boldsymbol{K}\) and \(\boldsymbol{V}\) completely overlap, there is nothing to learn from regressing one on the other. By analogy, the greater the information overlap between \(\boldsymbol{K}\) and \(\boldsymbol{V}\), the less there is to learn between them — in other words, the less TTT memorizes the "training set."

In standard attention mechanisms, \(\boldsymbol{q}_t, \boldsymbol{k}_t, \boldsymbol{v}_t\) are all obtained from the same input \(\boldsymbol{x}_t\) via different linear projections. In other words, \(\boldsymbol{k}_t\) and \(\boldsymbol{v}_t\) share the same source \(\boldsymbol{x}_t\), which always has the feel of "predicting yourself from yourself" — and there is limited value in that.

## Convolution to the Rescue

How do we make TTT learn something valuable even when keys and values share the same source, or even when \(\boldsymbol{K} = \boldsymbol{V}\)? The answer actually dates back a long way — to Word2Vec and even earlier — and it is simply: don't "predict yourself," instead "predict your neighbors."

Take [*Word2Vec*](https://arxiv.org/abs/1301.3781) as an example: its training method is "center word predicts context." The previously popular [*BERT*](https://arxiv.org/abs/1810.04805) uses MLM for pretraining, which masks certain words and predicts them — this can be described as "context predicts center word." Today's mainstream LLMs use NTP (Next Token Prediction), predicting the next word from the preceding context. Clearly, the common thread is that none of them predict themselves — they all predict their surroundings.

So, to improve TTT, we need to change the \((\boldsymbol{k}_t, \boldsymbol{v}_t)\) pairing pattern of "predicting yourself from yourself." Given that current LLMs primarily use NTP, we can adopt the same approach within TTT: for example, using \((\boldsymbol{k}_{t-1}, \boldsymbol{v}_t)\) to construct the training pairs, meaning we use \(\boldsymbol{k}_{t-1}\) to predict \(\boldsymbol{v}_t\). This way, even if \(\boldsymbol{K} = \boldsymbol{V}\), the model can learn non-trivial results. In this setup, the TTT inner loop and the outer LLM training loop both perform NTP — a beautifully consistent design.

However, using only \(\boldsymbol{k}_{t-1}\) to predict \(\boldsymbol{v}_t\) seems to waste \(\boldsymbol{k}_t\), so a further idea is to mix \(\boldsymbol{k}_{t-1}\) and \(\boldsymbol{k}_t\) together in some way before predicting \(\boldsymbol{v}_t\). At this point, readers may have realized: "mixing \(\boldsymbol{k}_{t-1}\) and \(\boldsymbol{k}_t\) together in some way" — isn't that just a convolution with kernel_size=2! Therefore, adding a short convolution to \(\boldsymbol{K}\) transforms TTT's training objective from "predicting yourself" to NTP, giving TTT at least the ability to learn an n-gram model.

As for adding short convolutions to \(\boldsymbol{Q}\) and \(\boldsymbol{V}\), that is entirely incidental. According to reports from the FLA community[^fla], adding them to \(\boldsymbol{Q}\) and \(\boldsymbol{V}\) helps a little, but far less than adding short conv to \(\boldsymbol{K}\) — which serves as supporting evidence for our conjecture.

[^fla]: FLA (Flash Linear Attention) is an open-source community and library for efficient linear attention implementations. The "FLA group" (飞来阁) refers to their discussion forum.

## Summary

This article presents a speculative understanding (闭门造车, literally "building a cart behind closed doors") of the question "Why does linear attention need short conv?"

<hr class="section-divider">

*Citation: Su, J. (2025, October 5). 为什么线性注意力要加Short Conv？ [Why Does Linear Attention Need Short Conv?]. Scientific Spaces. [https://kexue.fm/archives/11320](https://kexue.fm/archives/11320)*

*Original content licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). This translation is shared under the same license.*
