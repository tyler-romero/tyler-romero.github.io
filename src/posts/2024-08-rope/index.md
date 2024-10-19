---
title: Diving into the math behind Rotary Position Embedding (RoPE) for LLMs and ViTs
subtitle: Imbuing modern LLMs with position information.
date: 2024-08-01T00:00:00-08:00
blurb: Covering how modern LLMs are imbued with position information.
tags: post
---

Rotary Position Embedding (RoPE) is currently the dominant scheme for encoding positional information in transformers.

To understand why positional encoding is necessary, it's helpful to first consider transformers as operating on sets. At their core, transformers process a set of tokens, each represented by a vector. The self-attention mechanism allows each token to attend to all other tokens in the set, regardless of their original order. Thus, a transformer without some sort of position encoding could not differentiate between "The cat chased the mouse" and "The mouse chased the cat".

## A primer on position encoding

### Starting with "absolute" position encodings
The original transformer from "Attention is All You Need" used sine and cosine functions of different frequencies to construct position embeddings, which are summed with token embeddings to create the inputs to the transformer. Those embeddings are defined as a set of sine and cosine functions with periods ranging from $2\pi$ to $2\pi * 10000$:
$$
PE_{pos} = \begin{bmatrix}
sin(pos) \\
cos(pos) \\
sin(pos/10000^{2/d_{model}}) \\
cos(pos/10000^{2/d_{model}}) \\
sin(pos/10000^{4/d_{model}}) \\
cos(pos/10000^{4/d_{model}}) \\
\vdots \\
sin(pos/10000) \\
cos(pos/10000) \\
\end{bmatrix}
$$

If we reorder the indices such that the first half of the embedding is made up of the `sine` functions and the second half is made up of the `cosine` functions, then we can better visualize a position embedding based on the position within the sequence it represents:

{% include "sinusoidal-visualization.html" %}


A range of frequencies is used to encode position information, the authors hypothesizing that this formulation would make it easy for the model to learn to attend by relative positions, since "for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$."

The authors of "Attention is All You Need" also experimented with learned position embeddings - that is, randomly initializing a learnable embedding table indexed by position. These learnable embeddings were summed with their corresponding token embedding and then trained via backprop. Learned position embeddings later rose to become the dominant paradigm when they were adopted by works such as BERT, GPT-2, and RoBERTa. In many cases, learned position embeddings simply outperformed sinusoidal position embeddings.

ViTs also adopted learned position embeddings....

Learned position embeddings come with several key downsides as a result of being based on a fixed embedding table:
1. Limited sequence length and lack of extrapolation
2. Increased parameter count

Both sinusoidal position embeddings and learned positional embeddings can be considered "absolute" position encodings - they directly model the position information of each input token.

### Now, "relative" position encodings
An orthogonal approach would be "relative" position encodings, which instead leverage the relative distance between two tokens when calculating attention.

Relative position encodings were introduced in [Self-Attention with Relative Position Representations](https://arxiv.org/pdf/1803.02155) by Shaw et al. Their approach modified the self-attention mechanism to include terms that change the key and value vectors based on the relative position of the query and key. The distance (in terms of number of tokens) between the query and key is used to index into a learned embedding table, which is then added to the key and value vectors.
<!-- TODO: Explain the attention mechanism in a bit more detail? -->

| Model              | Position Encoding                    | EN-DE BLEU | EN-FR BLEU |
|:-------------------|:-------------------------------------|:----------:|:----------:|
| Transformer (base) | Absolute Position Representations    |    26.5    |    38.2    |
| Transformer (base) | Relative Position Representations    |    **26.8**    |    **38.7**    |
| Transformer (big)  | Absolute Position Representations    |    27.9    |    41.2    |
| Transformer (big)  | Relative Position Representations    |    **29.2**    |    **41.5**    |
<span style="font-size: small;">**Table 1:** Experimental results from Shaw et al. for English-to-German and English-to-French transation tasks.</span>

Relative position encodings have been shown to outperform absolute position encodings in many settings, as shown in Table 1. However, they come with their own set of challenges, such as the need for additional parameters.
<!-- TODO: expand upon clipping long range dependencies and storing large numbers of embeddings -->

Since Shaw et al., many works have explored alternative ways to incorporate relative position information into transformers.
<!-- TODO: expand upon clipping long range dependencies and storing large numbers of embeddings -->

## Enter RoPE
Rotary Positional Embedding (RoPE) was introduced in Su et al. as a novel way to encode relative position information in transformers.