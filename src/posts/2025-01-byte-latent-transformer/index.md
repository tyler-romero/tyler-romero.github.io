---
title: Byte Latent Transformer Explained
subtitle: No more pesky tokens
date: 2025-01-07T00:00:00-08:00
blurb: Covering DPO, a recently-proposed alternative to RLHF for preference tuning.
tags: ["post", "machine-learning", "nlp", "language-models", "tokenization", "byte-latent-transformer"]
---

In November and December 2024, Meta AI published a series of papers that each individually feel like big steps forward in the development of Language Models:
1. [Memory Layers at Scale](https://ai.meta.com/research/publications/memory-layers-at-scale/) - a new way to trade off between parameter count and computation budget in large language models that improves over Mixture-of-Experts.
2. [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769) - a very natural extension to contemporary LLMs that avoids converting to language space during portions of the auto-regressive generation process. This facilitates more advanced reasoning via a process that is akin to a breadth-first search.
3. [Byte Latent Transformer: Patches Scale Better than Tokens](https://ai.meta.com/research/publications/byte-latent-transformer-patches-scale-better-than-tokens/) - a new byte-level LLM architecture that matches tokenization-based LLM performance with improvements to inference efficiency. Tokens are replaced with dynamically sized patches of bytes.

I want to get around to discussing all three of these papers, but I am going to lead off by discussing the third - the Byte Latent Transformer.

## Contemporary tokenization - Byte Pair Encoding
One trend in deep learning research has been to move away from heuristics and towards end-to-end solutions that learn the best way to represent data. This paper is a great example of that trend. Modern LLMs run on tokenized data, but the tokenization process is a heuristic that can be improved upon. In particular, many LLMs use [Byte-Pair Encoding (BPE)](http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM) in order to group bytes into tokens.

Byte-Pair Encoding was originally developed as a simple general-purpose data compression algorithm, but it has been [adapted for use in NLP](https://aclanthology.org/P16-1162/) as a way to create a vocabulary of sub-word tokens that can be used to represent arbitrary text. BPE has several properties that make it a strong heuristic for tokenization.
1. Since BPE works on sub-word tokens, it can be used to represent any word, even rare or unseen words, by combining multiple sub-word tokens.
2. BPE is a data-driven approach. It creates a vocabulary of tokens by iteratively merging the most frequent pairs of tokens in the training data. This leads to a vocabulary that is well-adapted to the structure of actual language data - resulting in tokens that are semantically meaningful.
3. BPE balances vocabulary size between the extremes of character-level tokenization (very small vocab) and word-level tokenization (very large vocab). This is desirable because the character-level vocab is expensive at inference time (one forward pass for every character) and the word-level can't be used to represent unseen words.

Its notable that, for a given dataset used to create a BPE tokenizer, the larger the vocabulary size, the larger the average token size. With a larger average token size, fewer tokens are needed to represent a given input text. This is a good thing (at the expense of a larger vocab embedding table) because it means that the model can consume more input text per pass during training, and the model can generate more output text per pass during inference.

## Desiring more

Can we do better than Byte-Pair Encoding though? It is possible that a scheme that is learned end-to-end could do better.

There are some desiderata that we might want to consider:
1. Auto-regressively generating individual tokens is too expensive, we want to be able to generate multiple tokens via one forward pass.
2. Not all sets of characters are equally challenging for the model to predict. Some are easily predictable given the preceding text (e.g. repetitive text). We want a technique that can allocate compute efficiently - so it is not wasted predicting a sequence of obvious characters.
3. We want a scheme that is model-aware. That is, it combines bytes in a way that is dependent on the model we are working with, in addition to the underlying training data.
4. We want the byte-ingestion scheme to naturally extend to other data modalities (e.g. visual, auditory, etc.) (NOTE not satisfied)

## Beyond Tokens
As a brief reminder, ASCII characters can be encoded with 1 byte. Most latin-script characters can be represented in 2 bytes with UTF-8. And all characters in UTF-8 can be encoded with 4 bytes or less. can be encoded with 2 b In most computing contexts, 1 byte corresponds to 1 character.

Now, a bit of new vocabulary. A *token* refers to a sequence of bytes drawn from a fixed vocabulary that is determined before training. In contrast, a *patch* refers to a dynamically grouped sequence of bytes with no fixed vocabulary.

The key difference is that tokens are treated as atomic units of input and output for an LLM, while patches serve as computational units but preserve bytes as the atomic units of input and output.

Now - the first question that arises is how do we group tokens into patches? Several strategies are possible:
1. **k-strided**: Group bytes into fixed-size patches of size k. For example, MegaByte used k=4.
2. **space-delimited**: Group bytes between spaces, effectively treating words as computational units.
3. **byte-pair encoding**: Apply the same BPE process used for tokenization to instead generate patches.
4. **entropy-based**: Group bytes into sequences until encountering a highly uncertain next byte. **This is the approach that the BLT paper proposes.** This data- and model-aware approach aims to create patches of similar computational complexity.

### Entropy Patching

The entropy-based approach is intuitively appealing. Entropy ($H$) measures the average amount of "surprise" or uncertainty in a random variable $X$. When all outcomes are equally likely, entropy is maximized - there's maximum uncertainty about what will occur. When one outcome is very likely and others unlikely, entropy is low - we're more certain about what will happen.

For example, consider predicting the next character in these two scenarios:
1. After "The quick brown f", the next character is very likely to be "o" - low entropy
2. At the start of a new sentence, many characters are possible - high entropy

The formula below captures this mathematically - for each possible outcome $x_i$, we multiply its probability $p(x_i)$ by the log of its probability. More likely outcomes contribute less to the entropy (as log p is less negative), while unlikely outcomes contribute more:

$$
H(X) = \mathbb{E}[-\log p(X)]=-\sum_i p(x_i)\log p(x_i)
$$

We can use a byte-level language model (where $V$ is the vocabulary of all 256 possible values of a byte) to estimate the probability distribution $p_e$ over the next byte, conditioned on all previous bytes:
$$
H(x_i) = \sum_{v \in V} p_e(x_i=v|x_{<i}) \log p_e(x_i=v|x_{<i})
$$

<!-- TODO: discuss receptive field and lookup table -->

<!-- TODO: nice interactive viz of byte-level entropies -->

<!-- NOTE: Why not do this same sort of strategy with tokens? Why not try to aggregate the entropy across bytes to actually get a similar amount of info in each patch? -->

BLT uses a small (100M parameter) byte-level language model with sliding-window attention of 512 bytes to estimate byte-wise entropy. This entropy model is trained ahead-of-time and frozen, so during BLT training it simply provides entropy estimates for each byte.

Using these per-byte entropy estimates and a chosen threshold, we can segment a byte sequence by ending patches whenever a byte's entropy exceeds the threshold.

<!-- Debug: Print the data -->
<pre>
{{ entropy-values.values | json }}
</pre>

<!-- Debug: Start of entropy viz -->
<div
  class="entropy-viz"
  data-entropy-viz
  data-entropy-data='[
    {"char": "T", "entropy": 0.8},
    {"char": "h", "entropy": 0.3},
    {"char": "e", "entropy": 0.2},
    {"char": " ", "entropy": 0.9},
    {"char": "q", "entropy": 0.7},
    {"char": "u", "entropy": 0.4},
    {"char": "i", "entropy": 0.3},
    {"char": "c", "entropy": 0.4},
    {"char": "k", "entropy": 0.3},
    {"char": " ", "entropy": 0.8},
    {"char": "b", "entropy": 0.6},
    {"char": "r", "entropy": 0.4},
    {"char": "o", "entropy": 0.3},
    {"char": "w", "entropy": 0.4},
    {"char": "n", "entropy": 0.3},
    {"char": " ", "entropy": 0.8},
    {"char": "f", "entropy": 0.5},
    {"char": "o", "entropy": 0.2},
    {"char": "x", "entropy": 0.3},
    {"char": " ", "entropy": 0.9},
    {"char": "j", "entropy": 0.7},
    {"char": "u", "entropy": 0.4},
    {"char": "m", "entropy": 0.3},
    {"char": "p", "entropy": 0.4},
    {"char": "e", "entropy": 0.3},
    {"char": "d", "entropy": 0.4},
    {"char": " ", "entropy": 0.8},
    {"char": "o", "entropy": 0.6},
    {"char": "v", "entropy": 0.4},
    {"char": "e", "entropy": 0.3},
    {"char": "r", "entropy": 0.4},
    {"char": " ", "entropy": 0.8},
    {"char": "t", "entropy": 0.6},
    {"char": "h", "entropy": 0.3},
    {"char": "e", "entropy": 0.2},
    {"char": " ", "entropy": 0.9}
  ]'
></div>
<!-- Debug: End of entropy viz -->