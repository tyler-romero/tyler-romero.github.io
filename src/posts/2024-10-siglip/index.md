---
title: SigLIP
subtitle: The spiritual successor to CLIP
date: 2024-10-24T00:00:00-08:00
blurb: Covering DPO, a recently-proposed alternative to RLHF for preference tuning.
tags: ["post", "siglip", "clip", "contrastive learning", "computer vision", "cv"]
---

One major lesson that has come out of the last decade of machine learning research is that models and techniques that harness compute resources efficiently tend to outcompete their counterparts. Additionally, (deceptively) simple and scalable semi-supervised training processes such as next word prediction and contrastive language-image pretraining (CLIP) complement efficient models and techniques to enable processing vast quantitis of data, imbuing the resulting models with powerful and interesting properties.

When CLIP came out, it represented an advance in both of these areas. Contrastive language-image pretraining is conceptually simple and endlessly scalable due to the vast quantities of image-text pairs available on the internet. It was also formulated with compute scalability in mind.

However CLIP models have recently been surplated by SigLIP (Sigmoid Loss for Language Image Pre-Training) models. SigLIP models outperform CLIP models in fair comparisons on downstream benchmarks such as ImageNet and COCO. Plus, CLIP models used to be the go-to[^vlms-with-clip] choise of vision encoder for Vision-Language Models (VLMs) but recently-published state-of-the-art VLMs have been using SigLIP instead.

[^vlms-with-clip]: llava, ...

I love the [SigLIP paper](https://arxiv.org/pdf/2303.15343) for its clarity of explanation and of thought. In my opinion, its a better introduction to CLIP than the actual CLIP paper, and the authors do a great job of explaining the practical details of efficiently computing the SigLIP loss. This is a great paper to read if you are new to machine learning research and I highly recommend it to experienced practitioners as well.

In this post I am going to cover what I think are the most interesting parts of the paper, and in a follow up post I'll present some missing experimental results comparing CLIP and SigLIP models on downstream tasks.

## 50,000 foot overview of CLIP
The CLIP recipie pretrains an image encoder by contrasting image-text pairs, such as are available on image captions across the internet. A text embedding is generated for the caption by feeding it through a text encoder and an embedding is generated for the image by feeding it through an image encoder. The text encoder and image encoder are trained via a contrastive loss function that encourages embeddings from matching image-text pairs (i.e. image-text pairs that are actually found together in the wild) to be nearby in embedding space while simultamiously pushing the embeddings of unrelated image-text pairs apart in embedding space.

CLIP's loss function is batch-level. A batch of $N$ image-text pairs is sampled, and so within a batch, there are $N$ matching (positive) examples and $N^2-N$ dissimilar (negative) examples.

$$
\mathcal{L}_{clip}=-\frac{1}{2|\mathcal{B}|}\sum^{|\mathcal{B}|}_{i=1}{\left(\log{\frac{e^{tx_iy_i}}{\sum^{|\mathcal{B}|}_{j=1}e^{tx_iy_j}}+\log{\frac{e^{tx_iy_i}}{\sum^{|\mathcal{B}|}_{j=1}e^{tx_jy_i}}}}\right)}
$$

Also notice that there are two softmax functions being applied. One is the image-to-text softmax and one is the text-to-image softmax. The softmax function is applied twice to separately normalize the pairwise similarity for texts and images.

Each softmax operation is an operation across the full training batch, since computing a softmax requires computing a global normalization factor. This makes distributing this loss calculation across multiple GPUs less efficient, because the multiple cross-GPU communication operations are required for each batch.

## SigLIP

### Formulation
SigLIP addresses these inefficiencies by


SigLIP addresses these inefficiencies by simply declining to compute a global softmax and instead processing every image-text pair separately using a sigmoid-based loss.

<!--TODO: talk about how we're now in a binary classification setting w/ binary classification labels.-->

$$
\mathcal{L}_{siglip}=-\frac{1}{|\mathcal{B}|}\sum^{|\mathcal{B}|}_{i=1}\sum^{|\mathcal{B}|}_{j=1}{\log\frac{1}{1 + e^{z_{ij}(-tx_iy_j+b)}}}=-\frac{1}{|\mathcal{B}|}\sum^{|\mathcal{B}|}_{i=1}\sum^{|\mathcal{B}|}_{j=1}{\mathcal{L}_{ij}}
$$

Since this loss is example-level instead of batch-level, every example's loss can be computed independently.
$$
\mathcal{L}_{ij}=\log\frac{1}{1 + e^{z_{ij}(-tx_iy_j+b)}}
$$

Since we are now in a binary-classification setting, we need to reckon with the heavy class imbalance we are now faced with. As with CLIP, within a batch, there are $N$ positive examples and $N^2-N$ negative examples. The many negative examples dominate the loss at initialization.

### Efficient Implementation
![siglip efficient](/assets/img/siglip-efficient.png)


## Does SigLIP inherit CLIP's issues?
1. CLIP models are blind
2. CLIP models create image- and text- embedding clouds that are non-overlapping.