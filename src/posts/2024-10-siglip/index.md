---
title: "SigLIP: The spiritual successor to CLIP"
subtitle: Computational efficiency is important!
date: 2024-10-24T00:00:00-08:00
blurb: Covering DPO, a recently-proposed alternative to RLHF for preference tuning.
tags: ["post", "siglip", "clip", "contrastive learning", "computer vision", "cv"]
---

One major lesson that has come out of the last decade of machine learning research is that models and techniques that harness compute resources efficiently tend to outcompete their counterparts. Additionally, (deceptively) simple and scalable semi-supervised training processes such as next word prediction and contrastive language-image pretraining (CLIP) complement efficient models and techniques to enable processing vast quantitis of data, imbuing the resulting models with powerful and interesting properties.

When CLIP came out, it represented an advance in both of these areas. Contrastive language-image pretraining is conceptually simple and endlessly scalable due to the vast quantities of image-text pairs available on the internet. It was also formulated with compute scalability in mind.

However CLIP models have recently been surplated by SigLIP (Sigmoid Loss for Language Image Pre-Training) models. SigLIP models outperform CLIP models in fair comparisons on downstream benchmarks such as ImageNet and COCO. Plus, CLIP models used to be the go-to[^vlms-with-clip] choise of vision encoder for Vision-Language Models (VLMs) but recently-published state-of-the-art VLMs have been using SigLIP instead.

![LLaVA style VLM architecture](/assets/img/vlm-arch.png)

[^vlms-with-clip]: LLaVA, ... and ... all use CLIP as their vision encoder.

I love the [SigLIP paper](https://arxiv.org/pdf/2303.15343) for its clarity of explanation and of thought. In my opinion, its a better introduction to CLIP than the actual CLIP paper, and the authors do a great job of explaining the practical details of efficiently computing the SigLIP loss. This is a great paper to read if you are new to machine learning research and I highly recommend it to experienced practitioners as well.

In this post I am going to cover what I think are the most interesting parts of the paper, and in a follow up post I'll present some missing experimental results comparing CLIP and SigLIP models on downstream tasks.

## 50,000 foot overview of CLIP
The CLIP recipie pretrains an image encoder by contrasting image-text pairs, such as are available on image captions across the internet. A text embedding is generated for the caption by feeding it through a text encoder and an embedding is generated for the image by feeding it through an image encoder. The text encoder and image encoder are trained via a contrastive loss function that encourages embeddings from matching image-text pairs (i.e. image-text pairs that are actually found together in the wild) to be nearby in embedding space while simultamiously pushing the embeddings of unrelated image-text pairs apart in embedding space.

CLIP's loss function is batch-level. A batch of $N$ image-text pairs is sampled, and so within a batch, there are $N$ matching (positive) examples and $N^2-N$ dissimilar (negative) examples.

$$
\mathcal{L}_{clip}
=-\frac{1}{2|\mathcal{B}|}\sum^{|\mathcal{B}|}_{i=1}{\left(\log{\frac{e^{tx_iy_i}}{\sum^{|\mathcal{B}|}_{j=1}e^{tx_iy_j}}+\log{\frac{e^{tx_iy_i}}{\sum^{|\mathcal{B}|}_{j=1}e^{tx_jy_i}}}}\right)}
$$

Notice that there are two softmax functions being applied. One is the image-to-text softmax and one is the text-to-image softmax. The softmax function is applied twice to separately normalize the pairwise similarity for texts and images. $t$ (i.e. temperature) is a learnable scalar parameter. Pairwise similarities are computed via a dot product ($x_iy_j$).

Each softmax operation is an operation across the full training batch, since computing a softmax requires computing a global normalization factor. This makes distributing this loss calculation across multiple GPUs less efficient, because the multiple cross-GPU communication operations are required for each batch.

<!--TODO: Discuss how CLIP loss is computed in practical/device terms -->

## 10,000 foot overview of SigLIP

### Formulation
SigLIP addresses these inefficiencies by switching from a global softmax-based loss to an image-text-pair-wise sigmoid-based loss. This change allows us to view our contrastive setting as a standard binary-classification task, where matching pairs are assigned positive labels and all other (non-matching) pairs are assigned negative labels.

<!--TODO: Correct dot products between x_i and y_j. Bold to make them appear as vectors -->

$$
\mathcal{L}_{siglip}
=-\frac{1}{|\mathcal{B}|}\sum^{|\mathcal{B}|}_{i=1}\sum^{|\mathcal{B}|}_{j=1}{\log\frac{1}{1 + e^{z_{ij}(-tx_iy_j+b)}}}
=-\frac{1}{|\mathcal{B}|}\sum^{|\mathcal{B}|}_{i=1}\sum^{|\mathcal{B}|}_{j=1}{\log\sigma\left(-z_{ij}(tx_iy_j-b)\right)}  \\[10pt]
=-\frac{1}{|\mathcal{B}|}\sum^{|\mathcal{B}|}_{i=1}\sum^{|\mathcal{B}|}_{j=1}{\mathcal{L}_{ij}}
$$

Where $z_{ij}$ is the label for a given image and text comparison, set to $1$ if the image and text are paired and $-1$ otherwise.

As we are now in a binary-classification setting, we need to reckon with the heavy class imbalance we are now faced with. As with CLIP, within a batch, there are $N$ positive examples and $N^2-N$ negative examples. The authors note that the many negative examples dominate the loss at initialization, and they propose to compensate for this by adding a learnable bias parameter $b$. They then initialize $b=-10$ and $t=10$. If we substitue the initial values of those parameters into our loss function, we see:
$$
\mathcal{L}_{siglip}
=-\frac{1}{|\mathcal{B}|}\sum^{|\mathcal{B}|}_{i=1}\sum^{|\mathcal{B}|}_{j=1}{\log\sigma\left(-z_{ij}(10x_iy_j+10)\right)}
$$

### Efficient computation of SigLIP's loss

Since SigLIP's loss is example-level instead of batch-level, the loss for each pair can be computed independently.
$$
\mathcal{L}_{ij}
=\log\sigma\left(-z_{ij}(tx_iy_j-b)\right)
$$

We can use this fact to implement an efficient method of computing SigLIP's loss with minimal memory requirements and minimal inter-device communication.

This lovely figure from the SigLIP paper gives a good sense for the process in a setting with three devices (i.e. GPUs, TPUs, etc) and a global "batch size" of 12.

<!-- <figure class="fullwidth"> -->
![siglip efficient](/assets/img/siglip-efficient.png)
<!-- </figure> -->

In essence, each device begins  with an equal number of text and image examples. (_left_) The loss computation proceeds by computing the loss for every possible image-text pair present locally on a given device. Those losses are summed per-device and stored locally. (_middle_) Then, the *text* examples are swapped between devices in a as if along a ring[^ring] (not images, since more bytes would need to be communicated). We can again compute the loss for every image-text pair present on the same device. These losses can be summed with the running per-device loss computed in the first step. (_right_) This proceedure proceeds until all image-text pairs within a batch have been processed. Finally, a single cross-device summation is required to compute the final value of the loss for the batch.

The per-device memory requirements here scale with the square of the local batch size ($b=\frac{|B|}{d}$). Which means that this training proceedure can be scaled to arbitrarily large batch sizes by increasing the number of devices.

[^ring]: (device 1) -> (device 2); (device 2) -> (device 3); (device 3) -> (device 1)


## Does SigLIP inherit CLIP's issues?
Recently, several works have pointed out issues with the underlying vision capabilities of CLIP models.
1. CLIP models are blind
2. CLIP models create image- and text- embedding clouds that are non-overlapping.

## Interested in learning more?
I recommend reading the CLIP and SigLIP papers. They are both excelent.

## References