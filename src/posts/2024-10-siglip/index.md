---
title: "SigLIP: The spiritual successor to CLIP"
subtitle: Computational efficiency is important!
date: 2024-10-24T00:00:00-08:00
blurb: Covering DPO, a recently-proposed alternative to RLHF for preference tuning.
tags: ["post", "siglip", "clip", "contrastive learning", "computer vision", "cv"]
---

One major lesson that has come out of the last decade of machine learning research is that models and techniques that harness compute resources efficiently tend to outcompete their counterparts. Additionally, (deceptively) simple and scalable semi-supervised training processes such as next-word prediction and [contrastive language-image pretraining (CLIP)](https://arxiv.org/abs/2103.00020) complement efficient models and techniques to enable processing vast quantitis of data, imbuing the resulting models with powerful and interesting properties.

When CLIP came out, it represented an advance in both of these areas. Contrastive language-image pretraining is conceptually simple and endlessly scalable due to the vast quantities of image-text pairs available on the internet. It was also formulated with compute scalability in mind.

However CLIP models have recently been surplated by SigLIP (Sigmoid Loss for Language Image Pre-Training) models. SigLIP models outperform CLIP models in fair comparisons on downstream benchmarks such as ImageNet and COCO. Plus, CLIP models used to be the go-to choise of vision encoder for Vision-Language Models (VLMs)[^llava-arch] but many recently-published state-of-the-art VLMs have been using SigLIP instead.

[^llava-arch]: ![LLaVA style VLM architecture](/assets/img/vlm-arch.png)

| Model                                             | Vision Encoder | Date      |
|:--------------------------------------------------|:--------------|:-----------|
| [Flamingo](https://arxiv.org/abs/2204.14198)      | CLIP          | Apr 2022  |
| [LLaVA](https://arxiv.org/abs/2304.08485)         | CLIP          | Apr 2023  |
| [IDEFICS](https://arxiv.org/abs/2306.16527)       | CLIP          | June 2023 |
| [Qwen-VL](https://arxiv.org/abs/2308.12966)       | CLIP          | Aug 2023  |
| [LLaVA-1.5](https://arxiv.org/abs/2310.03744)     | CLIP          | Oct 2023  |
| [ShareGPT4V](https://arxiv.org/abs/2311.12793)    | CLIP          | Nov 2023  |
| [CogVLM](https://arxiv.org/abs/2311.03079)        | CLIP          | Nov 2023  |
| [DeepSeek-VL](https://arxiv.org/abs/2403.05525)   | SigLIP + [SAM](https://arxiv.org/abs/2304.02643) + [ViTDet](https://arxiv.org/abs/2203.16527)       | Mar 2024  |
| [IDEFICS2](https://arxiv.org/abs/2405.02246)      | SigLIP        | May 2024  |
| [PaliGemma](https://arxiv.org/abs/2407.07726)     | SigLIP        | July 2024 |
| [VILA2](https://arxiv.org/abs/2407.17453)         | SigLIP        | July 2024 |
| [IDEFICS3](https://arxiv.org/abs/2408.12637)      | SigLIP        | Aug 2024  |
| [MiniCPM-V](https://arxiv.org/abs/2408.01800)     | SigLIP        | Aug 2024  |
| [LLaVA-OneVision](https://arxiv.org/abs/2408.03326)| SigLIP       | Aug 2024  |
| [Qwen2-VL](https://arxiv.org/abs/2409.12191)      | CLIP (DFN)    | Sep 2024  |

I love the [SigLIP paper](https://arxiv.org/pdf/2303.15343) for its clarity of explanation and of thought. In my opinion, its a better introduction to CLIP than the actual CLIP paper, and the authors do a great job of explaining the practical details of efficiently computing the SigLIP loss. This is a great paper to read if you are new to machine learning research and I highly recommend it to experienced practitioners as well.

In this post I am going to cover what I think are the most interesting parts of the paper, and in a follow up post I'll present some missing experimental results comparing CLIP and SigLIP models on downstream tasks.

## 50,000 foot overview of CLIP
The CLIP recipie pretrains an image encoder by contrasting image-text pairs, such as are available on image captions across the internet. A text embedding is generated for the caption by feeding it through a text encoder and an embedding is generated for the image by feeding it through an image encoder. The text encoder and image encoder are trained via a contrastive loss function that encourages embeddings from matching image-text pairs (i.e. image-text pairs that are actually found together in the wild) to be nearby in embedding space while simultamiously pushing the embeddings of unrelated image-text pairs apart in embedding space.

CLIP's loss function is batch-level. A batch of $N$ image-text pairs is sampled, and so within a batch, there are $N$ matching (positive) examples and $N^2-N$ dissimilar (negative) examples.

$$
\mathcal{L}_{clip}
=-\frac{1}{2|\mathcal{B}|}\sum^{|\mathcal{B}|}_{i=1}{\left(\log{\frac{e^{t\mathbf{x_i}\mathbf{y_i}}}{\sum^{|\mathcal{B}|}_{j=1}e^{t\mathbf{x_i}\mathbf{y_j}}}+\log{\frac{e^{t\mathbf{x_i}\mathbf{y_i}}}{\sum^{|\mathcal{B}|}_{j=1}e^{t\mathbf{x_j}\mathbf{y_i}}}}}\right)}
$$

Notice that there are two softmax functions being applied. One is the image-to-text softmax and one is the text-to-image softmax. The softmax function is applied twice to separately normalize the pairwise similarity for texts and images. $t$ (i.e. temperature) is a learnable scalar parameter. Pairwise similarities are computed via a dot product[^dotproduct].

[^dotproduct]: A dot product measures how similar two vectors are in direction and magnitude. Mathematically, $\mathbf{v} \cdot \mathbf{w} = |\mathbf{v}||\mathbf{w}|\cos(\theta_{vw})$, where $\theta_{vw}$ is the angle between the vectors. When vectors point in similar directions, their dot product is larger. ![Geometric interpretation of a Dot Product](/assets/img/dot-product.png)

Each softmax operation is an operation across the full training batch, since computing a softmax requires computing a global normalization factor. This makes distributing this loss calculation across multiple GPUs less efficient, because the multiple cross-GPU communication operations are required for each batch.

<!--TODO: Discuss how CLIP loss is computed in practical/device terms -->

## 10,000 foot overview of SigLIP

### Formulation
SigLIP addresses these inefficiencies by switching from a global softmax-based loss to an image-text-pair-wise sigmoid-based loss. This change allows us to view our contrastive setting as a standard binary-classification task, where matching pairs are assigned positive labels and all other (non-matching) pairs are assigned negative labels.

<!--TODO: Correct dot products between x_i and y_j. Bold to make them appear as vectors -->

$$
\mathcal{L}_{siglip}
=-\frac{1}{|\mathcal{B}|}\sum^{|\mathcal{B}|}_{i=1}\sum^{|\mathcal{B}|}_{j=1}{\log\frac{1}{1 + e^{z_{ij}(-t\mathbf{x_i}\mathbf{y_j}+b)}}}
=-\frac{1}{|\mathcal{B}|}\sum^{|\mathcal{B}|}_{i=1}\sum^{|\mathcal{B}|}_{j=1}{\log\sigma\left(-z_{ij}(t\mathbf{x_i}\mathbf{y_j}-b)\right)}  \\[10pt]
=-\frac{1}{|\mathcal{B}|}\sum^{|\mathcal{B}|}_{i=1}\sum^{|\mathcal{B}|}_{j=1}{\mathcal{L}_{ij}}
$$

Where $z_{ij}$ is the label for a given image and text comparison, set to $1$ if the image and text are paired and $-1$ otherwise.

As we are now in a binary-classification setting, we need to reckon with the heavy class imbalance we are now faced with. As with CLIP, within a batch, there are $N$ positive examples and $N^2-N$ negative examples. The authors note that the many negative examples dominate the loss at initialization, and they propose to compensate for this by adding a learnable bias parameter $b$. They then initialize $b=-10$ and $t=10$. If we substitue the initial values of those parameters into our loss function, we see:
$$
\mathcal{L}_{siglip}
=-\frac{1}{|\mathcal{B}|}\sum^{|\mathcal{B}|}_{i=1}\sum^{|\mathcal{B}|}_{j=1}{\log\sigma\left(-z_{ij}(10\mathbf{x_i}\mathbf{y_j}+10)\right)}
$$

The logistic function is an S-shaped (or sigmoid) function commonly denoted using $\sigma(x)$. It frequently appears when working with probabilities because it can "squash" values in $\mathbb{R}$ (the set of all real numbers) into $(0, 1)$ (the set of probabilities values, excluding exactly 0 or 1). ![Sigmoid Function](/assets/img/sigmoid.png)

### Efficient computation of SigLIP's loss

Since SigLIP's loss is example-level instead of batch-level, the loss for each pair can be computed independently.
$$
\mathcal{L}_{ij}
=\log\sigma\left(-z_{ij}(t\mathbf{x_i}\mathbf{y_j}-b)\right)
$$

We can use this fact to implement an efficient method of computing SigLIP's loss with minimal memory requirements and minimal inter-device communication.

This lovely figure from the SigLIP paper gives a good sense for the process in a setting with three devices (i.e. GPUs, TPUs, etc) and a global "batch size" of 12.

<!-- <figure class="fullwidth"> -->
![siglip efficient](/assets/img/siglip-efficient.png)
<!-- </figure> -->

In essence, each device begins  with an equal number of text and image examples. (_left_) The loss computation proceeds by computing the loss for every possible image-text pair present locally on a given device. Those losses are summed per-device and stored locally. (_middle_) Then, the *text* examples are swapped between devices in a as if along a ring[^ring] (not images, since more bytes would need to be communicated). We can again compute the loss for every image-text pair present on the same device. These losses can be summed with the running per-device loss computed in the first step. (_right_) This proceedure proceeds until all image-text pairs within a batch have been processed. Finally, a single cross-device summation is required to compute the final value of the loss for the batch.

The per-device memory requirements here scale with the square of the local batch size ($b=\frac{|B|}{d}$). Which means that this training proceedure can be scaled to arbitrarily large batch sizes by increasing the number of devices.

[^ring]: Text examples swapped along a ring. ![A ring of text transmissions](/assets/img/device-ring.png)


## Does SigLIP inherit CLIP's issues?
Recently, several works have pointed out issues with the underlying vision capabilities of CLIP models. For example, [Liang et al.](https://arxiv.org/abs/2203.02053) showed that the image and text embedding clouds created by CLIP models are non-overlapping.

Perhaps more concerning is the fact that CLIP models are blind to certain types of visual information. For example, [Tong et al.](https://arxiv.org/abs/2401.06209) showed...

<blockquote class="twitter-tweet" data-dnt="true" data-theme="light"><p lang="en" dir="ltr">Glad to see SigLIP doing well (relatively speaking) on this new benchmark! DFN by <a href="https://twitter.com/Vaishaal?ref_src=twsrc%5Etfw">@Vaishaal</a> also doing quite well. The rest seems pretty far behind, even with larger models.<br><br>Now I wonder how well CapPa will do here, given the SugarCrepe results, I think very well :) <a href="https://t.co/kELoELt3sv">https://t.co/kELoELt3sv</a> <a href="https://t.co/sVk3ukAgT2">pic.twitter.com/sVk3ukAgT2</a></p>&mdash; Lucas Beyer (bl16) (@giffmana) <a href="https://twitter.com/giffmana/status/1746867524996620363?ref_src=twsrc%5Etfw">January 15, 2024</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

## Interested in learning more?
I recommend reading the CLIP and SigLIP papers. They are both excelent.

## References

1. Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever. ["Learning Transferable Visual Models From Natural Language Supervision"](https://arxiv.org/abs/2103.00020). arXiv:2103.00020, 2021.

2. Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, Lucas Beyer. ["Sigmoid Loss for Language Image Pre-Training"](https://arxiv.org/pdf/2303.15343). arXiv:2303.15343, 2023.

3. Xiaohua Liang, Haohan Wang, Jong-Hwi Yoon, Davide Mottin, Shafiq Joty. ["Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Learning"](https://arxiv.org/abs/2203.02053). arXiv:2203.02053, 2022.

4. Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee. ["Visual Instruction Tuning"](https://arxiv.org/abs/2304.08485). arXiv:2304.08485, 2023.