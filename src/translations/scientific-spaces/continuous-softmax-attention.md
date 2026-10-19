---
title: "Continuous Softmax Attention"
subtitle: "Translated from [连续化 Softmax Attention](https://kexue.fm/archives/11980) by Jianlin Su (苏剑林)"
date: 2026-10-19T22:21:00+08:00
blurb: "Making Softmax Attention continuous by linearly interpolating keys and values yields a closed form that adds a Short Conv-like correction to the attention logits plus a sink token. In language-model training, its early advantage fades by the end."
tags: ["translation", "attention", "softmax-attention", "model-architecture"]
math: true
---

*Translator's note (Opus 5.5): This is an English translation of [连续化 Softmax Attention](https://kexue.fm/archives/11980) by Jianlin Su (苏剑林), originally published on October 19, 2026 on [Scientific Spaces (科学空间)](https://kexue.fm). The translation preserves the author's first-person voice.*

<hr class="section-divider">

Softmax Attention surely needs no introduction by now: it is a weighted mixing operator defined over a finite set of vectors. Recently, the paper [*Continuum Attention*](https://papers.cool/arxiv/2610.20586) proposed a very interesting idea: making attention continuous. This reminded me of [EFLA](https://papers.cool/arxiv/2512.12602), which I introduced earlier in [*Why Does DeltaNet Need L2 Normalize?*](https://kexue.fm/archives/11486). EFLA likewise makes a discrete RNN continuous and then discretizes it again, arriving at some interesting results. Similar examples include the earlier [SSM](https://kexue.fm/tag/ssm/) line of work.

Seen this way, "continuize, then discretize" seems to be a tried-and-true (屡试不爽) path of exploration. So what interesting results can we get by making Softmax Attention continuous? Let's take a closer look together.

## Problem Statement

Standard Softmax Attention is defined as

\[
\boldsymbol{o}_t = \frac{\sum_{i=1}^t e^{\boldsymbol{q}_t\cdot\boldsymbol{k}_i} \boldsymbol{v}_i}{\sum_{i=1}^t e^{\boldsymbol{q}_t\cdot\boldsymbol{k}_i}} \tag{1}
\]

By continuization, we mean changing it into<label for="mn-continuous-attention-interpolation" class="margin-toggle">&#8853;</label><input type="checkbox" id="mn-continuous-attention-interpolation" class="margin-toggle" aria-label="Discrete-to-continuous attention diagram" /><span class="marginnote marginnote-detail"><img src="/assets/img/continuous-attention-interpolation.png" alt="Three stages of continuous attention: discrete key-value pairs at integer positions, piecewise-linear interpolation between neighboring pairs, and integration over every interval into a weighted output mixture." sizes="(max-width: 760px) 88vw, 340px" /></span>

\[
\boldsymbol{o}_t = \frac{\int_0^t e^{\boldsymbol{q}_t\cdot\boldsymbol{k}_s} \boldsymbol{v}_s ds}{\int_0^t e^{\boldsymbol{q}_t\cdot\boldsymbol{k}_s} ds} \tag{2}
\]

where \(\boldsymbol{k},\boldsymbol{v}\) at non-integer points are defined by linear interpolation. That is, for any \(s \in [i-1, i], i\in\mathbb{N}_+\), we define

\[
\begin{aligned}
\boldsymbol{k}_s
&= (i - s)\boldsymbol{k}_{i-1} \\
&\quad + (s - i + 1) \boldsymbol{k}_i, \\[6pt]
\boldsymbol{v}_s
&= (i - s)\boldsymbol{v}_{i-1} \\
&\quad + (s - i + 1) \boldsymbol{v}_i
\end{aligned} \tag{3}
\]

This definition requires introducing two additional vectors, \(\boldsymbol{k}_0,\boldsymbol{v}_0\) (used for \(s\in[0,1]\)). We will discuss their values at the end.

Note that the definition in equation (2) differs slightly from the original paper. For the sake of symmetry, the original paper changes the integration interval to \([1/2, t + 1/2]\). I feel that although this adjustment is more symmetric, it is actually less natural in the causal setting, so this post uses the integration interval \([0,t]\) instead. Of course, the derivation that follows has little to do with the choice of interval; technically, it is the same either way.

## Preliminaries

Equation (2) is only a formal definition. To actually use it, we must compute the integrals explicitly. By the additivity of integrals, we only need to compute each segment \(\int_i^{i+1}\) separately. Writing \(a_{t,i} = \boldsymbol{q}_t\cdot\boldsymbol{k}_i, \delta_{t,i} = a_{t,i} - a_{t,i-1}\), it is not hard to obtain

\[
\begin{aligned}
\int_{i-1}^i
e^{\boldsymbol{q}_t\cdot\boldsymbol{k}_s}\boldsymbol{v}_s\,ds
&= \int_0^1 e^{(1-s)a_{t,i-1}+sa_{t,i}} \\
&\quad {}\cdot
\bigl((1-s)\boldsymbol{v}_{i-1}+s\boldsymbol{v}_i\bigr)\,ds \\
&= \boldsymbol{v}_i
\int_0^1 e^{a_{t,i-1}+s\delta_{t,i}}\,ds \\
&\quad +(\boldsymbol{v}_{i-1}-\boldsymbol{v}_i) \\
&\qquad {}\cdot
\int_0^1 e^{a_{t,i-1}+s\delta_{t,i}}(1-s)\,ds \\[4pt]
&= e^{a_{t,i-1}}\varphi_1(\delta_{t,i})\boldsymbol{v}_i \\
&\quad +e^{a_{t,i-1}}\varphi_2(\delta_{t,i}) \\
&\qquad {}\cdot(\boldsymbol{v}_{i-1}-\boldsymbol{v}_i)
\end{aligned}
\tag{4}
\]

\[
\begin{aligned}
\int_{i-1}^i e^{\boldsymbol{q}_t\cdot\boldsymbol{k}_s}\,ds
&= \int_0^1 e^{(1-s)a_{t,i-1}+sa_{t,i}}\,ds \\
&= \int_0^1 e^{a_{t,i-1}+s\delta_{t,i}}\,ds \\
&= e^{a_{t,i-1}}\varphi_1(\delta_{t,i})
\end{aligned}
\tag{5}
\]

Here

\[
\begin{aligned}
\varphi_1(x)
&\triangleq \int_0^1 e^{sx}\,ds
= \frac{e^x-1}{x}, \\
\varphi_2(x)
&\triangleq \int_0^1 e^{sx}(1-s)\,ds \\
&= \frac{e^x-1-x}{x^2}
\end{aligned}
\tag{6}
\]

At \(x=0\), both are defined by their limits, giving \(\varphi_1(0)=1\) and \(\varphi_2(0)=1/2\), respectively. They also satisfy the following identity:

\[
\begin{aligned}
\varphi_1(x)-\varphi_2(x)
&= \frac{xe^x-e^x+1}{x^2} \\
&= e^x\cdot
\frac{e^{-x}-1-(-x)}{(-x)^2} \\
&= e^x\varphi_2(-x)
\end{aligned}
\tag{7}
\]

This identity will be used in the simplification in the next section. In addition, the integral definitions make it clear that \(\varphi_1(x)\) and \(\varphi_2(x)\) are always positive, so the logarithms we take later will never be undefined.

## Assembly and Simplification

Substituting the above integrals into equation (2), we obtain

\[
\boldsymbol{o}_t =
\frac{
\begin{aligned}
&\sum_{i=1}^t
e^{a_{t,i-1}}\varphi_1(\delta_{t,i})\boldsymbol{v}_i \\
&\quad +e^{a_{t,i-1}}\varphi_2(\delta_{t,i}) \\
&\qquad {}\cdot(\boldsymbol{v}_{i-1}-\boldsymbol{v}_i)
\end{aligned}
}{
\sum_{i=1}^t e^{a_{t,i-1}}\varphi_1(\delta_{t,i})
}
\tag{8}
\]

First, we simplify the numerator:

\[
\begin{aligned}
&\sum_{i=1}^t
e^{a_{t,i-1}}\varphi_1(\delta_{t,i})\boldsymbol{v}_i \\
&\quad +e^{a_{t,i-1}}\varphi_2(\delta_{t,i})
(\boldsymbol{v}_{i-1}-\boldsymbol{v}_i) \\
={}&\sum_{i=1}^t e^{a_{t,i-1}}
\bigl(\varphi_1(\delta_{t,i})-\varphi_2(\delta_{t,i})\bigr)
\boldsymbol{v}_i \\
&\quad +e^{a_{t,i-1}}\varphi_2(\delta_{t,i})
\boldsymbol{v}_{i-1} \\
={}&\sum_{i=1}^t e^{a_{t,i-1}+\delta_{t,i}}
\varphi_2(-\delta_{t,i})\boldsymbol{v}_i \\
&\quad +e^{a_{t,i-1}}\varphi_2(\delta_{t,i})
\boldsymbol{v}_{i-1} \\
={}&\sum_{i=1}^t e^{a_{t,i}}
\varphi_2(-\delta_{t,i})\boldsymbol{v}_i \\
&\quad +\sum_{i=0}^{t-1} e^{a_{t,i}}
\varphi_2(\delta_{t,i+1})\boldsymbol{v}_i \\
={}&e^{a_{t,0}}\varphi_2(\delta_{t,1})\boldsymbol{v}_0 \\
&\quad +e^{a_{t,t}}\varphi_2(-\delta_{t,t})\boldsymbol{v}_t \\
&\quad +\sum_{i=1}^{t-1} e^{a_{t,i}}
\bigl(\varphi_2(-\delta_{t,i}) \\
&\qquad\qquad
{}+\varphi_2(\delta_{t,i+1})\bigr)\boldsymbol{v}_i
\end{aligned}
\tag{9}
\]

So the numerator has the form \(\sum_{i=0}^t e^{a_{t,i} + b_{t,i}} \boldsymbol{v}_i\), where

\[
\begin{aligned}
a_{t,i}
&= \boldsymbol{q}_t\cdot\boldsymbol{k}_i, \\
e^{b_{t,0}}
&= \varphi_2(a_{t,1}-a_{t,0}), \\
e^{b_{t,i}}
&= \varphi_2(a_{t,i-1}-a_{t,i}) \\
&\quad +\varphi_2(a_{t,i+1}-a_{t,i}), \\
&\qquad 1\leq i\leq t-1, \\
e^{b_{t,t}}
&= \varphi_2(a_{t,t-1}-a_{t,t})
\end{aligned}
\tag{10}
\]

Note that the denominator is exactly what we get by replacing every \(\boldsymbol{v}_i\) with \(1\), so the denominator is directly \(\sum_{i=0}^t e^{a_{t,i} + b_{t,i}}\). Therefore,

\[
\boldsymbol{o}_t = \frac{\sum_{i=0}^t e^{a_{t,i} + b_{t,i}} \boldsymbol{v}_i}{\sum_{i=0}^t e^{a_{t,i} + b_{t,i}}} \tag{11}
\]

## Analysis of the Result

Compared with standard Softmax Attention,

\[
\boldsymbol{o}_t = \frac{\sum_{i=1}^t e^{a_{t,i}} \boldsymbol{v}_i}{\sum_{i=1}^t e^{a_{t,i}}} \tag{12}
\]

the changes brought by continuization are the extra \(\boldsymbol{k}_0,\boldsymbol{v}_0\) and a correction term \(b_{t,i}\) added to each \(a_{t,i}\). The extra \(\boldsymbol{k}_0,\boldsymbol{v}_0\) are not really an essential change, and we will briefly discuss them in a moment. For now, let's focus on \(b_{t,i}\).

For a general \(1\leq i\leq t-1\), \(b_{t,i}\) depends on \(a_{t,i}\) as well as its left and right neighbors \(a_{t,i-1},a_{t,i+1}\). Formally, this looks a lot like adding a Short Conv to the Attention matrix, and this becomes even more evident after a Taylor expansion of \(b_{t,i}\):<label for="mn-continuous-attention-short-conv" class="margin-toggle">&#8853;</label><input type="checkbox" id="mn-continuous-attention-short-conv" class="margin-toggle" aria-label="Short Conv correction diagram" /><span class="marginnote marginnote-detail"><img src="/assets/img/continuous-attention-short-conv.png" alt="A three-tap Short Conv stencil applies weights plus one, minus two, and plus one to neighboring attention logits, producing their second difference and the approximate correction b sub i equals that difference divided by six." sizes="(max-width: 760px) 88vw, 340px" /></span>

\[
\begin{aligned}
b_{t,i}
&= \log\bigl(\varphi_2(a_{t,i-1}-a_{t,i}) \\
&\qquad +\varphi_2(a_{t,i+1}-a_{t,i})\bigr) \\
&= \frac{
a_{t,i-1}+a_{t,i+1}-2a_{t,i}
}{6}+\cdots
\end{aligned} \tag{13}
\]

Here, \(a_{t,i-1} + a_{t,i+1} - 2a_{t,i}\) is precisely the second-order difference of \(a_{t,i}\), which is a simple Short Conv operation. So continuous Softmax Attention and "adding a Short Conv to the Attention matrix" really do reach similar ends by different routes (异曲同工).

As for \(\boldsymbol{k}_0,\boldsymbol{v}_0\), they are essentially equivalent to the sink token we introduce to address the Attention Sink problem. In the context of continuization, their main purpose is to fill out all \(t\) intervals, so they are purely a mathematical device. We generally have three choices: all zeros; \(\boldsymbol{k}_0=\boldsymbol{k}_1,\boldsymbol{v}_0=\boldsymbol{v}_1\); or trainable. In the experiments below, I chose all zeros.

## Simple Experiments

Because \(b_{t,i}\) depends on both \(a_{t,i}\) and its neighbors, continuous Softmax Attention cannot reuse existing FlashAttention implementations. However, the implementation principle carries over, and the theoretical FLOPs are essentially the same. I had K3 write a simple kernel. It runs at only about 1/4 the speed of Softmax Attention, mainly because it contains too many fine-grained operations that are hard to accelerate. Still, it is enough to run some experiments for now.

The first experimental finding is that, although continuization introduces a locality prior, it cannot fully replace position encoding. In an architecture that uses Full Attention throughout, continuous Softmax Attention without RoPE is clearly worse than standard Softmax Attention with RoPE. Conversely, once RoPE is added, it closes the gap.

As for training dynamics, with RoPE added to both, continuous Softmax Attention is clearly better than standard Softmax Attention early in training, but the gap gradually narrows. By the end of training, the two curves essentially coincide and the advantage disappears. My guess is that the core reason is that natural language is not a true time series, so continuization does not bring any substantive benefit. Perhaps an advantage would only show up on genuine time series.

I also ran an experiment that switched from standard Softmax Attention to the continuous version partway through training. The loss jump at the switch was not very large, and the model eventually converged slowly to essentially the same performance as the standard version.

## Summary

This post briefly derived continuous Softmax Attention, and the result resembles applying a Short Conv operation to the Attention matrix. Although my experiments did not find any advantage from this continuization on natural language tasks, it is still an interesting result worth appreciating.

<hr class="section-divider">

*Citation: Su, J. (2026, October 19). 连续化 Softmax Attention [Continuous Softmax Attention]. Scientific Spaces. [https://kexue.fm/archives/11980](https://kexue.fm/archives/11980)*

*Original content licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). This translation is shared under the same license.*