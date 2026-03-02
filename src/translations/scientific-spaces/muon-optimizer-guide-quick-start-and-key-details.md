---
title: "Muon Optimizer Guide: Quick Start and Key Details"
subtitle: "Translated from [Muon优化器指南：快速上手与关键细节](https://kexue.fm/archives/11416) by Jianlin Su (苏剑林)"
date: 2025-11-19T00:00:00+08:00
blurb: "A practical guide to switching from Adam to Muon, covering the four variants, dimension ordering pitfalls, and hyperparameter conversion rules."
tags: ["translation", "muon", "optimization", "adam"]
math: true
code: true
---

*Translator's note (Opus 4.6): This is an English translation of [Muon优化器指南：快速上手与关键细节](https://kexue.fm/archives/11416) by Jianlin Su (苏剑林), originally published on November 19, 2025 on [Scientific Spaces (科学空间)](https://kexue.fm). The translation preserves the author's first-person voice.*

<hr class="section-divider">

By now, many readers have likely come across news about the Muon optimizer. Muon was originally proposed around October of last year by [Keller Jordan](https://x.com/kellerjordan0/status/1842300916864844014) on Twitter — barely over a year ago. Yet in that single year, Muon has already been tested on models with tens of billions, hundreds of billions, and even trillions of parameters, demonstrating that it is a highly competitive optimizer.

Today, Muon is built into training frameworks like [Torch](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html) and [Keras](https://keras.io/api/optimizers/muon/), and even large-scale frameworks like [Megatron](https://github.com/NVIDIA/Megatron-LM/blob/dev/megatron/core/optimizer/muon.py) have begun adding support — a sign that it has earned broad industry recognition. However, for readers who are only familiar with Adam, how to quickly and effectively switch to Muon may still be confusing. This post aims to provide a quick-start tutorial.

## Brief Introduction

Muon's originator is [Keller Jordan](https://x.com/kellerjordan0/status/1842300916864844014), currently at OpenAI. As mentioned, Muon was first published on Twitter, and to this day the author has only written a blog post — [*Muon: An optimizer for hidden layers in neural networks*](https://kellerjordan.github.io/posts/muon/) — rather than a paper. His position is that "whether or not you write a paper has nothing to do with whether the optimizer works"[^original-tweet].

[^original-tweet]: [Original tweet](https://x.com/kellerjordan0/status/1890178773586489716)

Muon is an optimizer specifically designed for matrix parameters. Some related works share similar characteristics, such as [*Shampoo*](https://arxiv.org/abs/1802.09568), and the earlier [*Stochastic Spectral Descent*](https://kexue.fm/archives/10592), among others. Many works can be connected to Muon to varying degrees, but none fully subsumes it, so in my view Muon counts as an entirely new contribution.

In China, the earliest article introducing Muon to a broader audience was my blog post [*Appreciating the Muon Optimizer: The Essential Leap from Vectors to Matrices*](https://kexue.fm/archives/10592), and the first large-scale validation of Muon was probably our February release of [*Moonlight*](https://arxiv.org/abs/2502.16982). The Moonlight variant of Muon was subsequently used in the trillion-parameter [*K2*](https://arxiv.org/abs/2507.20534). After K2, [*GLM-4.5*](https://arxiv.org/abs/2508.06471) also adopted this Muon variant.

As Muon co-author Jeremy Bernstein wrote in his blog post [*Deriving Muon*](https://jeremybernste.in/writing/deriving-muon), what makes Muon special to me is that it can be derived from more fundamental optimization principles and is effective in practice. By comparison, although Adam is also effective, it is more of a heuristic approach.

## Four Variants

This post does not intend to cover Muon's mathematical details or implementation, but rather focuses on the technical details and caveats of switching from Adam to Muon. As mentioned, Muon is designed specifically for matrix parameter optimization and uses a non-element-wise update rule, which can be confusing for new users.

Furthermore, to my knowledge Muon currently has at least four slightly different versions, and this multi-version situation adds to the confusion. If users are unaware of the details, they may set hyperparameters incorrectly (especially the learning rate) and get poor results. Below I will clarify these issues. For a matrix \(\boldsymbol{W}\in\mathbb{R}^{d_{in}\times d_{out}}\), with \(\boldsymbol{G}\) as its gradient, the four Muon variants are:

\[
\boldsymbol{M}_t = \beta \boldsymbol{M}_{t-1} + \boldsymbol{G}_t
\]

\[
\boldsymbol{W}_t = \boldsymbol{W}_{t-1} - \eta_t \left(\text{msign}(\boldsymbol{M}_t) + \lambda \boldsymbol{W}_{t-1}\right) \quad \color{#87CEEB}{\text{(Naive)}}
\]

\[
\boldsymbol{W}_t = \boldsymbol{W}_{t-1} - \eta_t \left(\sqrt{\max(1,\, d_{out}/d_{in})}\;\text{msign}(\boldsymbol{M}_t) + \lambda \boldsymbol{W}_{t-1}\right) \quad \color{#87CEEB}{\text{(KellerJordan)}}
\]

\[
\boldsymbol{W}_t = \boldsymbol{W}_{t-1} - \eta_t \left(\sqrt{d_{out}/d_{in}}\;\text{msign}(\boldsymbol{M}_t) + \lambda \boldsymbol{W}_{t-1}\right) \quad \color{#87CEEB}{\text{(MuP)}}
\]

\[
\boldsymbol{W}_t = \boldsymbol{W}_{t-1} - \eta_t \left(0.2\times\sqrt{\max(d_{out},d_{in})}\;\text{msign}(\boldsymbol{M}_t) + \lambda \boldsymbol{W}_{t-1}\right) \quad \color{#87CEEB}{\text{(Moonlight)}}
\]

To enable Nesterov momentum, replace \(\text{msign}(\boldsymbol{M}_t)\) with \(\text{msign}(\beta\boldsymbol{M}_t + \boldsymbol{G}_t)\). In implementations, \(\text{msign}\) is typically named `zeropower_via_newtonschulz`; ordinary users need not worry about the implementation details.

The only difference between the four versions is the scaling factor in front of \(\text{msign}\). The "KellerJordan" and "MuP" versions are very similar, while the "Moonlight" version is slightly different. Keras only implements the "KellerJordan" version, whereas Torch implements both the "KellerJordan" and "Moonlight" versions. The naive version is currently uncommon; I personally use my own "MuP" version.

## The Two Dimensions

An important detail here is that the "KellerJordan" and "MuP" versions are sensitive to the ordering of \(d_{in}\) and \(d_{out}\). So the first thing to sort out is what \(d_{in}\) and \(d_{out}\) mean — the first dimension of the matrix is not necessarily \(d_{in}\), and the second is not necessarily \(d_{out}\).

\(d_{in}\) and \(d_{out}\) refer to the input and output dimensions of the linear layer respectively, so which is which depends on the specific implementation. For example, Keras's Dense layer computes \(\boldsymbol{x}\boldsymbol{W}\), so the first dimension of \(\boldsymbol{W}\) is \(d_{in}\) and the second is \(d_{out}\). However, Torch's Linear layer computes \(\boldsymbol{x}\boldsymbol{W}^{\top}\), so the second dimension of \(\boldsymbol{W}\) is \(d_{in}\), and the first dimension is \(d_{out}\).

Therefore, to implement the "KellerJordan" version of Muon for Torch's Linear layer, the scaling factor should be `max(1, W.shape[0]/W.shape[1])**0.5`, whereas for Keras it should be `max(1, W.shape[1]/W.shape[0])**0.5`. Consequently, the current Keras (v3.12) Muon implementation is actually incorrect, because it copied Torch's scaling factor implementation directly[^keras-source].

[^keras-source]: [Keras source code](https://github.com/keras-team/keras/blob/v3.12.0/keras/src/optimizers/muon.py#L198)

If you are writing your own model, you need to judge carefully based on your own code. For example, it is entirely possible to mix Torch's built-in Linear layer with hand-written `x @ W` in the same model, in which case you cannot uniformly use `W.shape[0]/W.shape[1]` or `W.shape[1]/W.shape[0]`. Of course, if you find sorting this out too tedious, you can use the "Moonlight" version — its scaling factor is symmetric with respect to \(d_{in}\) and \(d_{out}\).

## Hyperparameter Settings

Once you have \(d_{in}\) and \(d_{out}\) sorted out, what remains is how to set the learning rate \(\eta_t\) and weight decay coefficient \(\lambda\). The assumption here is that you already have tuning experience with Adam and have achieved good results, and you want to quickly migrate to Muon to try it out.

Let us first look at the "Moonlight" version. Its scaling factor is obtained by aligning with Adam's Update RMS. For details, see [*Muon Sequel: Why We Chose to Try Muon?*](https://kexue.fm/archives/10739). As for the \(0.2\) "magic number," see [*Why Is Adam's Update RMS 0.2?*](https://kexue.fm/archives/11267). In short, the "Moonlight" Muon aligns with Adam's update magnitude, so the simplest approach when migrating from Adam is: **change nothing** — just reuse Adam's \(\eta_t\) and \(\lambda\).

Now consider the remaining three versions. We know that mainstream models typically have a hidden_size (denoted \(d\)), and most matrix shapes do not deviate significantly from \(d\times d\). So we can approximate with \(d_{in}=d_{out}=d\), in which case all three versions are identical, and compared to the "Moonlight" version they are missing a factor of \(0.2\sqrt{d}\). Since the "Moonlight" version aligns with Adam's update magnitude and requires no hyperparameter changes, the learning rate for the other three versions should be scaled up by \(0.2\sqrt{d}\) to match Adam's update magnitude, and correspondingly \(\lambda\) should be divided by \(0.2\sqrt{d}\).

Substituting \(d=1024, 2048, 4096\) gives \(6.4, 9, 12.8\) respectively. If you cannot remember \(0.2\sqrt{d}\), a simple rule of thumb is: when using any of the other three Muon variants, **multiply Adam's learning rate by 10** to get Muon's learning rate. If you plug Adam's learning rate directly into Muon, you will get underfitting and conclude that Muon is far worse than Adam. To my knowledge, some negative reviews of Muon stem from exactly this mistake.

So does this mean the "Moonlight" version is easier to use? The "Moonlight" version does have solid practical results, but calling it "easier to use" is really an evaluation from Adam's perspective. The advantage of the "MuP" or "KellerJordan" versions is learning rate transferability — once you tune the learning rate on a small model, it often works well when applied directly to a large model. For this, see Jeremy Bernstein's blog post [*Deriving Muon*](https://jeremybernste.in/writing/deriving-muon) or my blog post [*Higher-Order MuP: Simpler Yet Smarter Spectral Condition Scaling*](https://kexue.fm/archives/10795).

## Other Parameters

If Muon only handles matrix parameters, what about the rest? For example, the bias term in linear layers or the gamma parameter in RMSNorm — these are 1-dimensional parameters. And convolutional layers may have 3D or 4D parameter tensors.

First, a correction: Muon does not just handle matrix parameters — Muon handles **matrix parameters of linear layers with dense inputs**. If this sounds confusing, just remember that the matrix parameters of the Embedding layer and the final classification layer (including GPT's LM Head) cannot use Muon, or performance will suffer noticeably. For these matrix parameters that cannot use Muon, as well as 1D, 3D, and higher-dimensional parameters, if you do not want to think too hard, just use Adam. Most Muon implementations are mixed with Adam anyway, and users can select which layers use Adam.

If you are willing to tinker, you can also apply Muon to 3D and 4D convolutional parameters. Take Conv2D as an example: the kernel shape is typically \((w, h, d_{in}, d_{out})\). Its equivalent implementation flattens the \((w, h, d_{in})\) input patch into a vector of dimension \(w \times h \times d_{in}\), and reshapes the kernel to \((w\times h \times d_{in},\, d_{out})\), then performs matrix multiplication. So to use Muon, you first reshape the momentum to \((w\times h \times d_{in},\, d_{out})\), compute \(\text{msign}\), then reshape back to update.

Similarly, the gamma parameter of RMSNorm can be viewed as multiplication by a diagonal matrix, so you can treat its momentum as a diagonal matrix and compute \(\text{msign}\), which is equivalent to SignSGDM. The Embedding layer can be viewed as multiple \((1,d)\) matrices for computing \(\text{msign}\), which yields Normalized SGDM (see [*Appreciating the Muon Optimizer: The Essential Leap from Vectors to Matrices*](https://kexue.fm/archives/10592)). If you want to tinker further — for example, in Multi-Head Attention, could you take each head's projection matrix and compute \(\text{msign}\) separately...

As they say, never stop tinkering (生命不息，折腾不止)!

## Expected Results

Finally, if you have followed the instructions above and everything is running correctly, you can start hoping for good fortune.

What kind of results should you expect? If there are no abnormal situations like gradient explosions, then in most cases Muon will be slightly better than Adam. Of course, it is also possible that in some cases Muon will be slightly worse. But either way, the gap between them will not be very large. If you observe one being dramatically better than the other, you should probably reconsider whether something is misconfigured on one side.

That said, none of this is absolute. Under certain extreme settings, it is indeed possible for Muon to be much better than Adam, with Adam failing no matter how you tune it. In any case, good luck. If you observe interesting phenomena, feel free to share and analyze them together.

<hr class="section-divider">

*Citation: Su, J. (2025, November 19). Muon优化器指南：快速上手与关键细节 [Muon Optimizer Guide: Quick Start and Key Details]. Scientific Spaces. [https://kexue.fm/archives/11416](https://kexue.fm/archives/11416)*

*Original content licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). This translation is shared under the same license.*
