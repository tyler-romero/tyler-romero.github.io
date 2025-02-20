---
title: NanoGPT Speedrun Living Worklog
subtitle: How fast can I train GPT-2 on two RTX 4090 GPUs?
date: 2025-02-18T00:00:00-08:00
blurb: How fast can I train GPT-2 on two RTX 4090 GPUs? This is a living worklog of my progress.
tags: ["post", "llm", "gpt2", "speedrun", "nanogpt", "worklog", "muon"]
---

I've seen [some](https://x.com/kellerjordan0/status/1859331370268623321) [really](https://x.com/kellerjordan0/status/1842300916864844014) [awesome](https://x.com/kellerjordan0/status/1876048851158880624) [GPT-2](https://x.com/hi_tysam/status/1879687807678959729) speedrun results from people like [Keller Jordan](https://x.com/kellerjordan0), [Fern](https://x.com/hi_tysam), [Braden Koszarsky](https://x.com/KoszarskyB), and others. I got a little inspired and wanted to see how fast I could train GPT-2 on my own hardware.

Technically, [the NanoGPT speedrun](https://x.com/kellerjordan0/status/1798863559243513937) is to train a neural network to 3.28 validation loss on FineWeb as fast as possible on an 8xH100 node. [Keller Jordan maintains a leaderboard here](https://github.com/KellerJordan/modded-nanogpt?tab=readme-ov-file#world-record-history). At the time of writing (Jan 16, 2025), the record is 3.14 minutes (!).

I have access to **2xRTX 4090 GPUs** and I want to see how fast I can train GPT-2 on them by following the same rules as the NanoGPT speedrun. If I see some success, I may try to transfer my methods to an 8xH100 node for comparison with the main leaderboard.

I'll be documenting my progress here and updating this post as I go. Code can be found in [this GitHub repo](https://github.com/tyler-romero/nanogpt-speedrun).

## Progress so far
| #                                                    | Description           | Record time | Training Tokens | Tokens/Second | Date       | Commit                                                                                                      | Log                                                                                                              |
| :--------------------------------------------------- | :-------------------- | :---------- | :-------------- | :------------ | :--------- | :---------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------- |
| [1](#1-initial-setup-and-baseline)                   | Initial baseline      | 8.13 hours  | 6.44B           | 221k          | 2025/01/16 | [b3c32f8](https://github.com/tyler-romero/nanogpt-speedrun/commit/b3c32f8937c1f4655c5eb9607970e03e351a6c08) | [here](https://github.com/tyler-romero/nanogpt-speedrun/blob/main/logs/4c627c0d-029c-4f8a-bd18-40f99b43b22e.txt) |
| [2.1](#21-architectural-changes-and-training-tweaks) | Architectural changes | 7.51 hours  | 5.07B           | 188k          | 2025/01/18 | [b7bb93f](https://github.com/tyler-romero/nanogpt-speedrun/commit/b7bb93fd988d73a55184c553f0020feec1454340) | [here](https://github.com/tyler-romero/nanogpt-speedrun/blob/main/logs/14fcdb07-443d-4d1c-b307-061bc4bd2cd6.txt) |
| [2.2](#22-muon-optimizer)                            | Muon optimizer        | 4.53 hours  | 3.04B           | 187k          | 2025/01/23 | [b91c2c0](https://github.com/tyler-romero/nanogpt-speedrun/commit/b91c2c00673b125944abde277dd5ef3dc141284d) | [here](https://github.com/tyler-romero/nanogpt-speedrun/blob/main/logs/59951c17-fbe5-4577-a1bc-6dc0c1802d2e.txt) |
| [2.3](#23-dataloading-tweaks)                        | Dataloading tweaks    | 4.26 hours  | 3.31B           | 216k          | 2025/02/18 | [d59944d](https://github.com/tyler-romero/nanogpt-speedrun/commit/d59944dbe8535fea8ea107d9a6fb133de5346de5) | [here](https://github.com/tyler-romero/nanogpt-speedrun/blob/main/logs/08047f73-cb01-4f47-a901-de901b2a6b6e.txt) |


## 1. Initial setup and baseline

Part of the goal of this project is for me to learn as I go, so I am going to start at the beginning - with with Andrej Karpathy's [PyTorch GPT-2 trainer](https://github.com/karpathy/llm.c/blob/7b929300217ff1a974b63791a228928b39b26409/train_gpt2.py) from [llm.c](https://github.com/karpathy/llm.c). This is the script that Keller Jordan used for [his initial baseline](https://github.com/KellerJordan/modded-nanogpt/tree/master?tab=readme-ov-file#modded-nanogpt). This trainer is very similar to the NanoGPT trainer with some minor modifications / simplifications (such as no dropout).

I have upstreamed some QOL improvements and basic tweaks to the training script from Keller's fork, but have not changed any of the core training / modeling logic. Specifically:
1. Implemented gradient accumulation so that my 2x24GB GPUs simulate the training experience of a 8xH100 machine.
2. Increased learning rate to 0.0015 and halved the batch size (total batch size is 262144 - that is bs of `32/device * 2 devices * 1024 sequence length * 4 gradient accum steps`).
3. Improved learning rate schedule (linear warmup then linear decay).
4. Removed all affine scale/bias parameters and switched to RMSNorm.
5. Padded the vocab size from 50257 to 50304 to make it a multiple of 128 (for better tensor core utilization).
6. Using Pytorch 2.5.1 (the switch from 2.4 to 2.5 gave ~9% speedup on the 8xH100 leaderboard).

Additionally, I added `wandb` logging for easy tracking of training progress - optimistically I may need to remove this one day as it slightly increases step time.

Commit with the initial setup is here: [`b3c32f8`](https://github.com/tyler-romero/nanogpt-speedrun/blob/main/logs/4c627c0d-029c-4f8a-bd18-40f99b43b22e.txt).

The baseline run time on my 2xRTX 4090 setup is **8.13 hours**.

<!-- TODO: plot -->

## 2. Implementing major improvements from the 8xH100 leaderboard

Waiting 8 hours for a result is too slow for effective experimentation, so I'm going to begin by implementing some of the notable improvements from the 8xH100 leaderboard. I'll start with the most impactful/easiest changes first:
1. Architectural changes and training tweaks
2. Muon optimizer
3. Dataloading tweaks

### 2.1 Architectural changes and training tweaks
There are some basic architectural changes and modernizations that can be made to the model that will speed up training. These changes are general improvements to the transformer decoder architecture that have been generally adopted since the original GPT-2 paper. The changes are:
1. [RoPE (Rotary Positional Embeddings)](https://arxiv.org/abs/2104.09864). There are [many](https://www.jitx.io/posts/rope-embeddings) [good](https://blog.eleuther.ai/rotary-embeddings/) explanations of RoPE out there so I won't go into detail here.
2. [ReLU^2 Activation](https://arxiv.org/pdf/2109.08668)[^relu2]. Many activations that are better than GeLU have been proposed since GPT-2. ReLU^2 is a simple one that has been shown to be effective in decreasing training time required to reach a certain validation loss.
3. No gradient clipping. Gradient clipping can help stabilize training but it also slows down training. Since we are speed-running, we will remove gradient clipping. This also eliminates a hyperparameter that needs to be tuned.
4. [Trapezoidal learning rate schedule](https://arxiv.org/abs/2405.18392). While cosine learning rate schedules are the de-facto standard, they can be difficult to work with since changing the number of training steps changes the entire schedule. Trapezoidal learning rate schedules are often easier to reason about / tune around, and they have been show to match the performance of cosine schedules.

[^relu2]: ReLU^2 activation function. ![Relu Activation plot](/assets/img/relu2.png)

In addition, learning rate and batch size have been tuned.

Once again, many of these changes are [downstreamed](https://en.wikipedia.org/wiki/Downstream_(software_development)) from the [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) repository / 8xH100 speedrun. Its not efficient to reinvent the wheel, and I want to get training time down as fast as possible in the beginning.

After implementing these changes (commit [`b7bb93f`](https://github.com/tyler-romero/nanogpt-speedrun/commit/b7bb93fd988d73a55184c553f0020feec1454340)), the new run time is **7.51 hours**. This run was more data-efficient than the baseline, requiring only 5.07B tokens. However, the tokens/second increased, likely due to the larger batch size (more gradient accumulation steps which tends to translate to lower throughput) and the architectural changes, such as the inclusion of RoPE. Once I have a shorter run time, I will be able to tune more effectively and see if I can remove gradient accumulation.

![Section 2.1 loss plot](/assets/img/2p1_loss_plot.png)

### 2.2 Muon Optimizer
The [Muon Optimizer](https://kellerjordan.github.io/posts/muon/) is a new optimizer developed with and for the NanoGPT speedrun by Jordan et al. It is a variant of SGD with Momentum that applies a postprocessing step to the gradient updates to approximately orthogonalize each update matrix. Muon has [some](https://kellerjordan.github.io/posts/muon/#why-is-it-good-to-orthogonalize-the-update) [connections](https://x.com/leloykun/status/1846842883967692926) to approximate second-order optimizers[^steepest] like [Shampoo](https://arxiv.org/abs/1802.09568).

[^steepest]: But are these approximate second-order methods actually second-order? [New research](https://arxiv.org/abs/2409.20325v1) suggests that methods like Shampoo and Adam can be viewed as variants of steepest descent under specific norms, and thus are actually first-order methods.

I highly recommend reading the original [Muon blog post](https://kellerjordan.github.io/posts/muon/) for more details, as well as checking out the optimizer comparison for GPT-2 speedrunning that Keller Jordan put to gether [here](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/102924_Optimizers).

Muon works on square matrices, so it is not a drop-in replacement for AdamW. However it can be used to optimize all of the hidden layers of our GPT-2 model. The output layer and the token embeddings will still be optimized with AdamW.

Just like on the 8xH100 leaderboard, we observe a massive speedup when switching to Muon. The new run time is **4.53 hours**, requiring only 3.04B tokens. The tokens/second is also very similar to the previous run, which is a good sign that we are not losing throughput by switching optimizers.

![Section 2.2 loss plot](/assets/img/2p2_loss_plot.png)

### 2.3 Dataloading Tweaks
As we have improved our data efficiency via architecture tweaks and an optimizer change, our training throughput has dropped from 221k tokens/second to 187k tokens/second. That is a ~15% drop in throughput. Recovering most of that throughput could provide a significant improvement to our run time. An obvious place to start is with our dataloading and gradient accumulation logic.

Up until now, we have loaded a full-batch of data on each device and then split that full batch into smaller chunks (micro-batches) for each gradient accumulation step (recall that we are doing 8 accumulation steps per gradient update). We can instead make a minor tweak to our logic to load only the next micro-batch at each step of the dataloader, and then step the dataloader for each gradient accumulation step.

We also increase our torch version from `2.5` to `2.6` (which was recently released), and, in accordance with the [new official rules](https://github.com/KellerJordan/modded-nanogpt?tab=readme-ov-file#timing-change-after-record-21) designated on 2025/02/01, we have removed the use of `torch._inductor.config.coordinate_descent_tuning`.

These tweak brings our throughput back up to 216k tokens/second. In order to make runs more consistently hit the 3.28 validation loss target[^variance], we have also slightly increased the total number of training steps, so now 3.31B tokens are consumed. The new run time is **4.26 hours**, and the changes can be found at [`d59944d`](https://github.com/tyler-romero/nanogpt-speedrun/commit/d59944dbe8535fea8ea107d9a6fb133de5346de5).

[^variance]: Note that there is some variance in the amount of time it takes for a speedrun candidate to run. For a speedrun to be an official record, it must attain a *mean* validation loss of less than 3.28. I have been a bit lax about this so far because the time difference between runs has been large, and variance relatively small.

![Section 2.3 loss plot](/assets/img/2p3_loss_plot.png)

At this point, we code that can train GPT-2 almost twice as fast as the baseline.


## References
<textarea id="bibtex_input" style="display:none;">
@misc{modded_nanogpt_2024,
  author       = {Keller Jordan and Jeremy Bernstein and Brendan Rappazzo and
                  @fernbear.bsky.social and Boza Vlado and You Jiacheng and
                  Franz Cesista and Braden Koszarsky and @Grad62304977},
  title        = {modded-nanogpt: Speedrunning the NanoGPT baseline},
  year         = {2024},
  url          = {https://github.com/KellerJordan/modded-nanogpt},
  note = {GitHub repository}
}
@software{hlb-gpt_2024,
  author={Fern},
  month={3},
  year = {2024},
  title={hlb-gpt},
  url={https://github.com/tysam-code/hlb-gpt},
  version = {0.4.0},
  note = {GitHub repository}
}
@misc{su2023roformerenhancedtransformerrotary,
      title={RoFormer: Enhanced Transformer with Rotary Position Embedding},
      author={Jianlin Su and Yu Lu and Shengfeng Pan and Ahmed Murtadha and Bo Wen and Yunfeng Liu},
      year={2023},
      eprint={2104.09864},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2104.09864},
}
@misc{so2022primersearchingefficienttransformers,
      title={Primer: Searching for Efficient Transformers for Language Modeling},
      author={David R. So and Wojciech Mańke and Hanxiao Liu and Zihang Dai and Noam Shazeer and Quoc V. Le},
      year={2022},
      eprint={2109.08668},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2109.08668},
}
@misc{hagele2024scalinglawscomputeoptimaltraining,
      title={Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations},
      author={Alexander Hägele and Elie Bakouch and Atli Kosson and Loubna Ben Allal and Leandro Von Werra and Martin Jaggi},
      year={2024},
      eprint={2405.18392},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2405.18392},
}
@misc{hoffmann2022trainingcomputeoptimallargelanguage,
      title={Training Compute-Optimal Large Language Models},
      author={Jordan Hoffmann and Sebastian Borgeaud and Arthur Mensch and Elena Buchatskaya and Trevor Cai and Eliza Rutherford and Diego de Las Casas and Lisa Anne Hendricks and Johannes Welbl and Aidan Clark and Tom Hennigan and Eric Noland and Katie Millican and George van den Driessche and Bogdan Damoc and Aurelia Guy and Simon Osindero and Karen Simonyan and Erich Elsen and Jack W. Rae and Oriol Vinyals and Laurent Sifre},
      year={2022},
      eprint={2203.15556},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2203.15556},
}
@misc{jordan2024muon,
  author       = {Keller Jordan and Yuchen Jin and Vlado Boza and Jiacheng You and
                  Franz Cesista and Laker Newhouse and Jeremy Bernstein},
  title        = {Muon: An optimizer for hidden layers in neural networks},
  year         = {2024},
  url          = {https://web.archive.org/web/20250122060345/https://kellerjordan.github.io/posts/muon/}
}
@misc{gupta2018shampoopreconditionedstochastictensor,
      title={Shampoo: Preconditioned Stochastic Tensor Optimization},
      author={Vineet Gupta and Tomer Koren and Yoram Singer},
      year={2018},
      eprint={1802.09568},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1802.09568},
}
@misc{bernstein2024oldoptimizernewnorm,
      title={Old Optimizer, New Norm: An Anthology},
      author={Jeremy Bernstein and Laker Newhouse},
      year={2024},
      eprint={2409.20325},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.20325},
}
</textarea>
<div id="bibtex_display"></div>