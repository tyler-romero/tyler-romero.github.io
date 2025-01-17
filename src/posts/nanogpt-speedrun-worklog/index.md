---
title: NanoGPT Speedrun Living Worklog
subtitle: How fast can I train GPT-2 on two RTX 4090 GPUs?
date: 2025-01-16T00:00:00-08:00
blurb: How fast can I train GPT-2 on two RTX 4090 GPUs? This is a living worklog of my progress.
tags: ["post", "llm", "gpt2", "speedrun", "nanogpt", "worklog"]
---

I've seen [some](https://x.com/kellerjordan0/status/1859331370268623321) [really](https://x.com/kellerjordan0/status/1842300916864844014) [awesome](https://x.com/kellerjordan0/status/1876048851158880624) [GPT-2](https://x.com/hi_tysam/status/1879687807678959729) speedrun results from people like [Keller Jordan](https://x.com/kellerjordan0), [Fern](https://x.com/hi_tysam), [Braden Koszarsky](https://x.com/KoszarskyB), and others. I got a little inspired and wanted to see how fast I could train GPT-2 on my own hardware.

Technically, [the NanoGPT speedrun](https://x.com/kellerjordan0/status/1798863559243513937) is to train a neural network to 3.28 validation loss on FineWeb as fast as possible on an **8xH100** machine. [Keller Jordan maintains a leaderboard here](https://github.com/KellerJordan/modded-nanogpt?tab=readme-ov-file#world-record-history). At the time of writing (Jan 16, 2025), the record is 3.14 minutes (!).

I have access to **2xRTX 4090 GPUs** and I want to see how fast I can train GPT-2 on them by following the same rules as the NanoGPT speedrun. If I see some success, I may try to transfer my methods to an 8xH100 node for comparison with the main leaderboard.

I'll be documenting my progress here and updating this post as I go. Code can be found in [this GitHub repo](https://github.com/tyler-romero/nanogpt-speedrun).

## Progress so far
| #    | Record time | Training Tokens | Description      | Date       | Commit                                                                                                      | Log                                                                                                              |
| :--- | :---------- | :-------------- | :--------------- | :--------- | :---------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------- |
| 1    | 8.13 hours  | 6.44e+09        | Initial baseline | 2025-01-16 | [b3c32f8](https://github.com/tyler-romero/nanogpt-speedrun/commit/b3c32f8937c1f4655c5eb9607970e03e351a6c08) | [here](https://github.com/tyler-romero/nanogpt-speedrun/blob/main/logs/4c627c0d-029c-4f8a-bd18-40f99b43b22e.txt) |

## 1. Initial setup and baseline

Part of the goal of this project is for me to learn as I go, so I am going to start at the beginning - with with Andrej Karpathy's [PyTorch GPT-2 trainer](https://github.com/karpathy/llm.c/blob/7b929300217ff1a974b63791a228928b39b26409/train_gpt2.py) from [llm.c](https://github.com/karpathy/llm.c). This is the script that Keller Jordan used for [his initial baseline](https://github.com/KellerJordan/modded-nanogpt/tree/master?tab=readme-ov-file#modded-nanogpt). This trainer is very similar to the NanoGPT trainer with some minor modifications / simplifications (such as no dropout).

I have upstreamed some QOL improvements and basic tweaks to the training script from Keller's fork, but have not changed any of the core training / modeling logic. Specifically:
1. Implemented gradient accumulation so that my 2x24GB GPUs simulate the training experience of a 8xH100 machine.
2. Increased learning rate to 0.0015 and halved the batch size (total batch size is 262144 - that is bs of `32/device * 2 devices * 1024 sequence length * 4 gradient accum steps`).
3. Improved learning rate schedule (linear warmup then linear decay).
4. Removed all affine scale/bias parameters and switched to RMSNorm.
5. Padded the vocab size from 50257 to 50304 to make it a multiple of 128 (for better tensor core utilization).

Additionally, I added `wandb` logging for easy tracking of training progress - optimistically I may need to remove this one day as it slightly increases step time.

Commit with the initial setup is here: [`b3c32f8`](https://github.com/tyler-romero/nanogpt-speedrun/blob/main/logs/4c627c0d-029c-4f8a-bd18-40f99b43b22e.txt).

The baseline run time on my 2xRTX 4090 setup is **8.13 hours**.

<!-- TODO: plot -->

<!-- ## 2. Implementing major improvements from the 8xH100 leaderboard

Waiting 8 hours for a result, so I'm going to begin by implementing some of the notable improvements from the 8xH100 leaderboard. I'll start with the most impactful/easiest changes first:
1. FlexAttention (30.2% speedup)
2. Muon Optimizer (29% speedup)
3. Architectural changes (31.8% speedup, then 24% speedup)
4. Untied embeddings and lm_head (10% speedup)

### 2.1 Muon Optimizer -->