---
title: NanoGPT Speedrun Worklog
subtitle: Experiments in training GPT-2 to 3.28 validation loss on two RTX 4090 GPUs
date: 2025-03-08T00:00:00-08:00
blurb: A worklog of cutting GPT-2 training time from 8.13 hours to 2.55 hours on two RTX 4090 GPUs.
tags: ["post", "llm", "gpt2", "speedrun", "nanogpt", "worklog", "muon"]
math: true
code: true
hero_image: /assets/img/golden-gardens.png
featured_image: /assets/img/golden-gardens-social.jpg
---

I saw [some](https://x.com/kellerjordan0/status/1859331370268623321) [really](https://x.com/kellerjordan0/status/1842300916864844014) [awesome](https://x.com/kellerjordan0/status/1876048851158880624) [GPT-2](https://x.com/hi_tysam/status/1879687807678959729) speedrun results from people like [Keller Jordan](https://x.com/kellerjordan0), [Fern](https://x.com/hi_tysam), [Braden Koszarsky](https://x.com/KoszarskyB), and others. I got a little inspired and decided to see how fast I could train GPT-2 on my own hardware.

Technically, [the NanoGPT speedrun](https://x.com/kellerjordan0/status/1798863559243513937) is to train a neural network to 3.28 validation loss on FineWeb as fast as possible on an 8xH100 node. [Keller Jordan maintains a leaderboard here](https://github.com/KellerJordan/modded-nanogpt?tab=readme-ov-file#world-record-history). When I started this experiment on January 16, 2025, the record was 3.14 minutes (!).

I had access to **2xRTX 4090 GPUs**, so I followed the same rules on my own hardware. Over six iterations, I reduced the training time from **8.13 hours to 2.55 hours**. This worklog records the changes that produced that result; the code and run logs are available in [the project repository](https://github.com/tyler-romero/nanogpt-speedrun).

## Results
| #                                                      | Description              | Record time | Training Tokens | Tokens/Second | Date       | Commit                                                                                                      | Log                                                                                                              |
| :----------------------------------------------------- | :----------------------- | :---------- | :-------------- | :------------ | :--------- | :---------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------- |
| [1](#1-initial-setup-and-baseline)                     | Initial baseline         | 8.13 hours  | 6.44B           | 221k          | 2025/01/16 | [b3c32f8](https://github.com/tyler-romero/nanogpt-speedrun/commit/b3c32f8937c1f4655c5eb9607970e03e351a6c08) | [here](https://github.com/tyler-romero/nanogpt-speedrun/blob/main/logs/4c627c0d-029c-4f8a-bd18-40f99b43b22e.txt) |
| [2.1](#21-architectural-changes-and-training-tweaks)   | Architectural changes    | 7.51 hours  | 5.07B           | 188k          | 2025/01/18 | [b7bb93f](https://github.com/tyler-romero/nanogpt-speedrun/commit/b7bb93fd988d73a55184c553f0020feec1454340) | [here](https://github.com/tyler-romero/nanogpt-speedrun/blob/main/logs/14fcdb07-443d-4d1c-b307-061bc4bd2cd6.txt) |
| [2.2](#22-muon-optimizer)                              | Muon optimizer           | 4.53 hours  | 3.04B           | 187k          | 2025/01/23 | [b91c2c0](https://github.com/tyler-romero/nanogpt-speedrun/commit/b91c2c00673b125944abde277dd5ef3dc141284d) | [here](https://github.com/tyler-romero/nanogpt-speedrun/blob/main/logs/59951c17-fbe5-4577-a1bc-6dc0c1802d2e.txt) |
| [2.3](#23-dataloading-tweaks)                          | Dataloading tweaks       | 4.26 hours  | 3.31B           | 216k          | 2025/02/18 | [d59944d](https://github.com/tyler-romero/nanogpt-speedrun/commit/d59944dbe8535fea8ea107d9a6fb133de5346de5) | [here](https://github.com/tyler-romero/nanogpt-speedrun/blob/main/logs/08047f73-cb01-4f47-a901-de901b2a6b6e.txt) |
| [2.4](#24-logit-soft-capping)                          | Logit Soft-capping at 30 | 4.01 hours  | 3.15B           | 218k          | 2025/02/23 | [12eab44](https://github.com/tyler-romero/nanogpt-speedrun/commit/12eab44ca1bce8783a3b4d43bfef357eff1a652e) | [here](https://github.com/tyler-romero/nanogpt-speedrun/blob/main/logs/2dbf7fa6-561c-49bc-8aae-665fefdd9a44.txt) |
| [3](#3-longer-training-and-evaluation-sequence-length) | Longer Sequence Length   | 2.55 hours  | 1.88B           | 205k          | 2025/03/03 | [d982ed5](https://github.com/tyler-romero/nanogpt-speedrun/commit/d982ed5900922e43a266c5d671b88f36efe72aaf) | [here](https://github.com/tyler-romero/nanogpt-speedrun/blob/main/logs/cf1ef5f9-9f79-4798-9360-2b174d8eb25f.txt) |

## 1. Initial setup and baseline

Part of the goal of this project was to learn as I went, so I started at the beginning—with Andrej Karpathy's [PyTorch GPT-2 trainer](https://github.com/karpathy/llm.c/blob/7b929300217ff1a974b63791a228928b39b26409/train_gpt2.py) from [llm.c](https://github.com/karpathy/llm.c). This is the script that Keller Jordan used for [his initial baseline](https://github.com/KellerJordan/modded-nanogpt/tree/master?tab=readme-ov-file#modded-nanogpt). The trainer is very similar to NanoGPT with some minor modifications and simplifications, such as removing dropout.

I upstreamed some QOL improvements and basic tweaks to the training script from Keller's fork, but did not change any of the core training or modeling logic. Specifically:
1. Implemented gradient accumulation so that my 2x24GB GPUs simulate the training experience of an 8xH100 machine.
2. Increased learning rate to 0.0015 and halved the batch size (total batch size is 262144 - that is bs of `32/device * 2 devices * 1024 sequence length * 4 gradient accum steps`).
3. Improved learning rate schedule (linear warmup then linear decay).
4. Removed all affine scale/bias parameters and switched to RMSNorm.
5. Padded the vocab size from 50257 to 50304 to make it a multiple of 128 (for better tensor core utilization).
6. Used PyTorch 2.5.1 (the switch from 2.4 to 2.5 gave ~9% speedup on the 8xH100 leaderboard).

Additionally, I added `wandb` logging to make it easier to track the training runs.

Commit with the initial setup is here: [`b3c32f8`](https://github.com/tyler-romero/nanogpt-speedrun/blob/main/logs/4c627c0d-029c-4f8a-bd18-40f99b43b22e.txt).

The baseline run time on my 2xRTX 4090 setup was **8.13 hours**.

## 2. Implementing major improvements from the 8xH100 leaderboard

Waiting 8 hours for a result was too slow for effective experimentation, so I began by implementing some of the notable improvements from the 8xH100 leaderboard. I started with the most impactful and easiest changes:
1. Architectural changes and training tweaks
2. Muon optimizer
3. Dataloading tweaks
4. Logit Softcapping

### 2.1 Architectural changes and training tweaks
There are some basic architectural changes and modernizations that can be made to the model that will speed up training. These changes are general improvements to the transformer decoder architecture that have been generally adopted since the original GPT-2 paper. The changes are:
1. [RoPE (Rotary Positional Embeddings)](https://arxiv.org/abs/2104.09864). There are [many](https://www.jitx.io/posts/rope-embeddings) [good](https://blog.eleuther.ai/rotary-embeddings/) explanations of RoPE out there so I won't go into detail here.
2. [ReLU^2 Activation](https://arxiv.org/pdf/2109.08668)[^relu2]. Many activations that are better than GeLU have been proposed since GPT-2. ReLU^2 is a simple one that has been shown to be effective in decreasing training time required to reach a certain validation loss.
3. No gradient clipping. Gradient clipping can help stabilize training, but it also slows down training. Since this was a speedrun, I removed it. This also eliminated a hyperparameter that needed to be tuned.
4. [Trapezoidal learning rate schedule](https://arxiv.org/abs/2405.18392). While cosine learning rate schedules are the de-facto standard, they can be difficult to work with since changing the number of training steps changes the entire schedule. Trapezoidal learning rate schedules are often easier to reason about / tune around, and they have been shown to match the performance of cosine schedules.

[^relu2]: ReLU^2 activation function. ![Relu Activation plot](/assets/img/relu2.png)

In addition, I tuned the learning rate and batch size.

Once again, many of these changes were [downstreamed](https://en.wikipedia.org/wiki/Downstream_(software_development)) from the [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) repository / 8xH100 speedrun. It wasn't efficient to reinvent the wheel, and I wanted to reduce training time quickly before doing more targeted experiments.

After implementing these changes (commit [`b7bb93f`](https://github.com/tyler-romero/nanogpt-speedrun/commit/b7bb93fd988d73a55184c553f0020feec1454340)), the new run time was **7.51 hours**. This run was more data-efficient than the baseline, requiring only 5.07B tokens. However, the tokens/second decreased, likely due to the larger batch size (more gradient accumulation steps tends to translate to lower throughput) and architectural changes such as the inclusion of RoPE. The shorter run time made further experimentation more practical.

![Section 2.1 loss plot](/assets/img/2p1_loss_plot.png)

### 2.2 Muon Optimizer
The [Muon Optimizer](https://kellerjordan.github.io/posts/muon/) was developed with and for the NanoGPT speedrun by Jordan et al. It is a variant of SGD with Momentum that applies a postprocessing step to the gradient updates to approximately orthogonalize each update matrix. Muon has [some](https://kellerjordan.github.io/posts/muon/#why-is-it-good-to-orthogonalize-the-update) [connections](https://x.com/leloykun/status/1846842883967692926) to approximate second-order optimizers[^steepest] like [Shampoo](https://arxiv.org/abs/1802.09568).

[^steepest]: But are these approximate second-order methods actually second-order? [New research](https://arxiv.org/abs/2409.20325v1) suggests that methods like Shampoo and Adam can be viewed as variants of steepest descent under specific norms, and thus are actually first-order methods.

I highly recommend reading the original [Muon blog post](https://kellerjordan.github.io/posts/muon/) for more details, as well as checking out the optimizer comparison for GPT-2 speedrunning that Keller Jordan put together [here](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/102924_Optimizers). For those interested in a more step-by-step walkthrough of Muon, check out [this excellent post](https://jeremybernste.in/writing/deriving-muon) by Jeremy Bernstein.

Muon is designed to work on *Linear* layers, so it is not quite a drop-in replacement for AdamW (e.g. it isn't meant to optimize Embedding layers). However, it can be used to optimize all of the hidden layers of our GPT-2 model. The output `lm_head` layer and token embeddings were still optimized with AdamW.

Just like on the 8xH100 leaderboard, I observed a massive speedup after switching to Muon. The new run time was **4.53 hours**, requiring only 3.04B tokens. The tokens/second remained very similar to the previous run, indicating that the optimizer change did not sacrifice throughput.

![Section 2.2 loss plot](/assets/img/2p2_loss_plot.png)

### 2.3 Dataloading Tweaks
The architecture and optimizer changes improved data efficiency, but training throughput dropped from 221k tokens/second to 187k tokens/second—a decrease of roughly 15%. Recovering that throughput offered another path to a shorter run time, so I turned to the dataloading and gradient accumulation logic.

Up to this point, I had loaded a full batch of data on each device and then split it into smaller chunks (micro-batches) for each gradient accumulation step. I changed the logic to load only the next micro-batch and advance the dataloader for each accumulation step.

I also upgraded PyTorch from `2.5` to `2.6`, which had recently been released, and removed `torch._inductor.config.coordinate_descent_tuning` in accordance with the [official rules introduced on February 1, 2025](https://github.com/KellerJordan/modded-nanogpt?tab=readme-ov-file#timing-change-after-record-21).

These tweaks brought throughput back up to 216k tokens/second. To make runs more consistently hit the 3.28 validation loss target[^variance], I also slightly increased the total number of training steps, bringing consumption to 3.31B tokens. The new run time was **4.26 hours**, and the changes can be found at [`d59944d`](https://github.com/tyler-romero/nanogpt-speedrun/commit/d59944dbe8535fea8ea107d9a6fb133de5346de5).

[^variance]: There is some variance in how long a speedrun candidate takes to reach the target. An official record must attain a *mean* validation loss below 3.28. I treated this somewhat loosely in the early experiments because the differences between runs were much larger than the observed variance.

![Section 2.3 loss plot](/assets/img/2p3_loss_plot.png)

At this point, the training time was almost half the baseline.

### 2.4 Logit Soft-capping
Logit soft-capping is a technique popularized by [Gemma 2](https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf) and initially used to improve the NanoGPT speedrun by [@Grad62304977](https://x.com/Grad62304977).

Soft-capping is essentially a smooth and differentiable version of clipping[^softcap]:
\[
\text{softcap(x, cap)} = \text{cap} \cdot \tanh\left(\frac{\text{x}}{\text{cap}}\right)
\]

[^softcap]: {-} Soft-capping vs Clipping at ±5: ![Soft-capping](/assets/img/softcap.png)

Logit soft-capping prevents logits from growing excessively large by scaling them to a fixed range, which seems to help improve training dynamics. One could argue that this imposes an inductive bias—and in this relatively small-model, low-data regime, that bias appeared to be helpful.

After implementing logit soft-capping with a cap of 30 (and doing some learning-rate tuning), the new run time was **4.01 hours**, requiring 3.15B tokens (commit [`12eab44`](https://github.com/tyler-romero/nanogpt-speedrun/commit/12eab44ca1bce8783a3b4d43bfef357eff1a652e)). Throughput remained steady at ~218k tokens/second.

![Section 2.4 loss plot](/assets/img/2p4_loss_plot.png)

## 3. Longer Training and Evaluation Sequence Length

Up to this point, I had trained and evaluated on sequences of 1024 tokens without being particularly clever about how those sequences were constructed. At each step, the dataloader simply placed the next 1024 tokens into an element of the batch without regard for document boundaries. That meant frequently starting in the middle of a document, cutting it off before its end, and attending to tokens *across documents* through a simple causal mask.

Cutting off documents in the middle is an especially large issue. See this plot of average loss vs sequence position:
![Average Loss vs Sequence Position](/assets/img/avg_loss_vs_seq_position.png)

Notice how the first twenty-five or so positions have a much higher average loss than later positions. At the beginning of a sequence, the model has much less context with which to predict the next token. Avoiding needless sequence restarts offered a way to reduce this loss penalty.

A natural question to ask at this point is: how long are sequences in our dataset, on average?
![Sequence Length CDF Plot](/assets/img/sequence_length_cdf.png)

The data revealed that approximately 20% of documents exceeded the 1024-token sequence length. Increasing the sequence length to >=8192 tokens would accommodate virtually all documents in the dataset without truncation.

To address these issues, I made two key changes. First, I extended the sequence length to minimize document splitting across sequence boundaries. Taking this approach to its logical conclusion, I eliminated the traditional batch dimension and effectively used a "batch size" of 1 containing multiple concatenated documents. Second, I added attention masking that prevented cross-document attention while retaining the computational efficiency of sparse attention patterns.

Fortunately, [FlexAttention](https://pytorch.org/blog/flexattention/) provides an elegant solution that maintains the performance benefits of [FlashAttention](https://huggingface.co/docs/text-generation-inference/en/conceptual/flash_attention) while enabling these improvements. One of FlexAttention's primary strengths is its ability to efficiently handle sparse, custom attention masks, making it ideal for our use case.

To implement FlexAttention, I defined a mask that handled the specific requirements of the dataset:
```python
def make_attn_mask(idx, eot_token, window_size=1024):
    # Create a causal mask (only attend to past tokens)
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    # Track document boundaries using end-of-text tokens
    documents = (idx == eot_token).cumsum(dim=1)

    # Only allow attention within the same document
    def document_mask(b, h, q_idx, kv_idx):
        return documents[b, q_idx] == documents[b, kv_idx]

    # Limit attention to an N-token window for efficiency
    def sliding_window_mask(b, h, q_idx, kv_idx):
        return q_idx - kv_idx <= window_size

    return and_masks(document_mask, causal_mask, sliding_window_mask)
```

Let's break down each mask:

1. **Causal Mask**: Standard in autoregressive language modeling. Ensures that tokens can only attend to previous tokens in the sequence, preventing information leakage from future tokens.

2. **Document Mask**: This restricts attention to tokens within the same document. By tracking document boundaries using end-of-text tokens, we prevent tokens from attending across different documents, which helps the model maintain coherent context within a single document.

3. **Sliding Window Mask**: This limits attention to a fixed window of tokens before the current position. This approach balances efficiency with context retention with a clear tradeoff: smaller windows are more efficient but may miss long-range dependencies, while larger windows capture more context at the expense of resources.

The individual component masks are visualized below:
![Causal, Document, Sliding Window Attention Masks](/assets/img/attention_masks.svg)

When combined with the `and_masks` function, these three masks[^redundant] work together to create an efficient attention pattern that respects document boundaries, maintains causality, and limits computational overhead for long sequences.

[^redundant]: Note that the causal mask is actually redundant to the sliding window mask, as the sliding window mask already ensures that tokens can only attend to previous tokens in the sequence. The causal mask is included here for clarity.

After incorporating FlexAttention with these masks and increasing the sequence length to 32768 tokens, I observed another massive speedup[^hack]. The final run time was **2.55 hours**, requiring only 1.88B tokens—a large data-efficiency improvement. Throughput dropped slightly to ~205k tokens/second. See commit [`d982ed5`](https://github.com/tyler-romero/nanogpt-speedrun/commit/d982ed5900922e43a266c5d671b88f36efe72aaf) for the full details.

[^hack]: This speedup is a bit of a hack against the target metric. Supporting longer sequences is a straightforward way to drop the loss on the validation set, but is unlikely to provide a meaningful improvement to the overall performance of the model on practical benchmarks.

![Section 3 loss plot](/assets/img/3_loss_plot.png)

## Summary

The final 2.55-hour run was **3.2x faster** than the 8.13-hour baseline and used 1.88B training tokens instead of 6.44B. Muon produced the largest improvement that was independent of the evaluation setup. The final reduction came from longer, document-aware sequences and was more specific to the validation-loss target.

This post records the completed 2025 round of experiments, with 2.55 hours as the last measured result. I may revisit the speedrun in the future, either on the same GPUs or on different hardware. The numbers here are specific to the two-RTX-4090 setup and are not directly comparable to the 8xH100 leaderboard times.

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
@misc{gemmateam2024gemma2improvingopen,
      title={Gemma 2: Improving Open Language Models at a Practical Size},
      author={Gemma Team and Morgane Riviere and Shreya Pathak and Pier Giuseppe Sessa and Cassidy Hardin and Surya Bhupatiraju and Léonard Hussenot and Thomas Mesnard and Bobak Shahriari and Alexandre Ramé and Johan Ferret and Peter Liu and Pouya Tafti and Abe Friesen and Michelle Casbon and Sabela Ramos and Ravin Kumar and Charline Le Lan and Sammy Jerome and Anton Tsitsulin and Nino Vieillard and Piotr Stanczyk and Sertan Girgin and Nikola Momchev and Matt Hoffman and Shantanu Thakoor and Jean-Bastien Grill and Behnam Neyshabur and Olivier Bachem and Alanna Walton and Aliaksei Severyn and Alicia Parrish and Aliya Ahmad and Allen Hutchison and Alvin Abdagic and Amanda Carl and Amy Shen and Andy Brock and Andy Coenen and Anthony Laforge and Antonia Paterson and Ben Bastian and Bilal Piot and Bo Wu and Brandon Royal and Charlie Chen and Chintu Kumar and Chris Perry and Chris Welty and Christopher A. Choquette-Choo and Danila Sinopalnikov and David Weinberger and Dimple Vijaykumar and Dominika Rogozińska and Dustin Herbison and Elisa Bandy and Emma Wang and Eric Noland and Erica Moreira and Evan Senter and Evgenii Eltyshev and Francesco Visin and Gabriel Rasskin and Gary Wei and Glenn Cameron and Gus Martins and Hadi Hashemi and Hanna Klimczak-Plucińska and Harleen Batra and Harsh Dhand and Ivan Nardini and Jacinda Mein and Jack Zhou and James Svensson and Jeff Stanway and Jetha Chan and Jin Peng Zhou and Joana Carrasqueira and Joana Iljazi and Jocelyn Becker and Joe Fernandez and Joost van Amersfoort and Josh Gordon and Josh Lipschultz and Josh Newlan and Ju-yeong Ji and Kareem Mohamed and Kartikeya Badola and Kat Black and Katie Millican and Keelin McDonell and Kelvin Nguyen and Kiranbir Sodhia and Kish Greene and Lars Lowe Sjoesund and Lauren Usui and Laurent Sifre and Lena Heuermann and Leticia Lago and Lilly McNealus and Livio Baldini Soares and Logan Kilpatrick and Lucas Dixon and Luciano Martins and Machel Reid and Manvinder Singh and Mark Iverson and Martin Görner and Mat Velloso and Mateo Wirth and Matt Davidow and Matt Miller and Matthew Rahtz and Matthew Watson and Meg Risdal and Mehran Kazemi and Michael Moynihan and Ming Zhang and Minsuk Kahng and Minwoo Park and Mofi Rahman and Mohit Khatwani and Natalie Dao and Nenshad Bardoliwalla and Nesh Devanathan and Neta Dumai and Nilay Chauhan and Oscar Wahltinez and Pankil Botarda and Parker Barnes and Paul Barham and Paul Michel and Pengchong Jin and Petko Georgiev and Phil Culliton and Pradeep Kuppala and Ramona Comanescu and Ramona Merhej and Reena Jana and Reza Ardeshir Rokni and Rishabh Agarwal and Ryan Mullins and Samaneh Saadat and Sara Mc Carthy and Sarah Cogan and Sarah Perrin and Sébastien M. R. Arnold and Sebastian Krause and Shengyang Dai and Shruti Garg and Shruti Sheth and Sue Ronstrom and Susan Chan and Timothy Jordan and Ting Yu and Tom Eccles and Tom Hennigan and Tomas Kocisky and Tulsee Doshi and Vihan Jain and Vikas Yadav and Vilobh Meshram and Vishal Dharmadhikari and Warren Barkley and Wei Wei and Wenming Ye and Woohyun Han and Woosuk Kwon and Xiang Xu and Zhe Shen and Zhitao Gong and Zichuan Wei and Victor Cotruta and Phoebe Kirk and Anand Rao and Minh Giang and Ludovic Peran and Tris Warkentin and Eli Collins and Joelle Barral and Zoubin Ghahramani and Raia Hadsell and D. Sculley and Jeanine Banks and Anca Dragan and Slav Petrov and Oriol Vinyals and Jeff Dean and Demis Hassabis and Koray Kavukcuoglu and Clement Farabet and Elena Buchatskaya and Sebastian Borgeaud and Noah Fiedel and Armand Joulin and Kathleen Kenealy and Robert Dadashi and Alek Andreev},
      year={2024},
      eprint={2408.00118},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.00118},
}
@misc{dong2024flexattentionprogrammingmodel,
      title={Flex Attention: A Programming Model for Generating Optimized Attention Kernels},
      author={Juechu Dong and Boyuan Feng and Driss Guessous and Yanbo Liang and Horace He},
      year={2024},
      eprint={2412.05496},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.05496},
}
@misc{bernstein2025deriving,
  author = {Jeremy Bernstein},
  title = {Deriving Muon},
  url = {https://jeremybernste.in/writing/deriving-muon},
  year = {2025}
}

</textarea>
