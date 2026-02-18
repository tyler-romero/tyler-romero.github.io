---
title: NanoGPT Speedrun Living Worklog
subtitle: How fast can I train GPT-2 on two RTX 4090 GPUs?
date: 2025-03-08T00:00:00-08:00
blurb: How fast can I train GPT-2 on two RTX 4090 GPUs? This is a living worklog of my progress.
tags: ["post", "llm", "gpt2", "speedrun", "nanogpt", "worklog", "muon"]
math: true
code: true
bibtex: true
---

I've seen [some](https://x.com/kellerjordan0/status/1859331370268623321) [really](https://x.com/kellerjordan0/status/1842300916864844014) [awesome](https://x.com/kellerjordan0/status/1876048851158880624) [GPT-2](https://x.com/hi_tysam/status/1879687807678959729) speedrun results from people like [Keller Jordan](https://x.com/kellerjordan0), [Fern](https://x.com/hi_tysam), [Braden Koszarsky](https://x.com/KoszarskyB), and others. I got a little inspired and wanted to see how fast I could train GPT-2 on my own hardware.

Technically, [the NanoGPT speedrun](https://x.com/kellerjordan0/status/1798863559243513937) is to train a neural network to 3.28 validation loss on FineWeb as fast as possible on an 8xH100 node. [Keller Jordan maintains a leaderboard here](https://github.com/KellerJordan/modded-nanogpt?tab=readme-ov-file#world-record-history). At the time of writing (Jan 16, 2025), the record is 3.14 minutes (!).

I have access to **2xRTX 4090 GPUs** and I want to see how fast I can train GPT-2 on them by following the same rules as the NanoGPT speedrun. If I see some success, I may try to transfer my methods to an 8xH100 node for comparison with the main leaderboard.

I'll be documenting my progress here and updating this post as I go. Code can be found in [this GitHub repo](https://github.com/tyler-romero/nanogpt-speedrun).

## Progress so far
| #                                                      | Description              | Record time | Training Tokens | Tokens/Second | Date       | Commit                                                                                                      | Log                                                                                                              |
| :----------------------------------------------------- | :----------------------- | :---------- | :-------------- | :------------ | :--------- | :---------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------- |
| [1](#1-initial-setup-and-baseline)                     | Initial baseline         | 8.13 hours  | 6.44B           | 221k          | 2025/01/16 | [b3c32f8](https://github.com/tyler-romero/nanogpt-speedrun/commit/b3c32f8937c1f4655c5eb9607970e03e351a6c08) | [here](https://github.com/tyler-romero/nanogpt-speedrun/blob/main/logs/4c627c0d-029c-4f8a-bd18-40f99b43b22e.txt) |
| [2.1](#21-architectural-changes-and-training-tweaks)   | Architectural changes    | 7.51 hours  | 5.07B           | 188k          | 2025/01/18 | [b7bb93f](https://github.com/tyler-romero/nanogpt-speedrun/commit/b7bb93fd988d73a55184c553f0020feec1454340) | [here](https://github.com/tyler-romero/nanogpt-speedrun/blob/main/logs/14fcdb07-443d-4d1c-b307-061bc4bd2cd6.txt) |
| [2.2](#22-muon-optimizer)                              | Muon optimizer           | 4.53 hours  | 3.04B           | 187k          | 2025/01/23 | [b91c2c0](https://github.com/tyler-romero/nanogpt-speedrun/commit/b91c2c00673b125944abde277dd5ef3dc141284d) | [here](https://github.com/tyler-romero/nanogpt-speedrun/blob/main/logs/59951c17-fbe5-4577-a1bc-6dc0c1802d2e.txt) |
| [2.3](#23-dataloading-tweaks)                          | Dataloading tweaks       | 4.26 hours  | 3.31B           | 216k          | 2025/02/18 | [d59944d](https://github.com/tyler-romero/nanogpt-speedrun/commit/d59944dbe8535fea8ea107d9a6fb133de5346de5) | [here](https://github.com/tyler-romero/nanogpt-speedrun/blob/main/logs/08047f73-cb01-4f47-a901-de901b2a6b6e.txt) |
| [2.4](#24-logit-soft-capping)                          | Logit Soft-capping at 30 | 4.01 hours  | 3.15B           | 218k          | 2025/02/23 | [12eab44](https://github.com/tyler-romero/nanogpt-speedrun/commit/12eab44ca1bce8783a3b4d43bfef357eff1a652e) | [here](https://github.com/tyler-romero/nanogpt-speedrun/blob/main/logs/2dbf7fa6-561c-49bc-8aae-665fefdd9a44.txt) |
| [3](#3-longer-training-and-evaluation-sequence-length) | Longer Sequence Length   | 2.55 hours  | 1.88B           | 205k          | 2025/03/03 | [d982ed5](https://github.com/tyler-romero/nanogpt-speedrun/commit/d982ed5900922e43a266c5d671b88f36efe72aaf) | [here](https://github.com/tyler-romero/nanogpt-speedrun/blob/main/logs/cf1ef5f9-9f79-4798-9360-2b174d8eb25f.txt) |

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
4. Logit Softcapping

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

I highly recommend reading the original [Muon blog post](https://kellerjordan.github.io/posts/muon/) for more details, as well as checking out the optimizer comparison for GPT-2 speedrunning that Keller Jordan put to gether [here](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/102924_Optimizers). For those interested in a more step-by-step walkthrough of Muon, check out [this excellent post](https://jeremybernste.in/writing/deriving-muon) by Jeremy Bernstein.

Muon is designed to work on *Linear* layers, so it is not quite a drop-in replacement for AdamW (e.g. it isn't meant to optimize Embedding layers). However it can be used to optimize all of the hidden layers of our GPT-2 model. The output `lm_head` layer and the token embeddings will still be optimized with AdamW.

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

### 2.4 Logit Soft-capping
Logit soft-capping is a technique popularized by [Gemma 2](https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf) and initially used to improve the NanoGPT speedrun by [@Grad62304977](https://x.com/Grad62304977).

Soft-capping is essentially a smooth and differentiable version of clipping[^softcap]:
$$
\text{softcap(x, cap)} = \text{cap} \cdot \tanh\left(\frac{\text{x}}{\text{cap}}\right)
$$

[^softcap]: {-} Soft-capping vs Clipping at ±5: ![Soft-capping](/assets/img/softcap.png)

Logit soft-capping prevents logits from growing excessively large by scaling them to a fixed range, which seems to help improve training dynamics. One could argue that this is imposing an inductive bias - and since we're in a relatively small model/low data regime that this is helpful.

After implementing logit soft-capping with a cap of 30 (and doing some learning-rate tuning), the new run time is **4.01 hours**, requiring 3.15B tokens (commit [`12eab44`](https://github.com/tyler-romero/nanogpt-speedrun/commit/12eab44ca1bce8783a3b4d43bfef357eff1a652e)). Throughput remained steady at ~218k tokens/second.

![Section 2.4 loss plot](/assets/img/2p4_loss_plot.png)

## 3 Longer Training and Evaluation Sequence Length

So far, we've been training and evaluating on sequences of 1024 tokens. We also haven't been particularly clever about how those sequences are processed. At each step, we simply load the next 1024 tokens into an element of the batch without regard for where the document starts or stops. That means much of the time we are starting in the middle of a document and cutting that document off before it reaches its end. We are also attending to tokens *across documents* since we're just using a simple causal mask.

Cutting off documents in the middle is an especially large issue. See this plot of average loss vs sequence position:
![Average Loss vs Sequence Position](/assets/img/avg_loss_vs_seq_position.png)

Notice how the first twenty-five or so positions have a much higher average loss than the later positions. This is because at the beginning of the sequence the LLM has much less information with which to make informed predictions about the next token in the sequence. We want to avoid needlessly restarting documents/sequences in order to avoid this loss penalty!

A natural question to ask at this point is: how long are sequences in our dataset, on average?
![Sequence Length CDF Plot](/assets/img/sequence_length_cdf.png)

The data reveals that approximately 20% of documents exceed our current 1024 token sequence length. By increasing the sequence length to >=8192 tokens, we can accommodate virtually all documents in our dataset without truncation.

To address the issues identified above, we'll implement two key improvements. First, we'll extend our sequence length to minimize document splitting across sequence boundaries. Taking this approach to its logical conclusion, we'll eliminate the traditional batch dimension entirely and instead maximize sequence length (effectively using a "batch size" of 1 that contains multiple concatenated documents). Second, we'll implement sophisticated attention masking that prevents cross-document attention while simultaneously leveraging the computational efficiency of sparse attention patterns.

Fortunately, [FlexAttention](https://pytorch.org/blog/flexattention/) provides an elegant solution that maintains the performance benefits of [FlashAttention](https://huggingface.co/docs/text-generation-inference/en/conceptual/flash_attention) while enabling these improvements. One of FlexAttention's primary strengths is its ability to efficiently handle sparse, custom attention masks, making it ideal for our use case.

To implement FlexAttention, we need to define an appropriate attention mask that handles our specific requirements:
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

In order to build intuition about the individual component masks, we visualize them below:
![Causal, Document, Sliding Window Attention Masks](/assets/img/attention_masks.svg)

When combined with the `and_masks` function, these three masks[^redundant] work together to create an efficient attention pattern that respects document boundaries, maintains causality, and limits computational overhead for long sequences.

[^redundant]: Note that the causal mask is actually redundant to the sliding window mask, as the sliding window mask already ensures that tokens can only attend to previous tokens in the sequence. The causal mask is included here for clarity.

After incorporating FlexAttention with these masks, and increasing our sequence length to 32768 tokens, we observe a massive speedup[^hack]. The new run time is **2.55 hours**, requiring only 1.88B tokens (a huge data-efficiency improvement). Our throughput dropped slightly to ~205k tokens/second. See commit [`d982ed5`](https://github.com/tyler-romero/nanogpt-speedrun/commit/d982ed5900922e43a266c5d671b88f36efe72aaf) for the full details.

[^hack]: This speedup is a bit of a hack against the target metric. Supporting longer sequences is a straightforward way to drop the loss on the validation set, but is unlikely to provide a meaningful improvement to the overall performance of the model on practical benchmarks.

![Section 3 loss plot](/assets/img/3_loss_plot.png)

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
<div id="bibtex_display"></div>