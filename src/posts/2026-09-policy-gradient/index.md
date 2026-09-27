---
title: "Policy Gradient for LLMs, Explained Visually"
subtitle: A from-scratch derivation of REINFORCE for language models
date: 2026-09-27T00:00:00-07:00
blurb: "How language models learn from rewarded samples: a visual, from-scratch derivation of REINFORCE and group-centered baselines."
tags: ["post", "reinforcement-learning", "rl", "policy-gradient", "reinforce", "grpo", "rlvr"]
math: true
code: true
---

Most RL algorithms used to train language models, from PPO to GRPO, are elaborations of one idea: the policy gradient. This post derives it from scratch for an LLM solving a problem with a checkable answer. It follows one prompt, "What is 17 × 24?", from next-token probabilities to the gradient that makes correct answers more likely.

## Language models as policies

Given a prompt \(x\), a language model generates a completion \(y = (y_1, \dots, y_T)\) one token at a time. In RL terms, the model is a _policy_: at each step it looks at the prefix it has produced so far and outputs a distribution over the next token, from which one token is sampled.

![The prompt "What is 17 × 24? Show your work, then give a final answer." and a completion in progress, "17 × 24 = 17 × 20 + 17 × 4 = 340 +", followed by an empty slot. Below it, the model's probabilities for the next token: 68 at 0.82, the slip 58 at 0.07, and small amounts for other tokens. One token is sampled, appended, and the process repeats.](/assets/img/policy-gradient-next-token.png)

The probability of the full completion is the [product of the per-token probabilities](/posts/2024-04-dpo/#the-probability-of-a-completion):[^theta]

[^theta]: {-} \(\theta\) is the model's parameters: its weights, which training adjusts.

\[
p_\theta(y \mid x) = \prod_{t=1}^{T} p_\theta(y_t \mid x, y_{<t})
\]

Generating a completion traces one path through a tree of possible continuations.[^tree-cond] The slot from the previous figure is one branch point: 68 leads to the correct answer, and the 58 slip to a wrong one. Once the model emits a stop token, a reward function grades the finished completion.

[^tree-cond]: {-} Each \(p(\cdot)\) in the tree is conditioned on the prompt and on every token before it on the path, so \(p(24)\) means \(p(24 \mid x, 17, \times)\).

![A tree of possible completions for "What is 17 × 24?". The sampled path starts 17, ×, 24, and its probability is the product 0.6 × 0.9 × 0.7. At "340 +", it branches to 68 with probability 0.82 or 58 with 0.07. The completion reaching 408 gets reward 1 from the verifier, and the one reaching 398 gets reward 0.](/assets/img/policy-gradient-completion-tree.png)

For reasoning tasks, the reward function is often a verifier that returns \(R(x, y) = 1\) if the final answer is correct and \(0\) otherwise.[^verifiable] Our goal is to find the parameters \(\theta\) that maximize the expected reward, which we call the objective \(J(\theta)\):[^j-name]

[^j-name]: Read the objective as: draw a prompt \(x\) from the training set \(\mathcal{D}\), let the model generate a completion \(y\), compute its reward, and average over many such draws. The letter \(J\) comes from optimal control, where it names a cost to minimize. RL [borrowed it](http://incompleteideas.net/book/the-book-2nd.html) for a reward to **maximize**, which is why it isn't \(L\), the usual letter for a loss.

[^verifiable]: {-} This setup is often called [RL with verifiable rewards](https://arxiv.org/abs/2411.15124) (RLVR). Math problems with a known answer and coding tasks with unit tests are common examples.

\[
J(\theta) = \mathbb{E}_{x \sim \mathcal{D}, \; y \sim p_\theta(\cdot \mid x)}\big[R(x, y)\big]
\]

This is the standard reinforcement learning objective.[^standard-rl]

[^standard-rl]: In general RL, an agent collects a reward after each of many actions, and the objective is the expected total reward over an episode, often discounted. Generating a completion is an episode where each token is an action and the only reward comes at the end, so the total is just \(R(x, y)\).

From here on, I'll drop the prompt \(x\) from the notation; everything is conditioned on it.

## The policy gradient

To improve the model, we want to follow the gradient of the objective, \(\nabla_\theta J(\theta)\): the direction in parameter space that most increases the expected reward. This gradient is the **policy gradient**,[^pg-name] and methods that train by estimating and following it are called policy gradient methods. REINFORCE, [PPO](https://arxiv.org/abs/1707.06347), and GRPO are all examples. They share this expected-reward goal, but PPO and GRPO also change the update itself, clipping or reweighting it to keep training stable.

[^pg-name]: The name is short for the gradient of expected reward with respect to the _policy's_ parameters. It contrasts with value-based methods like [Q-learning](https://doi.org/10.1007/BF00992698), which learn how good each action is and act on those estimates instead of adjusting the policy directly. The term became standard with [Sutton et al. (2000)](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation).

The hard part is computing it. Written out, the objective is a sum over every possible completion:

\[
J(\theta) = \sum_y p_\theta(y) \, R(y)
\]

\(\theta\) appears only in \(p_\theta(y)\), how likely each completion is, and not in the reward \(R(y)\). If we could evaluate this sum, we could differentiate it directly, but there are far too many completions to enumerate.[^num-completions]

[^num-completions]: {-} With a vocabulary of about 150,000 tokens, even a 100-token completion has more than \(10^{500}\) possibilities.

The usual fix for an expectation we can't enumerate is to estimate it by sampling: generate \(N\) completions, compute their rewards, and average them. That gives a fine estimate of \(J\), but not one we can differentiate. The completions are discrete token sequences,[^sampling] and their rewards come from a verifier, a unit test, or a person, none of which we can backpropagate through. There is no path for autograd to follow from the reward back to \(\theta\).

Compare supervised fine-tuning, where the completion \(y\) is fixed training data and \(\theta\) appears directly in the loss \(-\log p_\theta(y)\). Here, \(\theta\) decides _which_ completions we get, not how any one of them is graded. What we need is a way to rewrite \(\nabla_\theta J\) as an average, over sampled completions, of something we _can_ differentiate.

[^sampling]: At each step, the model's probabilities define a categorical distribution over the vocabulary, and we draw one token from it. The result is an integer token ID: a small change to \(\theta\) either leaves it unchanged or flips it, so there is no smooth gradient. (Sampling often reshapes the distribution with a temperature or top-p cutoff. The derivations here assume we sample from the model's probabilities as-is.)

## The log-derivative trick

The workaround is a one-line identity, \(\nabla_\theta p_\theta = p_\theta \nabla_\theta \log p_\theta\),[^identity-proof] which moves the gradient inside the expectation:

[^identity-proof]: {-} By the chain rule, \(\nabla_\theta \log p_\theta = \nabla_\theta p_\theta / p_\theta\). Multiply both sides by \(p_\theta\).

\[
\begin{aligned}
\nabla_\theta J(\theta)
&= \nabla_\theta \, \mathbb{E}_{y \sim p_\theta}\big[R(y)\big] \\
&= \nabla_\theta \sum_y p_\theta(y) \, R(y) \\
&= \sum_y \underbrace{\nabla_\theta p_\theta(y)}_{\mathclap{\text{depends on } \theta}} \, R(y) \\
&= \sum_y \underbrace{p_\theta(y) \, \nabla_\theta \log p_\theta(y)}_{\mathclap{\text{the identity}}} \, R(y) \\
&= \mathbb{E}_{y \sim p_\theta}\big[R(y) \, \underbrace{\nabla_\theta \log p_\theta(y)}_{\mathclap{\text{the score}}}\big]
\end{aligned}
\]

The quantity \(\nabla_\theta \log p_\theta(y)\) is called the **score**.[^score-name] It points in the direction in parameter space that most increases the log-probability of the completion \(y\). The final line is an expectation over completions sampled from the policy \(p_\theta\) itself,[^pg-theorem] so we can estimate it by sampling. Perform \(N\) _rollouts_ (that is, draw \(N\) completions from the policy), score each one, and average:

[^score-name]: Don't read much into the word: it doesn't rate how good a completion is. The name comes from statistics, where \(\nabla_\theta \log p\) is the score function of [maximum-likelihood estimation](https://doi.org/10.1017/S0305004100009580). REINFORCE is sometimes called the score-function estimator for the same reason.

[^pg-theorem]: This is the language-model case of the _policy gradient theorem_ ([Sutton et al., 2000](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation)). In general RL, it reads \(\nabla_\theta J = \mathbb{E}\big[\sum_t Q^\pi(s_t, a_t)\, \nabla_\theta \log \pi_\theta(a_t \mid s_t)\big]\) (\(\pi_\theta\) is the usual RL notation for the policy \(p_\theta\)), where \(Q^\pi(s_t, a_t)\) is the expected future reward after taking action \(a_t\) in state \(s_t\). For a completion, the state is the prefix and the action is the next token. With only a final reward, \(Q^\pi\) is the expected reward of finishing from that prefix, and the sampled \(R\) is a one-sample estimate of it.

\[
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} R(y_i) \, \nabla_\theta \log p_\theta(y_i), \qquad y_i \sim p_\theta
\]

This is the REINFORCE estimator,[^reinforce] also known as the _Monte Carlo policy gradient_: it estimates the gradient by averaging over complete sampled rollouts, using each one's actual reward rather than a learned estimate of how good it was.[^unbiased] Each term pairs a direction with a weight: the score points toward making that rollout's completion more likely, and the reward sets how much that direction counts.

[^unbiased]: It is unbiased: averaged over many batches, it equals the true gradient. Any single batch, though, can point well away from it.

[^reinforce]: Introduced by Ronald Williams in [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) (1992). PPO, GRPO, DAPO, and most other RL algorithms used for LLMs today are elaborations of this estimator.

<figure class="fullwidth">
  <img src="/assets/img/policy-gradient-intuition.png" sizes="(max-width: 760px) 100vw, 1400px" alt="Four rollouts for &quot;What is 17 × 24?&quot;: A and C reach 408 and earn reward 1, while B (an arithmetic slip) and D (an estimate) earn 0. A bar holding all the probability for this prompt shows A rising from 0.22 to 0.27 and C from 0.03 to 0.05 after one update, B and D barely changing, and unsampled completions shrinking.">
</figure>

Because all completions share a total probability of 1, A's and C's gains come from elsewhere, here mostly from completions nobody sampled. B barely changes, because it shares everything up to "340 +" with A, so reinforcing A also lifts most of B's path. The numbers are only illustrative: A and C are pushed up, but how every other completion moves depends on how the model's parameters are shared.

With the reward held fixed, \(R \, \nabla_\theta \log p_\theta(y)\) is exactly the gradient of \(R \log p_\theta(y)\): a log-likelihood on one of the model's own samples, weighted by its reward. So **policy gradient is supervised fine-tuning on your own samples, weighted by reward.** With \(+1/0\) rewards, as in the figure above, it is literally SFT on the correct completions, so incorrect completions are never pushed down directly; they only lose share.[^rft]

[^rft]: Training on your own correct samples is also used on its own, as rejection-sampling fine-tuning or [expert iteration](https://arxiv.org/abs/1705.08439). [STaR](https://arxiv.org/abs/2203.14465) is an early example for reasoning.

## From sequences to tokens

A completion's log-probability is a sum of per-token log-probabilities, so its score breaks into one term per token:

\[
\nabla_\theta \log p_\theta(y) = \sum_{t=1}^{T} \nabla_\theta \log p_\theta(y_t \mid y_{<t})
\]

Each term is the score of a single token: the direction that most increases the probability of choosing \(y_t\), given everything generated before it. So we can study the update one position at a time. Pick a single position, such as the slot right after "340 +" in the 17 × 24 example, and hold its prefix \(y_{<t}\) fixed. For each token \(v\) in the vocabulary, write

\[
p_v = p_\theta(v \mid y_{<t}) \qquad \text{and} \qquad s_v = \nabla_\theta \log p_v
\]

for the model's probability of choosing \(v\) at this position and that token's score. In the example, \(p_{68} = 0.82\) and \(p_{58} = 0.07\).

Only one token is actually sampled at each position. Its contribution to the update is \(R \, s_{y_t}\): that token's score, multiplied by the reward the whole completion earned.

To see what a token's score looks like, look at the last layer. At one position, the network outputs a logit \(z_u\) for every token \(u\) in the vocabulary, and \(p = \operatorname{softmax}(z)\). Differentiating the log-softmax gives the **logit gradient** of the chosen token \(v\):

\[
\log p_v = z_v - \log \sum_u e^{z_u}
\qquad\Longrightarrow\qquad
\frac{\partial \log p_v}{\partial z_u} =
\begin{cases}
1 - p_v & u = v \\
-p_u & u \neq v
\end{cases}
\]

It is positive on the chosen token's own logit, negative on every other logit in proportion to that token's probability, and sums to zero. The score is this vector carried back through the network to the parameters by the chain rule:

\[
s_v = \sum_u \big(\mathbb{1}[u = v] - p_u\big) \, \nabla_\theta z_u
\]

As \(p_v \to 1\), every coefficient in this sum goes to zero, so the score does too. Across completions A and B from the figure above:

<figure class="fullwidth">
  <img src="/assets/img/policy-gradient-token-credit.png" sizes="(max-width: 760px) 100vw, 1400px" alt="Completions A (reward 1) and B (reward 0) as rows of tokens, each labeled with its probability and its own-logit gradient, 1 − p, with an arrow of that length. In A, uncertain tokens like 17, 24, and 68 get large gradients and confident filler almost none. Zoom-ins show the full logit gradient summing to zero. In B, every score is multiplied by 0, so nothing updates.">
</figure>

The reward only says whether the finished completion was right, so every position gets the same \(R\), whatever its token did:[^credit] B's correct opening steps get nothing, and A's filler tokens get the full reward. What differs from token to token is the score, which is tiny for tokens the model was already sure of.[^step-caveat]

[^credit]: This is the _credit assignment_ problem. Methods that learn a value function, or use [process reward models](https://arxiv.org/abs/2305.20050) that grade intermediate steps, try to give individual tokens their own credit.

[^step-caveat]: The actual step also scales with the learning rate and depends on how the network maps parameters to logits, and a token's probability can still change because of other tokens' gradients.

## The score has zero mean

The score has a simple but important property. On-policy, when the token is sampled from the same distribution \(p\) whose score we compute, the expected score is exactly zero:

\[
\begin{aligned}
\mathbb{E}_{y_t \sim p}[s_{y_t}] &= \sum_v p_v \, \nabla_\theta \log p_v = \sum_v \nabla_\theta p_v \\
&= \nabla_\theta \sum_v p_v = \nabla_\theta 1 = 0
\end{aligned}
\]

Intuitively, probability is conserved. Any change to \(\theta\) that makes some tokens more likely must make others less likely by the same total amount. Weighted by how often each token is sampled, the pushes cancel.

We can check this at the logits. In vector form, the logit gradient from the previous section is \(e_v - p\), where \(e_v\) is the one-hot vector for \(v\); the zoom-ins in the figure above show it for 68 and for "=". Averaging these vectors over which token gets sampled, weighted by \(p\), gives \(\sum_v p_v (e_v - p) = p - p = 0\).

## Baselines and group centering

Nothing so far required the rewards to be \(+1/0\). What if we use \(+1/-1\) instead? That doubles every reward and then subtracts 1. Doubling just doubles the gradient. For the shift, the zero-mean identity is the answer: we can subtract any baseline \(b\) from the reward without changing the expected gradient, as long as \(b\) does not depend on the sampled token:

\[
\mathbb{E}_{p}\big[(R - b) \, s_{y_t}\big] = \mathbb{E}_{p}[R \, s_{y_t}] - b \, \underbrace{\mathbb{E}_{p}[s_{y_t}]}_{=\,0} = \mathbb{E}_{p}[R \, s_{y_t}]
\]

The quantity \(A = R - b\) is called the **advantage**. Subtracting a baseline adds no bias: REINFORCE stays unbiased, with exactly the same expected gradient. What it can change is the variance, and a well-chosen baseline reduces it dramatically.[^baselines] To see why, split each rollout's term in two:

\[
R \, s_{y_t} = \underbrace{b \, s_{y_t}}_{\text{mean zero, but noisy}} + \underbrace{(R - b) \, s_{y_t}}_{\text{the advantage-weighted part}}
\]

The first part contributes nothing to the expected gradient, but each sample of it is a large vector pointing somewhere different. In the extreme case where every completion earns \(R = 1\), the true gradient is zero, yet each sample still pushes its own log-probabilities up at random. With \(b = 1\), every term vanishes. How much a baseline helps depends on \(b\). The standard choice is the prompt's expected reward. It isn't exactly optimal,[^optimal-baseline] but it is simple to estimate, and because some prompts are far easier than others, estimating it per prompt removes a large source of noise.

[^optimal-baseline]: The variance depends on \(b\) only through \(\mathbb{E}\big[(R - b)^2 \lVert s \rVert^2\big]\), which is minimized at \(b^\star = \mathbb{E}\big[R \lVert s \rVert^2\big] / \mathbb{E}\big[\lVert s \rVert^2\big]\). This is close to \(\mathbb{E}[R]\) only when the reward is roughly unrelated to the size of the score. They can differ a lot: for a binary choice that succeeds with probability 0.1, successes have the larger score, so \(b^\star = 0.9\) while \(\mathbb{E}[R] = 0.1\).

[^baselines]: See [Greensmith, Bartlett, and Baxter (2004)](https://jmlr.org/papers/v5/greensmith04a.html) for a thorough treatment of baselines as variance reduction for policy gradient estimates.

GRPO[^grpo] popularized a simple, critic-free[^critic] baseline for LLMs: sample a group of \(G\) completions for each prompt and use the group's mean reward as the baseline for each of them:

\[
A_i = R_i - \frac{1}{G} \sum_{j=1}^{G} R_j
\]

[^critic]: A _critic_ is a second network trained to predict the expected reward, the value \(V(x)\), to use as the baseline. PPO usually trains one alongside the policy. GRPO replaces it with the group's mean reward, which saves a model of similar size to the policy.

[^grpo]: Introduced in [DeepSeekMath](https://arxiv.org/abs/2402.03300). GRPO also divides by the group's reward standard deviation. [Dr. GRPO](https://arxiv.org/abs/2503.20783) argues that this normalization introduces a bias toward easy and hard prompts. Here I use mean-centering only. Because the group mean includes the completion's own reward, the expected gradient is scaled by \(1 - 1/G\). When every prompt uses the same \(G\), that is just a slightly smaller step size. A leave-one-out baseline, as in [RLOO](https://arxiv.org/abs/2402.14740), removes the factor.

Correct completions now get a positive advantage and are reinforced. Incorrect completions get a negative advantage and are suppressed. Prompts where every completion succeeds, or every completion fails, contribute nothing. For the four rollouts from earlier:

![The same four rollouts with group-centered advantages. The group mean reward is 0.5, so A and C get advantage +0.5 and are pushed up, while B and D get −0.5 and are pushed down. Along the prefix A and B share, their pushes cancel.](/assets/img/policy-gradient-group-centered.png)

REINFORCE with group-centered advantages fits in a few lines of PyTorch:

```python
def reinforce_loss(logprobs, rewards, mask, group_size):
    # logprobs: (B, T) per-token log p_θ(y_t | y_<t)
    # rewards:  (B,)   one per completion, grouped by prompt
    # mask:     (B, T) 1 on completion tokens, else 0
    r = rewards.view(-1, group_size)
    advantages = (r - r.mean(dim=1, keepdim=True)).view(-1)
    seq_logprobs = (logprobs * mask).sum(dim=1)  # log p_θ(y)
    return -(advantages * seq_logprobs).mean()
```

The advantages are constants that carry no gradient, so differentiating this loss gives exactly the estimator from the previous sections, with advantages in place of rewards.[^token-norm] PPO, GRPO, [DAPO](https://arxiv.org/abs/2503.14476), and most other RL algorithms used for LLMs start from this loss and add clipping, masking, or reweighting.

[^token-norm]: Many implementations instead divide the summed loss by the total number of completion tokens in the batch. That changes more than the scale: the denominator varies with the sampled lengths, so it reweights completions by length. [Dr. GRPO](https://arxiv.org/abs/2503.20783) discusses this bias.

## The on-policy assumption

Both identities in this post, the log-derivative rewrite and the zero-mean score, assume that **the completions are sampled from the same distribution \(p_\theta\) that we differentiate**. In the rewrite, \(p_\theta\) is both the distribution we average over and the model we differentiate. In the zero-mean identity, averaging over \(p\) itself is what makes the pushes cancel. If the samples come from even a slightly different distribution, the expected score is no longer guaranteed to be zero, and subtracting a baseline shifts the expected gradient instead of leaving it unchanged.

In practice, this assumption rarely holds exactly. In the `reinforce_loss` above, `logprobs` comes from the training framework's forward pass, but the completions were generated by a separate inference engine, such as [vLLM](https://github.com/vllm-project/vllm) or [SGLang](https://github.com/sgl-project/sglang). The two are supposed to compute the same distribution, but they seldom do. They can run at different numerical precisions, and in asynchronous setups the inference engine may still be serving weights from a few updates ago. Either way, the completions come from a slightly different model than the one being trained. That gap is where the trouble starts.

## Further reading

- [OpenAI Spinning Up: Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) derives the same policy gradient in general RL notation, including the expected grad-log-prob lemma, which is the zero-mean identity above.
- Lilian Weng's [Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) surveys the family, from REINFORCE through actor-critic methods and PPO.
- Andrej Karpathy's [Deep Reinforcement Learning: Pong from Pixels](https://karpathy.github.io/2016/05/31/rl/) builds intuition for policy gradients by training a small network to play Pong.
- Nathan Lambert's [RLHF book](https://rlhfbook.com/) covers policy gradient methods for language models, including PPO, GRPO, and RLOO.

## References

<textarea id="bibtex_input" style="display:none;">
@article{williams1992reinforce,
      title={Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning},
      author={Ronald J. Williams},
      journal={Machine Learning},
      volume={8},
      pages={229--256},
      year={1992},
      url={https://link.springer.com/article/10.1007/BF00992696}
}
@inproceedings{sutton2000policy,
      title={Policy Gradient Methods for Reinforcement Learning with Function Approximation},
      author={Richard S. Sutton and David McAllester and Satinder Singh and Yishay Mansour},
      booktitle={Advances in Neural Information Processing Systems},
      volume={12},
      year={2000},
      url={https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation}
}
@misc{schulman2017ppo,
      title={Proximal Policy Optimization Algorithms},
      author={John Schulman and Filip Wolski and Prafulla Dhariwal and Alec Radford and Oleg Klimov},
      year={2017},
      eprint={1707.06347},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1707.06347}
}
@misc{yu2025dapo,
      title={DAPO: An Open-Source LLM Reinforcement Learning System at Scale},
      author={Qiying Yu and others},
      year={2025},
      eprint={2503.14476},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.14476}
}
@book{sutton2018rl,
      title={Reinforcement Learning: An Introduction},
      author={Richard S. Sutton and Andrew G. Barto},
      edition={2},
      publisher={MIT Press},
      year={2018},
      url={http://incompleteideas.net/book/the-book-2nd.html}
}
@misc{lambert2024tulu3,
      title={Tulu 3: Pushing Frontiers in Open Language Model Post-Training},
      author={Nathan Lambert and others},
      year={2024},
      eprint={2411.15124},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.15124}
}
@misc{lightman2023verify,
      title={Let's Verify Step by Step},
      author={Hunter Lightman and Vineet Kosaraju and Yura Burda and Harri Edwards and Bowen Baker and Teddy Lee and Jan Leike and John Schulman and Ilya Sutskever and Karl Cobbe},
      year={2023},
      eprint={2305.20050},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2305.20050}
}
@misc{anthony2017expert,
      title={Thinking Fast and Slow with Deep Learning and Tree Search},
      author={Thomas Anthony and Zheng Tian and David Barber},
      year={2017},
      eprint={1705.08439},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/1705.08439}
}
@article{watkins1992qlearning,
      title={Q-learning},
      author={Christopher J. C. H. Watkins and Peter Dayan},
      journal={Machine Learning},
      volume={8},
      pages={279--292},
      year={1992},
      url={https://doi.org/10.1007/BF00992698}
}
@article{fisher1925estimation,
      title={Theory of Statistical Estimation},
      author={Ronald A. Fisher},
      journal={Mathematical Proceedings of the Cambridge Philosophical Society},
      volume={22},
      number={5},
      pages={700--725},
      year={1925},
      url={https://doi.org/10.1017/S0305004100009580}
}
@article{greensmith2004variance,
      title={Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning},
      author={Evan Greensmith and Peter L. Bartlett and Jonathan Baxter},
      journal={Journal of Machine Learning Research},
      volume={5},
      pages={1471--1530},
      year={2004},
      url={https://jmlr.org/papers/v5/greensmith04a.html}
}
@misc{shao2024deepseekmath,
      title={DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models},
      author={Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Xiao Bi and Haowei Zhang and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
      year={2024},
      eprint={2402.03300},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2402.03300}
}
@misc{ahmadian2024rloo,
      title={Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs},
      author={Arash Ahmadian and Chris Cremer and Matthias Gallé and Marzieh Fadaee and Julia Kreutzer and Olivier Pietquin and Ahmet Üstün and Sara Hooker},
      year={2024},
      eprint={2402.14740},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2402.14740}
}
@misc{zelikman2022star,
      title={STaR: Bootstrapping Reasoning With Reasoning},
      author={Eric Zelikman and Yuhuai Wu and Jesse Mu and Noah D. Goodman},
      year={2022},
      eprint={2203.14465},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2203.14465}
}
@misc{liu2025drgrpo,
      title={Understanding R1-Zero-Like Training: A Critical Perspective},
      author={Zichen Liu and Changyu Chen and Wenjun Li and Penghui Qi and Tianyu Pang and Chao Du and Wee Sun Lee and Min Lin},
      year={2025},
      eprint={2503.20783},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.20783}
}
</textarea>
