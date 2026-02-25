---
title: Direct Preference Optimization Explained In-depth
subtitle: Simpler preference-tuning without reinforcement learning
date: 2024-04-13T00:00:00-08:00
blurb: Covering DPO, a recently-proposed alternative to RLHF for preference tuning.
tags: ["post", "machine-learning", "nlp", "language-models", "dpo", "rlhf", "ai-alignment"]
math: true
---

With my first blog post, I want to cover an excellent paper that was published last year: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) by Rafailov et al.

Commonly referred to as DPO, this method of preference tuning is an alternative to Reinforcement Learning from Human Feedback (RLHF) that avoids the actual reinforcement learning. In this blog post, I will explain DPO from first principles; readers do not need an understanding of RLHF. However, fair warning that there will be some math involved - mostly probability, algebra, and optimization - but I will do my best to explain everything clearly.

## Training, tuning, and aligning LLMs

To contextualize DPO, and preference-tuning in general, let's review the modern process for creating language models such as ChatGPT or Claude. The following steps are sequential, with each one building upon the previous:

1. **Pre-train a base model** on internet-scale data. Given a snippet of text, this model is trained to predict the immediate next word. This conceptually simple task scales up extremely well and allows LLMs to encode a huge amount of knowledge from their training data. Examples of base models include [GPT-3](https://arxiv.org/abs/2005.14165), [Llama3](https://ai.meta.com/blog/meta-llama-3/), and [Mistral](https://mistral.ai/news/announcing-mistral-7b/).

2. Take a pre-trained base model and **fine-tune it on a task-specific dataset of demonstrations**. For example, if you are trying to create a helpful dialog model like ChatGPT, you would want to tune your model on a dataset of conversational dialog, so that your model's outputs sound more like parts of a conversation and less like a Wikipedia page. In this stage, we still use the next word prediction task, and the fine-tuning procedure updates our model to make predictions that more closely align with the high-quality task-specific examples we are feeding it. Examples of fine-tuned models in this stage are [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html), [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/) and [Mistral-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1).

3. Finally, we **fine-tune the model based on human preferences**. Human preferences are powerful because they are so easily and cheaply _expressed_. Think of how easy it is to compare two movies and pick a favorite. Yet how difficult it would be to make a film that embodies the qualities that drive you to visit a theater. Similarly, it is challenging to describe exactly how we want our model to behave (as we attempt to do in step 2), but given examples of model behavior it is straightforward to indicate a preference for a specific type of behavior. For a while, this sort of preference-tuning was done using RLHF. Recently, RLHF has been somewhat supplanted by DPO due to the relative simplicity of the latter. LLMs that have been tuned using human preferences include [Llama 3 Instruct](https://ai.meta.com/blog/meta-llama-3), [ChatGPT-4](https://cdn.openai.com/papers/gpt-4-system-card.pdf), [Claude 3 Opus](https://www.anthropic.com/news/claude-3-family), and [Gemini Ultra](https://blog.google/technology/ai/google-gemini-ai).

The [Gemini whitepaper](https://arxiv.org/abs/2312.11805) provides a nice visual representation of these stages:

![LLM Training Stages](/assets/img/llm-training-stages.png)

## Tuning LLMs on preference data

It is hard and time-consuming work to create high-quality demonstrations of the behavior we want our LLM to mimic. And it would be expensive to hire labelers to help us create such data. However, once we have a model that is "good enough" at demonstrating desired behavior, we can shift into high gear. Given a prompt, we can [sample two different responses from our LLM](https://huggingface.co/blog/how-to-generate#sampling) by injecting a small amount of randomness[^temperature]. Now, it is cheap and easy to have a labeler express a preference for one of the two completions.

[^temperature]: This is typically done by generating text with a `temperature` that is greater than zero. [Here](https://lukesalamone.github.io/posts/what-is-temperature/) is a lovely little demo that explains how temperature affects model outputs visually.

While using ChatGPT or Gemini, you may have noticed that you will occasionally be asked to choose between two similar answers from which to continue your conversation. This preference is recorded and used to improve the model in a future round of preference-tuning. Similarly, [Chatbot Arena](https://chat.lmsys.org/) collects preference data for the purpose of rating LLMs based on human assessments:

![LMSys Chatbot Arena, a head-to-head comparison tool for instruction-tuned LLMs](/assets/img/chatbot-arena.png)

There are many publicly available preference datasets, such as LMSys' [Chatbot Arena Conversations dataset](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations), OpenAI's [WebGPT Comparisons dataset](https://huggingface.co/datasets/openai/webgpt_comparisons?row=1), and Anthropic's [Helpfulness-Harmlessness RLHF dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf) (explicit/offensive content warning).

Formally, these datasets can be expressed as follows:

\[
\mathcal{D}=\{x^{(i)},y_w^{(i)},y_l^{(i)}\}_{i=1}^N
\]

Where \(x\) is the context/prompt, \(y_w\) is the preferred completion, and \(y_l\) is the less desirable completion.

### The Bradley-Terry Model

So what do we do with all this preference data? We want to leverage it to modify our LLM to output responses that better conform to the preferences. To begin, let us explore a simple probability model:

\[
p^*(i \succ j) = \frac{s_i}{s_i + s_j}
\]

This is the Bradley-Terry model, which is a model for the outcome of pairwise comparisons. In plain English, it says "We model the true[^star] probability that outcome \(i\) is preferred to outcome \(j\) as the score of \(i\) over the combined scores of \(i\) and \(j\)".

[^star]: This is the reason for the "star" in \(p^*\): to indicate that we are modeling the true underlying distribution of human preferences. Likewise, shortly we will see \(r^*\), which indicates the true underlying reward function that grades our completions, and \(\pi^*\), which indicates the optimal policy we want our LLM to mimic.

Readers may be familiar with the Bradley-Terry model from the context of Elo scores, which are popular in [chess](https://www.chess.com/terms/elo-rating-chess) and [other](https://liquipedia.net/starcraft/Elo_rating#Detailed_Explanation) [competitive](https://www.goratings.org/en/) [games](https://lmsys.org/blog/2023-12-07-leaderboard/). The Bradley-Terry model is a generalization of the Elo rating system, where the probability of player A beating player B is given by \(p(A \succ B) = \frac{1}{1 + 10^{(R_B-R_A)/400}} = \frac{s_A}{s_A + s_B}\). Here \(R\) indicates a player's rating[^elo] and \(s = 10^{R/400}\).

[^elo]: So if player A's Elo rating is 2000 and player B's is 1600 then player A is expected to be 10 times more likely to win than player B, because \(p(A \succ B)=\frac{1}{1 + 10^{(1600-2000)/400}}=10/11\).

Under the Bradley-Terry model, it is common to choose to parameterize the score as \(s=e^r\), where \(r\) stands for reward. The term "reward" is borrowed from the world of reinforcement learning, where greater rewards are received for a more desirable series of actions - similar to achieving a higher score for performing better in a video game.

With this parameterization, our model starts to look pretty nice - a simple difference in reward values passed through the logistic function[^logistic].

\[
p^*(i \succ j) = \frac{s_i}{s_i + s_j} = \frac{e^{r^*_i}}{e^{r^*_i} + e^{r^*_j}} = \frac{1}{1+e^{-(r^*_i-r^*_j)}} = \sigma(r^*_i - r^*_j)
\]

[^logistic]: The logistic function is an S-shaped (or sigmoid) function commonly denoted using \(\sigma(x)\). It frequently appears when working with probabilities because it can "squash" values in \(\mathbb{R}\) (the set of all real numbers) into \((0, 1)\) (the set of probability values, excluding exactly 0 or 1). ![Sigmoid Function](/assets/img/sigmoid.png)

### Applying the Bradley-Terry Model to LLMs

Now, we want to take the Bradley-Terry model and leverage it alongside a dataset of preferences in order to improve our LLM's generated outputs.

In our preference dataset (\(\mathcal{D}\)), we have two completions and we want to model the probability of one being preferred over the other. In a sense, each completion elicits some reward based on its quality, and our ultimate goal will be to nudge our LLM to produce completions that are of higher quality. Therefore, we will parameterize the reward using our LLM. We will call this reward \(r^*(x, y)\), which just means that the reward is a function of the context/prompt (\(x\)) and the completion (\(y\)).

So after adapting our preference model to use our parameterized reward function, we have:

\[
p^*(y_1 \succ y_2 | x) = \sigma(r^*(x, y_1) - r^*(x, y_2))
\]

But talking in terms of optimal solutions and rewards does us no good, since we do not have access to the optimal reward function. In practice, it is common to learn a reward model \(r_\phi(x, y)\) that mimics the optimal reward function. We can estimate the parameters \(\phi\) of this reward model by framing this as a binary classification problem where our objective is to minimize the following [negative log-likelihood loss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html) function on our preference dataset \(\mathcal{D}\):[^expectation1]

\[
\mathcal{L}_R(r_\phi, \mathcal{D}) = -\mathbb{E}_{(x,y_w,y_l)\sim \mathcal{D}}[\log(\sigma(r_\phi(x,y_w) - r_\phi(x, y_l)))]
\]

[^expectation1]: {-} \(\mathbb{E}_{(x,y_1,y_2)\sim \mathcal{D}}[f(x,y_w,y_l)]\) is just a formal way of saying "the expected value of function \(f\) on data points sampled from our preference dataset".

Under the RLHF framework, we could leverage this learned reward model in a reinforcement learning setting to optimize an LLM to output completions that achieve high rewards. However, DPO takes a different tack - instead of the two-stage RLHF process, DPO reparameterizes the Bradley-Terry model so that we can use a similar loss function to directly optimize the parameters of our LLM such that it produces outputs that are preferred by human observers.

### The probability of a completion

At this point, the idea of optimizing LLMs based on preferences or rewards may feel fairly abstract. So we're going to take a moment to introduce a new probability function, \(\pi(y|x)\), that represents the literal output of our LLM. In reinforcement learning notation, \(\pi\) indicates a policy (i.e. a strategy), and policies are optimized to maximize reward. Specifically, \(\pi_\theta(y|x)\) is the probability of generating the completion \(y\) based on an LLM with parameters \(\theta\) given that we start with prompt \(x\).

What do we mean by "the probability of generating the completion \(y\)"? Our LLM is an auto-regressive text generator, and, upon each auto-regressive step, it computes a probability value for every word[^token] in its vocabulary.

[^token]: In practice, modern LLMs operate on tokens, not words. For our purposes, the difference doesn't really matter. You can learn more by playing with an [online tokenizer demo](https://platform.openai.com/tokenizer) or digging through Karpathy's [minbpe](https://github.com/karpathy/minbpe) repo.

![Next Word Prediction Graphic](/assets/img/next-word-prediction.png)
So - proceeding in order through every word in completion \(y\) - we compute the probability of the next word in the completion given all of the preceding words. Now, we have a probability value for every word in the completion! So we can compute the joint probability of generating the sequence of words as the product of the individual probabilities of observing each word along the way[^logprobs]:

\[
\pi_\theta(y|x)=\prod_{t=0}^{|y|}p_{LLM_\theta}(y_t|x,y_{0:t})
\]

[^logprobs]: Multiplying probabilities can result in numerical underflow. It is common to instead work with logprobs: \(\prod_i p_i=e^{\sum_i log p_i}\). Since every term in the summation of logprobs increases the magnitude of its output, underflow is avoided. OpenAI has a nice [guide to using token logprobs](https://cookbook.openai.com/examples/using_logprobs) returned by an LLM.

Another way to think about it is that there is a tree of possible completions and we are computing the probability of tracing one specific path from the root (end of the prompt) to a leaf (stop-token).

![Probability of Sequence Graphic](/assets/img/sequence-prediction.png)

When training, we know the entire text completion ahead of time, so, by applying a causal attention mask, we can calculate all of the individual next-word probabilities (and thus \(\pi_\theta(y|x)\)) via a single forward pass through our LLM.

## Optimizing our LLM based on preferences

Now that we've got our framework in place, let us remind ourselves of our goal: to improve the outputs of our LLM. Stated another way, we want the completion (y) our LLM provides for a prompt (x) to generate a large reward \(r(x, y)\). With this in mind, we can formulate an optimization problem where we want to find the parameters of our LLM (\(\theta\)) that maximize our expected reward for prompts similar to those we see in practice.[^expectation2]

\[
\max_{\theta}\mathbb{E}_{x\sim \mathcal{D},y\sim \pi_\theta(y|x)}[r(x, y)]
\]

[^expectation2]: {-} \(\mathbb{E}_{x\sim \mathcal{D},y\sim \pi_\theta(y|x)}[r(x, y)]\) is just a formal way of saying "the expected reward attained by completions generated/sampled from our model (\(y\sim \pi_\theta(y|x)\)) based on prompts sampled from our dataset (\(x\sim \mathcal{D}\))".

This is a bit too simplistic, however. In practice, we start with the parameters of our fine-tuned base model, and we have some belief that the outputs generated by our fine-tuned base model are pretty good, so we don't want the outputs of our model to change too much unless they improve the reward significantly. With that in mind, we amend our optimization problem to include a regularization constraint to help enforce this belief.

\[
\max_{\theta}\mathbb{E}_{x\sim \mathcal{D},y\sim \pi_\theta(y|x)}[r(x, y)] - \beta\mathbb{D}_{KL}[\pi_\theta(y|x) \ \Vert \ \pi_{ref}(y|x)]
\]

\(\mathbb{D}_{KL}[P \Vert Q]\) is the [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)[^kldiv], a statistical distance measure. It quantifies how the probability distribution P differs from probability distribution Q. This constraint based on the KL divergence just encodes the idea that we want to penalize outputs from our model (\(\pi_\theta\)) based on how much they differ from outputs from the fine-tuned model (i.e. the reference model) we started with (\(\pi_{ref}\)). \(\beta\) is a scalar hyperparameter that controls the strength of the constraint.

[^kldiv]: KL divergence is [one of many](https://arxiv.org/pdf/2006.05990.pdf) traditional methods for regularizing an RL agent's policy. In the cases of DPO and RLHF, it is a natural choice because we begin with a strong reference policy at hand - the LLM output by our fine-tuning procedure.

Now, we want to derive the optimal solution to this optimization problem. This will rely on [Gibbs' Inequality](https://en.wikipedia.org/wiki/Gibbs%27_inequality) - the fact that \(\mathbb{D}_{KL}[P \Vert Q]\geq0\) and \(\mathbb{D}_{KL}[P \Vert Q]=0\) if and only if \(P=Q\).[^gibbs]

[^gibbs]: The intuition here is that the KL-divergence is a distance measure (kind of), and there is no distance between P and Q if they are equal, and there must be some distance if they are not equal.

\[
\max_{\pi_\theta}\mathbb{E}_{x\sim \mathcal{D},y\sim \pi_\theta(y|x)}[r(x, y)] - \beta\mathbb{D}_{KL}\left[\pi_\theta(y|x) \ \Vert \ \pi_{ref}(y|x)\right] \\[10pt]
=\max_{\pi_\theta}\mathbb{E}_{x\sim \mathcal{D},y\sim \pi_\theta(y|x)}[r(x, y)] - \beta\mathbb{E}_{y\sim \pi_\theta(y|x)}\left[\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}\right] \\[10pt]
= \max_{\pi_\theta}\mathbb{E}_{x\sim \mathcal{D}}\mathbb{E}_{y\sim \pi_\theta(y|x)}\left[r(x,y) - \beta\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}\right] \\[10pt]
= \min_{\pi_\theta}\mathbb{E}_{x\sim \mathcal{D}}\mathbb{E}_{y\sim \pi_\theta(y|x)}\left[\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} - \frac{1}{\beta}r(x,y)\right] \\[10pt]
= \min_{\pi_\theta}\mathbb{E}_{x\sim \mathcal{D}}\mathbb{E}_{y\sim \pi_\theta(y|x)}\left[\log\frac{\pi_\theta(y|x)}{\frac{1}{Z(x)}\pi_{ref}(y|x)e^{\frac{1}{\beta}r(x,y)}} - \log Z(x)\right] = ...
\]

where \(Z(x)=\sum_y\pi_{ref}(y|x)e^{\frac{1}{\beta}r(x,y)}\). Importantly, this \(Z(x)\) term depends only on \(x\) and \(\pi_{ref}\) and not on \(y\) or \(\pi_\theta\). This lets us do a bit of reorganizing from where we just left off.

\[
...= \min_{\pi_\theta}\mathbb{E}_{x\sim \mathcal{D}}\left[\mathbb{E}_{y\sim \pi_\theta(y|x)}\left[log\frac{\pi_\theta(y|x)}{\frac{1}{Z(x)}\pi_{ref}(y|x)e^{\frac{1}{\beta}r(x,y)}}\right] - logZ(x)\right] \\[10pt]
= \min_{\pi_\theta}\mathbb{E}_{x\sim \mathcal{D}}\left[\mathbb{D}_{KL}\left(\pi_\theta(y|x)\ \Vert\ \frac{1}{Z(x)}\pi_{ref}(y|x)e^{\frac{1}{\beta}r(x,y)}\right) - logZ(x)\right]
\]

And we have nearly arrived! Since \(Z(x)\) does not depend on \(\pi_\theta\), we can just ignore it when deriving the optimal solution. We can now use Gibbs' inequality as mentioned above: \(\mathbb{D}_{KL}\left(\pi_\theta(y|x)\ \Vert\ \frac{1}{Z(x)}\pi_{ref}(y|x)e^{\frac{1}{\beta}r(x,y)}\right)\) is minimized at zero if, and only if, the two distributions on either side of \(\Vert\) are identical. So, the optimal solution (denoted as \(\pi^*\)) to our optimization problem for all \(x \in \mathcal{D}\) is:

\[
\pi^*(y|x)=\pi_\theta(y|x)=\frac{1}{Z(x)}\pi_{ref}(y|x)e^{\frac{1}{\beta}r(x,y)}
\]

### Direct Preference Optimization

So we know the optimal solution to our optimization problem, but can we access it? No. The term \(Z(x)=\sum_y\pi_{ref}(y|x)e^{\frac{1}{\beta}r(x,y)}\) is intractable - computing it requires summing over every possible string of words.

Instead, we can reorganize the optimal solution from above such that we express the reward function in terms of the optimal policy \(\pi_\theta\), the reference policy \(\pi_{ref}\), and the intractable function \(Z\):

\[
r(x,y) = \beta\log{\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}} + \beta\log{Z(x)}
\]

This same reorganization can be applied using the underlying ground-truth reward \(r^*\) and its corresponding optimal policy \(\pi^*\).

\[
r^*(x,y) = \beta\log{\frac{\pi^*(y|x)}{\pi_{ref}(y|x)}} + \beta\log{Z(x)}
\]

Now here comes the clever trick noticed by the authors of DPO. We can use this reorganized expression of the optimal solution to our optimization problem to _reparameterize_ the Bradley-Terry preference model from above so that it is expressed in terms of an optimal policy \(\pi^*\) and not in terms of an underlying reward function! And even better, once we plug everything in, we notice that the intractable \(Z(x)\) function cancels out!

\[
p^*(y_1 \succ y_2 | x) = \sigma(r^*(x, y_1) - r^*(x, y_2)) \\[10pt]
= \sigma\left(\beta\log{\frac{\pi^*(y_1|x)}{\pi_{ref}(y_1|x)}} + \beta\log{Z(x)} - \left(\beta\log{\frac{\pi^*(y_2|x)}{\pi_{ref}(y_2|x)}} + \beta\log{Z(x)}\right)\right) \\[10pt]
= \sigma\left(\beta\log{\frac{\pi^*(y_1|x)}{\pi_{ref}(y_1|x)}} - \beta\log{\frac{\pi^*(y_2|x)}{\pi_{ref}(y_2|x)}}\right)
\]

Now, with our reparameterized Bradley-Terry model, we can use supervised learning to directly learn a policy that mimics the optimal policy. We can minimize a negative log-likelihood loss function over our preference dataset \(\mathcal{D}\) to estimate the parameters of our policy \(\pi_\theta\):

\[
\mathcal{L}_{DPO}(\pi_\theta;\pi_{ref}) = -\mathbb{E}_{(y_w,y_l,x)\sim \mathcal{D}}\left[\log\left(\sigma\left(\beta\log{\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}} - \beta\log{\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}}\right)\right)\right] \\[10pt]= -\mathbb{E}_{(y_w,y_l,x)\sim \mathcal{D}}\left[\log\left(\sigma\left(\beta\left(\log{\frac{\pi_\theta(y_w|x)}{\pi_\theta(y_l|x)}} - \log{\frac{\pi_{ref}(y_w|x)}{\pi_{ref}(y_l|x)}}\right)\right)\right)\right]
\]

Recall that above we optimized a negative log-likelihood loss to estimate the parameters of a reward model that was then used downstream by RLHF to estimate the parameters of a policy model. But now we are directly optimizing the parameters of our LLM _policy_ model based on human preferences! Thus, Direct Preference Optimization.

![RLHF vs. DPO Graphic](/assets/img/rlhf-vs-dpo.png)

To be explicit about the benefits of DPO over RLHF:

1. We avoid the need to train a reward model to estimate human preferences.
2. We avoid needing to perform any type of reinforcement learning, which is notoriously difficult and requires a lot of tribal knowledge to get right.
3. We can directly optimize our LLM on human preferences using supervised learning, which is a much more straightforward and well-understood process.

The avoidance of reinforcement learning is particularly important. DPO has made preference-tuning a much more accessible process for practitioners who may not have the time, resources, or expertise to navigate the complexities of reinforcement learning.

### Properties and Caveats of DPO

One of the key properties of DPO is that when the Bradley-Terry model perfectly fits our preference data and RLHF learns the optimal reward function, then the global optimizer of RLHF and DPO is the same.

This is an important equivalence result; however, in practice:

1. The Bradley-Terry model often does not perfectly fit the preference data.[^cycle]
2. The reward function learned by RLHF will not be the optimal reward function.
3. Gradient descent on a highly non-convex loss landscape - such as that of an LLM - does not find the global optimizer.

[^cycle]: For example, a preference cycle would cause the Bradley-Terry model to fail to perfectly fit the data. The Bradley-Terry model assumes transitive preferences. For example, if \(A \succ B\) and \(B \succ C\) then it expects that \(A \succ C\). But if instead \(C \succ A\), then there is a cycle and transitivity is broken.

Another weakness of DPO is that it is prone to overfitting due to a lack of regularization. [Azar et al.](https://arxiv.org/abs/2310.12036) provide a compelling example[^notation]:

[^notation]: The original notation of the quote has been adjusted slightly to match the rest of this post.

> Consider the simple example where we have two actions \(y_1\) and \(y_2\) such that \(p^*(y_1 \succ y_2)=1\), i.e., \(y_1\) is always preferred to \(y_2\). Then the Bradley-Terry model would require that \((r(y_1)-r(y_2))\rightarrow+\infty\) to \[be satisfied]. If we plug this into the optimal policy then we would get that \(\frac{\pi^*(y_2)}{\pi^*(y_1)}=0\) (i.e. \(\pi^*(y_2)=0\)) ... Thus the strength of the KL-regularization becomes weaker and weaker the more deterministic the preferences.

They also point out that, in practice, we have a finite amount of preference data. Therefore, we are likely to empirically estimate \(\hat{p}(y_1 \succ y_2)=1\) simply because we've only seen a small number of comparisons between \(y\) and \(y'\). Therefore the empirical optimal policy would push \(\pi(y_2)=0\) regardless of the regularization term that is attempting to keep the policy similar to our reference policy.

Despite these shortcomings, DPO is a highly effective tool; at the time of writing, many of the most successful and performant open-source LLMs were instruction-tuned using DPO.

## Interested in learning more?

I highly recommend reading the [DPO paper](https://arxiv.org/abs/2305.18290). In this post, we've done a deep dive into the derivation of the DPO objective, but the paper covers other points of interest, such as experimental results and additional theoretical properties.

And if you're interested in learning more about preference-tuning in general, here are additional resources that provide a deeper dive into the topic:

- [OpenAI's post on aligning language models to follow human instructions](https://openai.com/research/instruction-following) (and the [InstructGPT paper](https://arxiv.org/abs/2203.02155))
- [HuggingFace's post on fine-tuning Llama2 with DPO](https://huggingface.co/blog/dpo-trl)
- [Direct Nash Optimization](https://arxiv.org/abs/2404.03715), a recently proposed approach, avoids using the Bradley-Terry model altogether since the Bradley-Terry model fails to express complex intransitive or cyclic preference relations.

## References
<textarea id="bibtex_input" style="display:none;">
@misc{rafailov2023direct,
      title={Direct Preference Optimization: Your Language Model is Secretly a Reward Model},
      author={Rafael Rafailov and Archit Sharma and Eric Mitchell and Stefano Ermon and Christopher D. Manning and Chelsea Finn},
      year={2023},
      eprint={2305.18290},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2305.18290}
}
@misc{bertrand2023limitations,
      title={On the Limitations of Elo: Real-World Games Are Transitive, Not Additive},
      author={Quentin Bertrand and Wojciech Marian Czarnecki and Gauthier Gidel},
      year={2023},
      eprint={2206.12301},
      archivePrefix={arXiv},
      primaryClass={cs.GT},
      url={https://arxiv.org/abs/2206.12301}
}
@misc{azar2023general,
      title={A General Theoretical Paradigm to Understand Learning from Human Preferences},
      author={Mohammad Gheshlaghi Azar and Mark Rowland and Bilal Piot and Daniel Guo and Daniele Calandriello and Michal Valko and Rémi Munos},
      year={2023},
      eprint={2310.12036},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2310.12036}
}
@misc{jitkrittum2013logsumexp,
      author={Wittawat Jitkrittum},
      title={Log-Sum-Exp Trick to Prevent Numerical Underflow},
      year={2013},
      url={http://wittawat.com/posts/log-sum_exp_underflow.html}
}
@misc{geminiteam2025geminifamilyhighlycapable,
      title={Gemini: A Family of Highly Capable Multimodal Models},
      author={Gemini Team and Rohan Anil and Sebastian Borgeaud and Jean-Baptiste Alayrac and Jiahui Yu and Radu Soricut and Johan Schalkwyk and Andrew M. Dai and Anja Hauth and Katie Millican and David Silver and Melvin Johnson and Ioannis Antonoglou and Julian Schrittwieser and Amelia Glaese and Jilin Chen and Emily Pitler and Timothy Lillicrap and Angeliki Lazaridou and Orhan Firat and James Molloy and Michael Isard and Paul R. Barham and Tom Hennigan and Benjamin Lee and Fabio Viola and Malcolm Reynolds and Yuanzhong Xu and Ryan Doherty and Eli Collins and Clemens Meyer and Eliza Rutherford and Erica Moreira and Kareem Ayoub and Megha Goel and Jack Krawczyk and Cosmo Du and Ed Chi and Heng-Tze Cheng and Eric Ni and Purvi Shah and Patrick Kane and Betty Chan and Manaal Faruqui and Aliaksei Severyn and Hanzhao Lin and YaGuang Li and Yong Cheng and Abe Ittycheriah and Mahdis Mahdieh and Mia Chen and Pei Sun and Dustin Tran and Sumit Bagri and Balaji Lakshminarayanan and Jeremiah Liu and Andras Orban and Fabian Güra and Hao Zhou and Xinying Song and Aurelien Boffy and Harish Ganapathy and Steven Zheng and HyunJeong Choe and Ágoston Weisz and Tao Zhu and Yifeng Lu and Siddharth Gopal and Jarrod Kahn and Maciej Kula and Jeff Pitman and Rushin Shah and Emanuel Taropa and Majd Al Merey and Martin Baeuml and Zhifeng Chen and Laurent El Shafey and Yujing Zhang and Olcan Sercinoglu and George Tucker and Enrique Piqueras and Maxim Krikun and Iain Barr and Nikolay Savinov and Ivo Danihelka and Becca Roelofs and Anaïs White and Anders Andreassen and Tamara von Glehn and Lakshman Yagati and Mehran Kazemi and Lucas Gonzalez and Misha Khalman and Jakub Sygnowski and Alexandre Frechette and Charlotte Smith and Laura Culp and Lev Proleev and Yi Luan and Xi Chen and James Lottes and Nathan Schucher and Federico Lebron and Alban Rrustemi and Natalie Clay and Phil Crone and Tomas Kocisky and Jeffrey Zhao and Bartek Perz and Dian Yu and Heidi Howard and Adam Bloniarz and Jack W. Rae and Han Lu and Laurent Sifre and Marcello Maggioni and Fred Alcober and Dan Garrette and Megan Barnes and Shantanu Thakoor and Jacob Austin and Gabriel Barth-Maron and William Wong and Rishabh Joshi and Rahma Chaabouni and Deeni Fatiha and Arun Ahuja and Gaurav Singh Tomar and Evan Senter and Martin Chadwick and Ilya Kornakov and Nithya Attaluri and Iñaki Iturrate and Ruibo Liu and Yunxuan Li and Sarah Cogan and Jeremy Chen and Chao Jia and Chenjie Gu and Qiao Zhang and Jordan Grimstad and Ale Jakse Hartman and Xavier Garcia and Thanumalayan Sankaranarayana Pillai and Jacob Devlin and Michael Laskin and Diego de Las Casas and Dasha Valter and Connie Tao and Lorenzo Blanco and Adrià Puigdomènech Badia and David Reitter and Mianna Chen and Jenny Brennan and Clara Rivera and Sergey Brin and Shariq Iqbal and Gabriela Surita and Jane Labanowski and Abhi Rao and Stephanie Winkler and Emilio Parisotto and Yiming Gu and Kate Olszewska and Ravi Addanki and Antoine Miech and Annie Louis and Denis Teplyashin and Geoff Brown and Elliot Catt and Jan Balaguer and Jackie Xiang and Pidong Wang and Zoe Ashwood and Anton Briukhov and Albert Webson and Sanjay Ganapathy and Smit Sanghavi and Ajay Kannan and Ming-Wei Chang and Axel Stjerngren and Josip Djolonga and Yuting Sun and Ankur Bapna and Matthew Aitchison and Pedram Pejman and Henryk Michalewski and Tianhe Yu and Cindy Wang and Juliette Love and Junwhan Ahn and Dawn Bloxwich and Kehang Han and Peter Humphreys and Thibault Sellam and James Bradbury and Varun Godbole and Sina Samangooei and Bogdan Damoc and Alex Kaskasoli and Sébastien M. R. Arnold and Vijay Vasudevan and Shubham Agrawal and Jason Riesa and Dmitry Lepikhin and Richard Tanburn and Srivatsan Srinivasan and Hyeontaek Lim and Sarah Hodkinson and Pranav Shyam and Johan Ferret and Steven Hand and Ankush Garg and Tom Le Paine and Jian Li and Yujia Li and Minh Giang and Alexander Neitz and Zaheer Abbas and Sarah York and Machel Reid and Elizabeth Cole and Aakanksha Chowdhery and Dipanjan Das and Dominika Rogozińska and Vitaliy Nikolaev and Pablo Sprechmann and Zachary Nado and Lukas Zilka and Flavien Prost and Luheng He and Marianne Monteiro and Gaurav Mishra and Chris Welty and Josh Newlan and Dawei Jia and Miltiadis Allamanis and Clara Huiyi Hu and Raoul de Liedekerke and Justin Gilmer and Carl Saroufim and Shruti Rijhwani and Shaobo Hou and Disha Shrivastava and Anirudh Baddepudi and Alex Goldin and Adnan Ozturel and Albin Cassirer and Yunhan Xu and Daniel Sohn and Devendra Sachan and Reinald Kim Amplayo and Craig Swanson and Dessie Petrova and Shashi Narayan and Arthur Guez and Siddhartha Brahma and Jessica Landon and Miteyan Patel and Ruizhe Zhao and Kevin Villela and Luyu Wang and Wenhao Jia and Matthew Rahtz and Mai Giménez and Legg Yeung and James Keeling and Petko Georgiev and Diana Mincu and Boxi Wu and Salem Haykal and Rachel Saputro and Kiran Vodrahalli and James Qin and Zeynep Cankara and Abhanshu Sharma and Nick Fernando and Will Hawkins and Behnam Neyshabur and Solomon Kim and Adrian Hutter and Priyanka Agrawal and Alex Castro-Ros and George van den Driessche and Tao Wang and Fan Yang and Shuo-yiin Chang and Paul Komarek and Ross McIlroy and Mario Lučić and Guodong Zhang and Wael Farhan and Michael Sharman and Paul Natsev and Paul Michel and Yamini Bansal and Siyuan Qiao and Kris Cao and Siamak Shakeri and Christina Butterfield and Justin Chung and Paul Kishan Rubenstein and Shivani Agrawal and Arthur Mensch and Kedar Soparkar and Karel Lenc and Timothy Chung and Aedan Pope and Loren Maggiore and Jackie Kay and Priya Jhakra and Shibo Wang and Joshua Maynez and Mary Phuong and Taylor Tobin and Andrea Tacchetti and Maja Trebacz and Kevin Robinson and Yash Katariya and Sebastian Riedel and Paige Bailey and Kefan Xiao and Nimesh Ghelani and Lora Aroyo and Ambrose Slone and Neil Houlsby and Xuehan Xiong and Zhen Yang and Elena Gribovskaya and Jonas Adler and Mateo Wirth and Lisa Lee and Music Li and Thais Kagohara and Jay Pavagadhi and Sophie Bridgers and Anna Bortsova and Sanjay Ghemawat and Zafarali Ahmed and Tianqi Liu and Richard Powell and Vijay Bolina and Mariko Iinuma and Polina Zablotskaia and James Besley and Da-Woon Chung and Timothy Dozat and Ramona Comanescu and Xiance Si and Jeremy Greer and Guolong Su and Martin Polacek and Raphaël Lopez Kaufman and Simon Tokumine and Hexiang Hu and Elena Buchatskaya and Yingjie Miao and Mohamed Elhawaty and Aditya Siddhant and Nenad Tomasev and Jinwei Xing and Christina Greer and Helen Miller and Shereen Ashraf and Aurko Roy and Zizhao Zhang and Ada Ma and Angelos Filos and Milos Besta and Rory Blevins and Ted Klimenko and Chih-Kuan Yeh and Soravit Changpinyo and Jiaqi Mu and Oscar Chang and Mantas Pajarskas and Carrie Muir and Vered Cohen and Charline Le Lan and Krishna Haridasan and Amit Marathe and Steven Hansen and Sholto Douglas and Rajkumar Samuel and Mingqiu Wang and Sophia Austin and Chang Lan and Jiepu Jiang and Justin Chiu and Jaime Alonso Lorenzo and Lars Lowe Sjösund and Sébastien Cevey and Zach Gleicher and Thi Avrahami and Anudhyan Boral and Hansa Srinivasan and Vittorio Selo and Rhys May and Konstantinos Aisopos and Léonard Hussenot and Livio Baldini Soares and Kate Baumli and Michael B. Chang and Adrià Recasens and Ben Caine and Alexander Pritzel and Filip Pavetic and Fabio Pardo and Anita Gergely and Justin Frye and Vinay Ramasesh and Dan Horgan and Kartikeya Badola and Nora Kassner and Subhrajit Roy and Ethan Dyer and Víctor Campos Campos and Alex Tomala and Yunhao Tang and Dalia El Badawy and Elspeth White and Basil Mustafa and Oran Lang and Abhishek Jindal and Sharad Vikram and Zhitao Gong and Sergi Caelles and Ross Hemsley and Gregory Thornton and Fangxiaoyu Feng and Wojciech Stokowiec and Ce Zheng and Phoebe Thacker and Çağlar Ünlü and Zhishuai Zhang and Mohammad Saleh and James Svensson and Max Bileschi and Piyush Patil and Ankesh Anand and Roman Ring and Katerina Tsihlas and Arpi Vezer and Marco Selvi and Toby Shevlane and Mikel Rodriguez and Tom Kwiatkowski and Samira Daruki and Keran Rong and Allan Dafoe and Nicholas FitzGerald and Keren Gu-Lemberg and Mina Khan and Lisa Anne Hendricks and Marie Pellat and Vladimir Feinberg and James Cobon-Kerr and Tara Sainath and Maribeth Rauh and Sayed Hadi Hashemi and Richard Ives and Yana Hasson and Eric Noland and Yuan Cao and Nathan Byrd and Le Hou and Qingze Wang and Thibault Sottiaux and Michela Paganini and Jean-Baptiste Lespiau and Alexandre Moufarek and Samer Hassan and Kaushik Shivakumar and Joost van Amersfoort and Amol Mandhane and Pratik Joshi and Anirudh Goyal and Matthew Tung and Andrew Brock and Hannah Sheahan and Vedant Misra and Cheng Li and Nemanja Rakićević and Mostafa Dehghani and Fangyu Liu and Sid Mittal and Junhyuk Oh and Seb Noury and Eren Sezener and Fantine Huot and Matthew Lamm and Nicola De Cao and Charlie Chen and Sidharth Mudgal and Romina Stella and Kevin Brooks and Gautam Vasudevan and Chenxi Liu and Mainak Chain and Nivedita Melinkeri and Aaron Cohen and Venus Wang and Kristie Seymore and Sergey Zubkov and Rahul Goel and Summer Yue and Sai Krishnakumaran and Brian Albert and Nate Hurley and Motoki Sano and Anhad Mohananey and Jonah Joughin and Egor Filonov and Tomasz Kępa and Yomna Eldawy and Jiawern Lim and Rahul Rishi and Shirin Badiezadegan and Taylor Bos and Jerry Chang and Sanil Jain and Sri Gayatri Sundara Padmanabhan and Subha Puttagunta and Kalpesh Krishna and Leslie Baker and Norbert Kalb and Vamsi Bedapudi and Adam Kurzrok and Shuntong Lei and Anthony Yu and Oren Litvin and Xiang Zhou and Zhichun Wu and Sam Sobell and Andrea Siciliano and Alan Papir and Robby Neale and Jonas Bragagnolo and Tej Toor and Tina Chen and Valentin Anklin and Feiran Wang and Richie Feng and Milad Gholami and Kevin Ling and Lijuan Liu and Jules Walter and Hamid Moghaddam and Arun Kishore and Jakub Adamek and Tyler Mercado and Jonathan Mallinson and Siddhinita Wandekar and Stephen Cagle and Eran Ofek and Guillermo Garrido and Clemens Lombriser and Maksim Mukha and Botu Sun and Hafeezul Rahman Mohammad and Josip Matak and Yadi Qian and Vikas Peswani and Pawel Janus and Quan Yuan and Leif Schelin and Oana David and Ankur Garg and Yifan He and Oleksii Duzhyi and Anton Älgmyr and Timothée Lottaz and Qi Li and Vikas Yadav and Luyao Xu and Alex Chinien and Rakesh Shivanna and Aleksandr Chuklin and Josie Li and Carrie Spadine and Travis Wolfe and Kareem Mohamed and Subhabrata Das and Zihang Dai and Kyle He and Daniel von Dincklage and Shyam Upadhyay and Akanksha Maurya and Luyan Chi and Sebastian Krause and Khalid Salama and Pam G Rabinovitch and Pavan Kumar Reddy M and Aarush Selvan and Mikhail Dektiarev and Golnaz Ghiasi and Erdem Guven and Himanshu Gupta and Boyi Liu and Deepak Sharma and Idan Heimlich Shtacher and Shachi Paul and Oscar Akerlund and François-Xavier Aubet and Terry Huang and Chen Zhu and Eric Zhu and Elico Teixeira and Matthew Fritze and Francesco Bertolini and Liana-Eleonora Marinescu and Martin Bölle and Dominik Paulus and Khyatti Gupta and Tejasi Latkar and Max Chang and Jason Sanders and Roopa Wilson and Xuewei Wu and Yi-Xuan Tan and Lam Nguyen Thiet and Tulsee Doshi and Sid Lall and Swaroop Mishra and Wanming Chen and Thang Luong and Seth Benjamin and Jasmine Lee and Ewa Andrejczuk and Dominik Rabiej and Vipul Ranjan and Krzysztof Styrc and Pengcheng Yin and Jon Simon and Malcolm Rose Harriott and Mudit Bansal and Alexei Robsky and Geoff Bacon and David Greene and Daniil Mirylenka and Chen Zhou and Obaid Sarvana and Abhimanyu Goyal and Samuel Andermatt and Patrick Siegler and Ben Horn and Assaf Israel and Francesco Pongetti and Chih-Wei "Louis" Chen and Marco Selvatici and Pedro Silva and Kathie Wang and Jackson Tolins and Kelvin Guu and Roey Yogev and Xiaochen Cai and Alessandro Agostini and Maulik Shah and Hung Nguyen and Noah Ó Donnaile and Sébastien Pereira and Linda Friso and Adam Stambler and Adam Kurzrok and Chenkai Kuang and Yan Romanikhin and Mark Geller and ZJ Yan and Kane Jang and Cheng-Chun Lee and Wojciech Fica and Eric Malmi and Qijun Tan and Dan Banica and Daniel Balle and Ryan Pham and Yanping Huang and Diana Avram and Hongzhi Shi and Jasjot Singh and Chris Hidey and Niharika Ahuja and Pranab Saxena and Dan Dooley and Srividya Pranavi Potharaju and Eileen O'Neill and Anand Gokulchandran and Ryan Foley and Kai Zhao and Mike Dusenberry and Yuan Liu and Pulkit Mehta and Ragha Kotikalapudi and Chalence Safranek-Shrader and Andrew Goodman and Joshua Kessinger and Eran Globen and Prateek Kolhar and Chris Gorgolewski and Ali Ibrahim and Yang Song and Ali Eichenbaum and Thomas Brovelli and Sahitya Potluri and Preethi Lahoti and Cip Baetu and Ali Ghorbani and Charles Chen and Andy Crawford and Shalini Pal and Mukund Sridhar and Petru Gurita and Asier Mujika and Igor Petrovski and Pierre-Louis Cedoz and Chenmei Li and Shiyuan Chen and Niccolò Dal Santo and Siddharth Goyal and Jitesh Punjabi and Karthik Kappaganthu and Chester Kwak and Pallavi LV and Sarmishta Velury and Himadri Choudhury and Jamie Hall and Premal Shah and Ricardo Figueira and Matt Thomas and Minjie Lu and Ting Zhou and Chintu Kumar and Thomas Jurdi and Sharat Chikkerur and Yenai Ma and Adams Yu and Soo Kwak and Victor Ähdel and Sujeevan Rajayogam and Travis Choma and Fei Liu and Aditya Barua and Colin Ji and Ji Ho Park and Vincent Hellendoorn and Alex Bailey and Taylan Bilal and Huanjie Zhou and Mehrdad Khatir and Charles Sutton and Wojciech Rzadkowski and Fiona Macintosh and Roopali Vij and Konstantin Shagin and Paul Medina and Chen Liang and Jinjing Zhou and Pararth Shah and Yingying Bi and Attila Dankovics and Shipra Banga and Sabine Lehmann and Marissa Bredesen and Zifan Lin and John Eric Hoffmann and Jonathan Lai and Raynald Chung and Kai Yang and Nihal Balani and Arthur Bražinskas and Andrei Sozanschi and Matthew Hayes and Héctor Fernández Alcalde and Peter Makarov and Will Chen and Antonio Stella and Liselotte Snijders and Michael Mandl and Ante Kärrman and Paweł Nowak and Xinyi Wu and Alex Dyck and Krishnan Vaidyanathan and Raghavender R and Jessica Mallet and Mitch Rudominer and Eric Johnston and Sushil Mittal and Akhil Udathu and Janara Christensen and Vishal Verma and Zach Irving and Andreas Santucci and Gamaleldin Elsayed and Elnaz Davoodi and Marin Georgiev and Ian Tenney and Nan Hua and Geoffrey Cideron and Edouard Leurent and Mahmoud Alnahlawi and Ionut Georgescu and Nan Wei and Ivy Zheng and Dylan Scandinaro and Heinrich Jiang and Jasper Snoek and Mukund Sundararajan and Xuezhi Wang and Zack Ontiveros and Itay Karo and Jeremy Cole and Vinu Rajashekhar and Lara Tumeh and Eyal Ben-David and Rishub Jain and Jonathan Uesato and Romina Datta and Oskar Bunyan and Shimu Wu and John Zhang and Piotr Stanczyk and Ye Zhang and David Steiner and Subhajit Naskar and Michael Azzam and Matthew Johnson and Adam Paszke and Chung-Cheng Chiu and Jaume Sanchez Elias and Afroz Mohiuddin and Faizan Muhammad and Jin Miao and Andrew Lee and Nino Vieillard and Jane Park and Jiageng Zhang and Jeff Stanway and Drew Garmon and Abhijit Karmarkar and Zhe Dong and Jong Lee and Aviral Kumar and Luowei Zhou and Jonathan Evens and William Isaac and Geoffrey Irving and Edward Loper and Michael Fink and Isha Arkatkar and Nanxin Chen and Izhak Shafran and Ivan Petrychenko and Zhe Chen and Johnson Jia and Anselm Levskaya and Zhenkai Zhu and Peter Grabowski and Yu Mao and Alberto Magni and Kaisheng Yao and Javier Snaider and Norman Casagrande and Evan Palmer and Paul Suganthan and Alfonso Castaño and Irene Giannoumis and Wooyeol Kim and Mikołaj Rybiński and Ashwin Sreevatsa and Jennifer Prendki and David Soergel and Adrian Goedeckemeyer and Willi Gierke and Mohsen Jafari and Meenu Gaba and Jeremy Wiesner and Diana Gage Wright and Yawen Wei and Harsha Vashisht and Yana Kulizhskaya and Jay Hoover and Maigo Le and Lu Li and Chimezie Iwuanyanwu and Lu Liu and Kevin Ramirez and Andrey Khorlin and Albert Cui and Tian LIN and Marcus Wu and Ricardo Aguilar and Keith Pallo and Abhishek Chakladar and Ginger Perng and Elena Allica Abellan and Mingyang Zhang and Ishita Dasgupta and Nate Kushman and Ivo Penchev and Alena Repina and Xihui Wu and Tom van der Weide and Priya Ponnapalli and Caroline Kaplan and Jiri Simsa and Shuangfeng Li and Olivier Dousse and Fan Yang and Jeff Piper and Nathan Ie and Rama Pasumarthi and Nathan Lintz and Anitha Vijayakumar and Daniel Andor and Pedro Valenzuela and Minnie Lui and Cosmin Paduraru and Daiyi Peng and Katherine Lee and Shuyuan Zhang and Somer Greene and Duc Dung Nguyen and Paula Kurylowicz and Cassidy Hardin and Lucas Dixon and Lili Janzer and Kiam Choo and Ziqiang Feng and Biao Zhang and Achintya Singhal and Dayou Du and Dan McKinnon and Natasha Antropova and Tolga Bolukbasi and Orgad Keller and David Reid and Daniel Finchelstein and Maria Abi Raad and Remi Crocker and Peter Hawkins and Robert Dadashi and Colin Gaffney and Ken Franko and Anna Bulanova and Rémi Leblond and Shirley Chung and Harry Askham and Luis C. Cobo and Kelvin Xu and Felix Fischer and Jun Xu and Christina Sorokin and Chris Alberti and Chu-Cheng Lin and Colin Evans and Alek Dimitriev and Hannah Forbes and Dylan Banarse and Zora Tung and Mark Omernick and Colton Bishop and Rachel Sterneck and Rohan Jain and Jiawei Xia and Ehsan Amid and Francesco Piccinno and Xingyu Wang and Praseem Banzal and Daniel J. Mankowitz and Alex Polozov and Victoria Krakovna and Sasha Brown and MohammadHossein Bateni and Dennis Duan and Vlad Firoiu and Meghana Thotakuri and Tom Natan and Matthieu Geist and Ser tan Girgin and Hui Li and Jiayu Ye and Ofir Roval and Reiko Tojo and Michael Kwong and James Lee-Thorp and Christopher Yew and Danila Sinopalnikov and Sabela Ramos and John Mellor and Abhishek Sharma and Kathy Wu and David Miller and Nicolas Sonnerat and Denis Vnukov and Rory Greig and Jennifer Beattie and Emily Caveness and Libin Bai and Julian Eisenschlos and Alex Korchemniy and Tomy Tsai and Mimi Jasarevic and Weize Kong and Phuong Dao and Zeyu Zheng and Frederick Liu and Fan Yang and Rui Zhu and Tian Huey Teh and Jason Sanmiya and Evgeny Gladchenko and Nejc Trdin and Daniel Toyama and Evan Rosen and Sasan Tavakkol and Linting Xue and Chen Elkind and Oliver Woodman and John Carpenter and George Papamakarios and Rupert Kemp and Sushant Kafle and Tanya Grunina and Rishika Sinha and Alice Talbert and Diane Wu and Denese Owusu-Afriyie and Cosmo Du and Chloe Thornton and Jordi Pont-Tuset and Pradyumna Narayana and Jing Li and Saaber Fatehi and John Wieting and Omar Ajmeri and Benigno Uria and Yeongil Ko and Laura Knight and Amélie Héliou and Ning Niu and Shane Gu and Chenxi Pang and Yeqing Li and Nir Levine and Ariel Stolovich and Rebeca Santamaria-Fernandez and Sonam Goenka and Wenny Yustalim and Robin Strudel and Ali Elqursh and Charlie Deck and Hyo Lee and Zonglin Li and Kyle Levin and Raphael Hoffmann and Dan Holtmann-Rice and Olivier Bachem and Sho Arora and Christy Koh and Soheil Hassas Yeganeh and Siim Põder and Mukarram Tariq and Yanhua Sun and Lucian Ionita and Mojtaba Seyedhosseini and Pouya Tafti and Zhiyu Liu and Anmol Gulati and Jasmine Liu and Xinyu Ye and Bart Chrzaszcz and Lily Wang and Nikhil Sethi and Tianrun Li and Ben Brown and Shreya Singh and Wei Fan and Aaron Parisi and Joe Stanton and Vinod Koverkathu and Christopher A. Choquette-Choo and Yunjie Li and TJ Lu and Abe Ittycheriah and Prakash Shroff and Mani Varadarajan and Sanaz Bahargam and Rob Willoughby and David Gaddy and Guillaume Desjardins and Marco Cornero and Brona Robenek and Bhavishya Mittal and Ben Albrecht and Ashish Shenoy and Fedor Moiseev and Henrik Jacobsson and Alireza Ghaffarkhah and Morgane Rivière and Alanna Walton and Clément Crepy and Alicia Parrish and Zongwei Zhou and Clement Farabet and Carey Radebaugh and Praveen Srinivasan and Claudia van der Salm and Andreas Fidjeland and Salvatore Scellato and Eri Latorre-Chimoto and Hanna Klimczak-Plucińska and David Bridson and Dario de Cesare and Tom Hudson and Piermaria Mendolicchio and Lexi Walker and Alex Morris and Matthew Mauger and Alexey Guseynov and Alison Reid and Seth Odoom and Lucia Loher and Victor Cotruta and Madhavi Yenugula and Dominik Grewe and Anastasia Petrushkina and Tom Duerig and Antonio Sanchez and Steve Yadlowsky and Amy Shen and Amir Globerson and Lynette Webb and Sahil Dua and Dong Li and Surya Bhupatiraju and Dan Hurt and Haroon Qureshi and Ananth Agarwal and Tomer Shani and Matan Eyal and Anuj Khare and Shreyas Rammohan Belle and Lei Wang and Chetan Tekur and Mihir Sanjay Kale and Jinliang Wei and Ruoxin Sang and Brennan Saeta and Tyler Liechty and Yi Sun and Yao Zhao and Stephan Lee and Pandu Nayak and Doug Fritz and Manish Reddy Vuyyuru and John Aslanides and Nidhi Vyas and Martin Wicke and Xiao Ma and Evgenii Eltyshev and Nina Martin and Hardie Cate and James Manyika and Keyvan Amiri and Yelin Kim and Xi Xiong and Kai Kang and Florian Luisier and Nilesh Tripuraneni and David Madras and Mandy Guo and Austin Waters and Oliver Wang and Joshua Ainslie and Jason Baldridge and Han Zhang and Garima Pruthi and Jakob Bauer and Feng Yang and Riham Mansour and Jason Gelman and Yang Xu and George Polovets and Ji Liu and Honglong Cai and Warren Chen and XiangHai Sheng and Emily Xue and Sherjil Ozair and Christof Angermueller and Xiaowei Li and Anoop Sinha and Weiren Wang and Julia Wiesinger and Emmanouil Koukoumidis and Yuan Tian and Anand Iyer and Madhu Gurumurthy and Mark Goldenson and Parashar Shah and MK Blake and Hongkun Yu and Anthony Urbanowicz and Jennimaria Palomaki and Chrisantha Fernando and Ken Durden and Harsh Mehta and Nikola Momchev and Elahe Rahimtoroghi and Maria Georgaki and Amit Raul and Sebastian Ruder and Morgan Redshaw and Jinhyuk Lee and Denny Zhou and Komal Jalan and Dinghua Li and Blake Hechtman and Parker Schuh and Milad Nasr and Kieran Milan and Vladimir Mikulik and Juliana Franco and Tim Green and Nam Nguyen and Joe Kelley and Aroma Mahendru and Andrea Hu and Joshua Howland and Ben Vargas and Jeffrey Hui and Kshitij Bansal and Vikram Rao and Rakesh Ghiya and Emma Wang and Ke Ye and Jean Michel Sarr and Melanie Moranski Preston and Madeleine Elish and Steve Li and Aakash Kaku and Jigar Gupta and Ice Pasupat and Da-Cheng Juan and Milan Someswar and Tejvi M. and Xinyun Chen and Aida Amini and Alex Fabrikant and Eric Chu and Xuanyi Dong and Amruta Muthal and Senaka Buthpitiya and Sarthak Jauhari and Nan Hua and Urvashi Khandelwal and Ayal Hitron and Jie Ren and Larissa Rinaldi and Shahar Drath and Avigail Dabush and Nan-Jiang Jiang and Harshal Godhia and Uli Sachs and Anthony Chen and Yicheng Fan and Hagai Taitelbaum and Hila Noga and Zhuyun Dai and James Wang and Chen Liang and Jenny Hamer and Chun-Sung Ferng and Chenel Elkind and Aviel Atias and Paulina Lee and Vít Listík and Mathias Carlen and Jan van de Kerkhof and Marcin Pikus and Krunoslav Zaher and Paul Müller and Sasha Zykova and Richard Stefanec and Vitaly Gatsko and Christoph Hirnschall and Ashwin Sethi and Xingyu Federico Xu and Chetan Ahuja and Beth Tsai and Anca Stefanoiu and Bo Feng and Keshav Dhandhania and Manish Katyal and Akshay Gupta and Atharva Parulekar and Divya Pitta and Jing Zhao and Vivaan Bhatia and Yashodha Bhavnani and Omar Alhadlaq and Xiaolin Li and Peter Danenberg and Dennis Tu and Alex Pine and Vera Filippova and Abhipso Ghosh and Ben Limonchik and Bhargava Urala and Chaitanya Krishna Lanka and Derik Clive and Yi Sun and Edward Li and Hao Wu and Kevin Hongtongsak and Ianna Li and Kalind Thakkar and Kuanysh Omarov and Kushal Majmundar and Michael Alverson and Michael Kucharski and Mohak Patel and Mudit Jain and Maksim Zabelin and Paolo Pelagatti and Rohan Kohli and Saurabh Kumar and Joseph Kim and Swetha Sankar and Vineet Shah and Lakshmi Ramachandruni and Xiangkai Zeng and Ben Bariach and Laura Weidinger and Tu Vu and Alek Andreev and Antoine He and Kevin Hui and Sheleem Kashem and Amar Subramanya and Sissie Hsiao and Demis Hassabis and Koray Kavukcuoglu and Adam Sadovsky and Quoc Le and Trevor Strohman and Yonghui Wu and Slav Petrov and Jeffrey Dean and Oriol Vinyals},
      year={2025},
      eprint={2312.11805},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2312.11805},
}
@misc{andrychowicz2020what,
      title={What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study},
      author={Marcin Andrychowicz and Anton Raichuk and Piotr Stańczyk and Manu Orsini and Sertan Girgin and Raphaël Marinier and Léonard Hussenot and Matthieu Geist and Olivier Pietquin and Marcin Michalski and Sylvain Gelly and Olivier Bachem},
      year={2020},
      eprint={2006.05990},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2006.05990}
}
</textarea>
