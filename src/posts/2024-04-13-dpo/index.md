---
title: (WIP) Direct Preference Optimization explained in depth
subtitle: Simpler preference-tuning without reinforcement learning
date: 2024-04-13T00:00:00-08:00
tags: post
---
With my first blog post, I want to cover an excellent paper that was published last year: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) by Rafailov et al.

Commonly referred to as DPO, this method of preference-tuning is an alternative to Reinforcement Learning from Human Feedback (RLHF) that avoids the actual reinforcement learning. In this blog post, I will explain DPO from first principles; readers do not need an understanding of RLHF.

# Training, tuning, and aligning LLMs
<!-- TODO: lifecycle graphic -->
In order to contextualize DPO, and preference-tuning in general, let's review the modern process for creating language models such as ChatGPT or Claude. The following steps are sequential, with each one building upon the previous:

1. **Pre-train a base model** on internet-scale data. Given a snippet of text, this model is trained to predict the immediate next word. This conceptually simple task scales up extremely well and allows LLMs to encode a huge amount of knowledge from their training data. Examples of base models include [GPT-3](https://arxiv.org/abs/2005.14165), [Llama](https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/) (and [Llama 2](https://ai.meta.com/resources/models-and-libraries/llama/)), and [Mistral](https://mistral.ai/news/announcing-mistral-7b/).

2. Take a pre-trained base model and **fine-tune it on a task-specific dataset**. For example, if you are trying to create a helpful dialog model like ChatGPT, you would want to tune your model on a dataset of conversational dialog, so that your model's outputs sound more like parts of a conversation and less like a Wikipedia page. In this stage, we still use the next word prediction task, and the fine-tuning procedure updates our model to make predictions that more closely align with the high-quality task-specific examples we are feeding it. Examples of fine-tuned models in this stage are [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html), [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/) and [Mistral-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1).

3. Finally, we **fine-tune the model based on human preferences**. Human preferences are powerful because they are so easily and cheaply *expressed*. Think of how easy it is to compare two movies and pick a favorite. Yet how difficult it would be to make a film that embodies the qualities that drive you to visit a theater. Similarly, it is challenging to describe exactly how we want our model to behave (as we attempt to do in step 2), but given examples of model behavior it is straightforward to indicate a preference for a specific type of behavior. For a while, this sort of preference-tuning was done using RLHF. Recently, RLHF has been somewhat supplanted by DPO due to the relative simplicity of the latter. LLMs that have been tuned using human preferences include [Llama 2 Chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [ChatGPT-4](https://cdn.openai.com/papers/gpt-4-system-card.pdf), [Claude 3 Opus](https://www.anthropic.com/news/claude-3-family), and [Gemini Ultra](https://blog.google/technology/ai/google-gemini-ai/#availability).

# Tuning LLMs on preference data

It is hard and time-consuming work to create high-quality demonstrations of the behavior we want our LLM to mimic. And it would be expensive to hire labelers to help us create such data. However, once we have a model that is "good enough" at demonstrating desired behavior, we can shift into high gear. Given a prompt, we can [sample two different responses from our LLM](https://huggingface.co/blog/how-to-generate#sampling) by injecting a small amount of randomness[^temperature]. Now, it is cheap and easy to have a labeler express a preference for one of the two completions.

[^temperature]: This is typically done by generating text with a `temperature` that is greater than zero. [Here](https://lukesalamone.github.io/posts/what-is-temperature/) is a lovely little demo that explains how temperature affects model outputs visually.

While using ChatGPT or Gemini, you may have noticed that you will occasionally be asked to choose between two similar answers from which to continue your conversation. This preference is recorded and used to improve the model in a future round of preference-tuning. Similarly, [Chatbot Arena](https://chat.lmsys.org/) collects preference data for the purpose of computing Elo scores to compare LLMs:

![LMSys Chatbot Arena, a head-to-head comparison tool for instruction-tuned LLMs](/assets/img/chatbot-arena.png)

There are many publicly available preference datasets, such as LMSys' [Chatbot Arena Conversations dataset](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations), OpenAI's [WebGPT Comparisons datataset](https://huggingface.co/datasets/openai/webgpt_comparisons?row=1), and Anthropic's [Helpfulness-Harmlessness RLHF dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf) (explicit/offensive content warning).

Formally, these datasets can be expressed as follows:
$$
\mathcal{D}=\{x^{(i)},y_w^{(i)},y_l^{(i)}\}_{i=1}^N
$$
Where $x$ is the context/prompt, $y_w$ is the preferred completion, and $y_l$ is the less desirable completion.

## The Bradley-Terry Model
So what do we do with all this preference data? We want to leverage it to modify our LLM to output responses that better conform to the preferences. To begin, let us explore a simple probability model:
$$
p^*(i \succ j) = \frac{s_i}{s_i + s_j}
$$

This is the Bradley-Terry model, which is a model for the outcome of pairwise comparisons. In plain English, it says "We model the true[^star] probability that outcome $i$ is preferred to outcome $j$ as the score of $i$ over the combined scores of $i$ and $j$".

[^star]: This is the reason for the "star" in $p^*$: to indicate that we are modeling the true underlying distribution of human preferences. Likewise, shortly we will see $r^*$, which indicates the true underlying reward function that grades our completions, and $\pi^*$, which indicates the optimal policy we want our LLM to mimic.

Readers may be familiar with the Bradley-Terry model from the context of Elo scores, which are popular in [chess](https://www.chess.com/terms/elo-rating-chess) and [other](https://liquipedia.net/starcraft/Elo_rating#Detailed_Explanation) [competitive](https://www.goratings.org/en/) [games](https://lmsys.org/blog/2023-12-07-leaderboard/). The Bradley-Terry model is a generalization of the Elo rating system, where the probability of player A beating player B is given by $p(A \succ B) = \frac{1}{1 + 10^{(R_B-R_A)/400}} = \frac{s_A}{s_A + s_B}$. Here $R$ indicates a player's rating[^elo] and $s = 10^{R/400}$.

[^elo]: So if player A's Elo rating is 2000 and player B's is 1600 then player A is expected to be 10 times more likely to win than player B, because $p(A \succ B)=\frac{1}{1 + 10^{(1600-2000)/400}}=10/11$.

Under the Bradley-Terry model, is common to choose to parameterize the score as $s=e^r$, where $r$ stands for reward. The term "reward" is borrowed from the world of reinforcement learning, where greater rewards are received for a more desirable series of actions - similar to achieving a higher score for performing better in a video game.

With this parameterization, our model starts to look pretty nice - a simple difference in reward values passed through the logistic function[^logistic].
$$
p^*(i \succ j) = \frac{s_i}{s_i + s_j} = \frac{e^{r^*_i}}{e^{r^*_i} + e^{r^*_j}} = \frac{1}{1+e^{-(r^*_i-r^*_j)}} = \sigma(r^*_i - r^*_j)
$$

[^logistic]: The logistic function is an S-shaped (or sigmoid) function commonly denoted using $\sigma(x)$. It frequently appears when working with probabilities because it can "squash" values in $\mathbb{R}$ (the set of all real numbers) into  $(0, 1)$ (the set of probabilities values, excluding exactly 0 or 1). ![Sigmoid Function](/assets/img/sigmoid.png)


## Applying the Bradley-Terry Model to LLMs
Now, we want to take the Bradley-Terry model and leverage it alongside a dataset of preferences in order to improve our LLM's generated outputs.

In our preference dataset ($\mathcal{D}$), we have two comparisons and we want to model the probability of one completion being preferred over the other. In a sense, each completion elicits some reward based on its quality, and our ultimate goal will be to nudge our LLM to produce completions that are of higher quality. Therefore, we will parameterize the reward using our LLM. We will call this reward $r^*(x, y)$, which just means that the reward is a function of the context/prompt ($x$) and the completion ($y$).

So after adapting our preference model to use our parameterized reward function, we have:
$$
p^*(y_1 \succ y_2 | x) = \sigma(r^*(x, y_1) - r^*(x, y_2))
$$

But talking in terms of optimal solutions and rewards does us no good, since we do not have access to the optimal reward function. In practice, it is common to learn a reward model $r_\phi(x, y)$ that mimics the optimal reward function. We can estimate the parameters $\phi$ of this reward model by framing this as a binary classification problem where our objective is to minimize the following [negative log-likelihood loss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html) function on our preference dataset $\mathcal{D}$:[^expectation1]
$$
\mathcal{L}_R(r_\phi, \mathcal{D}) = -\mathbb{E}_{(x,y_w,y_l)\sim \mathcal{D}}[\log(\sigma(r_\phi(x,y_w) - r_\phi(x, y_l)))]
$$

[^expectation1]: {-} $\mathbb{E}_{(x,y_1,y_2)\sim \mathcal{D}}[f(x,y_w,y_l)]$ is just a formal way of saying "the expected value of function $f$ on data points sampled from our preference dataset".

Under the RLHF framework, we could leverage this learned reward model in a reinforcement learning setting to optimize an LLM to output completions that achieve high rewards. However, DPO takes a different tack - instead of the two-stage RLHF process, DPO reparameterizes the Bradley-Terry model so that we can use a similar loss function to directly to optimize the parameters of our LLM such that it produces outputs that are preferred by human observers.


## The probability of a completion
At this point, the idea of optimizing LLMs based on preferences or rewards may feel fairly abstract. So we're going to take a moment to introduce a new probability function, $\pi(y|x)$, that represents the literal output of our LLM. In reinforcement learning notation, $\pi$ indicates a policy (i.e. a strategy), and policies are optimized to maximize reward. Specifically, $\pi_\theta(y|x)$ is the probability of generating the completion $y$ based on an LLM with parameters $\theta$ given that we start with prompt $x$.

What do we mean by "the probability of generating the completion $y$"? Our LLM is an auto-regressive text generator, and, upon each auto-regressive step, it computes a probability value for every word[^token] in its vocabulary.

[^token]: In practice, modern LLMs operate on tokens, not words. For our purposes, the difference doesn't really matter. You can learn more by playing with an [online tokenizer demo](https://platform.openai.com/tokenizer) or digging through Karparthy's [minbpe](https://github.com/karpathy/minbpe) repo.

![Next Word Prediction Graphic](/assets/img/next-word-prediction.png)
So - proceeding in order through every word in completion $y$ - we compute the probability of the next word in the completion given all of the proceeding words. Now, we have a probability value for every word in the completion! So we can compute the joint probability of generating the sequence of words as the product of the individual probabilities of observing each word along the way[^logprobs]:

$$
\pi_\theta(y|x)=\prod_{t=0}^{|y|}p_{LLM_\theta}(y_t|x,y_{0:t})
$$

[^logprobs]: Multiplying probabilities can result in numerical underflow. It is common to instead work with logprobs: $\prod_i p_i=e^{\sum_i log p_i}$. Since every term in the summation of logprobs increases the magnitude of its output, underflow is avoided. OpenAI has a nice [guide to using token logprobs](https://cookbook.openai.com/examples/using_logprobs) returned by an LLM.

Another way to think about it is that there is a tree of possible completions and we are computing the probability of tracing one specific path from the root (end of the prompt) to a leaf (stop-token).

![Probability of Sequence Graphic](/assets/img/sequence-prediction.png)

When training, we know the entire text completion ahead of time, so, by applying a causal attention mask, we can calculate all of the the individual next-word probabilities (and thus $\pi_\theta(y|x)$) via a single forward-pass through our LLM.

# Optimizing our LLM based on preferences
Ok, so now that we've got our framework in place. Let us remind ourselves of our goal: to improve the outputs of our LLM. Stated another way, we want the completion (y) our LLM provides for a prompt (x) to generate a large reward $r(x, y)$. With this in mind, we can formulate an optimization problem where we want to find the parameters of our LLM ($\theta$) that maximize our expected reward for prompts similar to those we see in practice.[^expectation2]
$$
\max_{\theta}\mathbb{E}_{x\sim \mathcal{D},y\sim \pi_\theta(y|x)}[r(x, y)]
$$

[^expectation2]: {-} $\mathbb{E}_{x\sim \mathcal{D},y\sim \pi_\theta(y|x)}[r(x, y)]$ is just a formal way of saying "the expected reward attained by completions generated/sampled from our model ($y\sim \pi_\theta(y|x)$) based on prompts sampled from our dataset ($x\sim \mathcal{D}$)".

This is a bit too simplistic, however. In practice, we start with the parameters of our fine-tuned base model, and we have some belief that the outputs generated by our fine-tuned base model are pretty good, so we don't want the outputs of our model to change too much unless they improve the reward significantly. With that in mind, we amend our optimization problem to include a constraint[^kldiv] to help enforce this belief.
$$
\max_{\theta}\mathbb{E}_{x\sim \mathcal{D},y\sim \pi_\theta(y|x)}[r(x, y)] - \beta\mathbb{D}_{KL}[\pi_\theta(y|x) \ \Vert \ \pi_{ref}(y|x)]
$$

[^kldiv]: $\mathbb{D}_{KL}[P \Vert Q]$ is the [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence), a statistical distance measure. It quantifies how the probability distribution P differs from probability distribution Q. $\mathbb{D}_{KL}[P \Vert Q] = \mathbb{E}_{P}[log(\frac{P(x)}{Q(x)})]$

This constraint just encodes the idea that we want to penalize outputs from our model ($\pi_\theta$) based on how much they differ from outputs from the fine-tuned model (e.g. the reference model) we started with ($\pi_{ref}$).

Now, we want to derive the optimal solution to this optimization problem. The derivation will rely on the fact that $\mathbb{D}_{KL}[P \Vert Q]\geq0$ and $\mathbb{D}_{KL}[P \Vert Q]=0$ if and only if $P=Q$[^gibbs].

[^gibbs]: See [Gibb's Inequality](https://en.wikipedia.org/wiki/Gibbs%27_inequality). The intuition here is that the KL-divergence is a distance measure (kind of), and there is no distance between P and Q if they are equal, and there must be some distance if they are not equal.

$$
\max_{\pi_\theta}\mathbb{E}_{x\sim \mathcal{D},y\sim \pi_\theta(y|x)}[r(x, y)] - \beta\mathbb{D}_{KL}\left[\pi_\theta(y|x) \ \Vert \ \pi_{ref}(y|x)\right] \\[10pt]
=\max_{\pi_\theta}\mathbb{E}_{x\sim \mathcal{D},y\sim \pi_\theta(y|x)}[r(x, y)] - \beta\mathbb{E}_{y\sim \pi_\theta(y|x)}\left[\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}\right] \\[10pt]
= \max_{\pi_\theta}\mathbb{E}_{x\sim \mathcal{D}}\mathbb{E}_{y\sim \pi_\theta(y|x)}\left[r(x,y) - \beta\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}\right] \\[10pt]
= \min_{\pi_\theta}\mathbb{E}_{x\sim \mathcal{D}}\mathbb{E}_{y\sim \pi_\theta(y|x)}\left[\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} - \frac{1}{\beta}r(x,y)\right] \\[10pt]
= \min_{\pi_\theta}\mathbb{E}_{x\sim \mathcal{D}}\mathbb{E}_{y\sim \pi_\theta(y|x)}\left[\log\frac{\pi_\theta(y|x)}{\frac{1}{Z(x)}\pi_{ref}(y|x)e^{\frac{1}{\beta}r(x,y)}} - \log Z(x)\right] = ...
$$
where $Z(x)=\sum_y\pi_{ref}(y|x)e^{\frac{1}{\beta}r(x,y)}$. Importantly, this $Z(x)$ term depends only on $x$ and $\pi_{ref}$ and not on $y$ or $\pi_\theta$. This lets us do a bit of reorganizing from where we just left off.
$$
...= \min_{\pi_\theta}\mathbb{E}_{x\sim \mathcal{D}}\left[\mathbb{E}_{y\sim \pi_\theta(y|x)}\left[log\frac{\pi_\theta(y|x)}{\frac{1}{Z(x)}\pi_{ref}(y|x)e^{\frac{1}{\beta}r(x,y)}}\right] - logZ(x)\right] \\[10pt]
= \min_{\pi_\theta}\mathbb{E}_{x\sim \mathcal{D}}\left[\mathbb{D}_{KL}\left(\pi_\theta(y|x)\ \Vert\ \frac{1}{Z(x)}\pi_{ref}(y|x)e^{\frac{1}{\beta}r(x,y)}\right) - logZ(x)\right]
$$
And we have nearly arrived! Since $Z(x)$ does not depend on $\pi_\theta$, we can just ignore it for the purpose of deriving the optimal solution. We can now rely on the property of the KL divergence mentioned above: it is minimized at zero if, and only if, the two distributions are identical. So, the optimal solution (denoted as $\pi^*$) for all $x \in \mathcal{D}$ to our optimization problem is:
$$
\pi^*(y|x)=\pi_\theta(y|x)=\frac{1}{Z(x)}\pi_{ref}(y|x)e^{\frac{1}{\beta}r(x,y)}
$$


## Direct Preference Optimization
So we know the optimal solution to our optimization problem, but can we access it? No. The term $Z(x)=\sum_y\pi_{ref}(y|x)e^{\frac{1}{\beta}r(x,y)}$ is intractable - computing it requires summing over every possible string of words.

Instead, we can reorganize the optimal solution from above such that we express the reward function in terms of the optimal policy $\pi_\theta$, the reference policy $\pi_{ref}$, and the intractable function $Z$:
$$
r(x,y) = \beta\log{\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}} + \beta\log{Z(x)}
$$

This same reorganization can be applied using the underlying ground-truth reward $r^*$ and its corresponding optimal policy $\pi^*$.
$$
r^*(x,y) = \beta\log{\frac{\pi^*(y|x)}{\pi_{ref}(y|x)}} + \beta\log{Z(x)}
$$

Now here comes the clever trick noticed by the authors of DPO. We can use this reorganized expression of the optimal solution to our optimization problem to *reparameterize* the Bradley-Terry preference model from above so that it is expressed in terms of an optimal policy $\pi^*$ and not in terms of an underlying reward function! And even better, once we plug everything in, we notice that the intractable $Z(x)$ function cancels out!
$$
p^*(y_1 \succ y_2 | x) = \sigma(r^*(x, y_1) - r^*(x, y_2)) \\[10pt]
= \sigma\left(\beta\log{\frac{\pi^*(y_1|x)}{\pi_{ref}(y_1|x)}} + \beta\log{Z(x)} - \left(\beta\log{\frac{\pi^*(y_2|x)}{\pi_{ref}(y_2|x)}} + \beta\log{Z(x)}\right)\right) \\[10pt]
= \sigma\left(\beta\log{\frac{\pi^*(y_1|x)}{\pi_{ref}(y_1|x)}} - \beta\log{\frac{\pi^*(y_2|x)}{\pi_{ref}(y_2|x)}}\right)
$$

Now, with our reparameterized Bradley-Terry model, we can use supervised learning to directly learn a policy that mimics the optimal policy. We can minimize a negative log-likelihood loss function over our preference dataset $\mathcal{D}$ to estimate the parameters of our policy $\pi_\theta$:
$$
\mathcal{L}_{DPO}(\pi_\theta;\pi_{ref}) = -\mathbb{E}_{(y_w,y_l,x)\sim \mathcal{D}}\left[\log\left(\sigma\left(\beta\log{\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}} - \beta\log{\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}}\right)\right)\right] \\[10pt]= -\mathbb{E}_{(y_w,y_l,x)\sim \mathcal{D}}\left[\log\left(\sigma\left(\beta\left(\log{\frac{\pi_\theta(y_w|x)}{\pi_\theta(y_l|x)}} - \log{\frac{\pi_{ref}(y_w|x)}{\pi_{ref}(y_l|x)}}\right)\right)\right)\right]
$$

Recall that above we optimized a negative log-likelihood loss to estimate the parameters of a reward model that was then used downstream by RLHF to estimate the parameters of a policy model. But now we are directly optimizing the parameters of our LLM *policy* model based on human preferences! Thus, Direct Preference Optimization.

<!-- TODO: discuss why DPO is more convenient than RLFH -->

## Properties and Caveats of DPO
One of the key properties of DPO is that when the Bradley-Terry model perfectly fits our preference data and RLHF learns the optimal reward function, then the global optimizer of RHLF and DPO is the same.

This is an important equivalence result; however, in practice:
1) The Bradley-Terry model often does not perfectly fit the preference data.[^cycle]
2) The reward function learned by RLHF will not be the optimal reward function.
3) Gradient descent on a highly non-convex loss landscape - such as that of an LLM - does not find the global optimizer.

[^cycle]: For example, a preference cycle would cause the Bradley-Terry model to fail to perfectly fit the data. The Bradley-Terry model assumes transitive preferences. For example, if $A \succ B$ and $B \succ C$ then it expects that $A \succ C$. But if instead $C \succ A$, then there is a cycle and transitivity is broken.

Another weakness of DPO is that it is prone to overfitting due to a lack of regularization. [Azar et al.](https://arxiv.org/abs/2310.12036) provide a compelling example[^notation]:

[^notation]: The original notation of the quote has been adjusted slightly to match the rest of this post.

> Consider the simple example where we have two actions $y_1$ and $y_2$ such that $p^*(y_1 \succ y_2)=1$, i.e., $y_1$ is always preferred to $y_2$. Then the Bradley-Terry model would require that $(r(y_1)-r(y_2))\rightarrow+\infty$ to \[be satisfied]. If we plug this into the optimal policy then we would get that $\frac{\pi^*(y_2)}{\pi^*(y_1)}=0$ (i.e. $\pi^*(y_2)=0$) ... Thus the strength of the KL-regularization becomes weaker and weaker the more deterministic the preferences.

They also point out that, in practice, we have a finite amount of preference data. Therefore, we are likely to empirically estimate $\hat{p}(y_1 \succ y_2)=1$ simply because we've only seen a small number of comparisons between $y$ and $y'$. Therefore the empirical optimal policy would push $\pi(y_2)=0$ regardless of the regularization term that is attempting to keep the policy similar to our reference policy.

Despite these shortcomings, DPO is a highly effective tool; at the time of writing, many of the most successful and performant open-source LLMs were instruction-tuned using DPO.

# Interested in learning more?
I highly recommend reading the [DPO paper](https://arxiv.org/abs/2305.18290). In this post, we've done a deep dive into the derivation of the DPO objective, but the paper covers other points of interest, such as experimental results and additional theoretical properties.

And if you're interested in learning more about preference-tuning in general, here are additional resources that provide a deeper dive into the topic:
* [OpenAI's post on aligning language models to follow human instructions](https://openai.com/research/instruction-following) (and the [InstructGPT paper](https://arxiv.org/abs/2203.02155))
* [HuggingFace's post on fine-tuning Llama2 with DPO](https://huggingface.co/blog/dpo-trl)
* [Direct Nash Optimization](https://arxiv.org/abs/2404.03715), a recently proposed approach which avoids using the Bradley-Terry model altogether, since the Bradley-Terry model fails to express complex intransitive or cyclic preference relations.


# References

[1] Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. arXiv. https://arxiv.org/abs/2305.18290.

[2] Bertrand, Q., Czarnecki, W. M., & Gidel, G. (2023). On the limitations of Elo: Real-world games are transitive, not additive. arXiv. https://arxiv.org/abs/2206.12301.

[3] Azar, M. G., Rowland, M., Piot, B., Guo, D., Calandriello, D., Valko, M., & Munos, R. (2023). A General Theoretical Paradigm to Understand Learning from Human Preferences. arXiv. https://arxiv.org/abs/2310.12036.

[4] Jitkrittum, W. (2013). Log-Sum-Exp Trick to Prevent Numerical Underflow. http://wittawat.com/posts/log-sum_exp_underflow.html