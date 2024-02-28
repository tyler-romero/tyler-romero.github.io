---
title: Direct Preference Optimization explained without any RLHF
subtitle: How the LLM sausage gets made
date: 2024-02-27T00:00:00-08:00
tags: post
---
With my first blog post, I want to cover a great paper that was published just a month ago: [Direct Preference Optimization by Rafailov et. al.](https://arxiv.org/abs/2305.18290)

Commonly refered to as DPO, this method is equivalent to Reinforcement Learning from Human Feedback (RLHF), but avoids any actual reinforcement learning. In this blog post I will explain DPO from first principles, avoiding any sort of motivation from the perspective of RLHF.

## Training, tuning, and aligning LLMs

The modern process for creating chat models such as ChatGPT or Claude has three steps:
1. **Pre-train a base model** on internet-scale data. Given a snippet of text, this model is trained to predict the immediate next word. This simple task scales up extreamly well, and allows LLMs to encode a huge amount of knowledge from their training data. Examples of base models include [GPT-3](https://arxiv.org/abs/2005.14165), [Llama](https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/) (and [Llama 2](https://ai.meta.com/resources/models-and-libraries/llama/)), and [Mistral](https://mistral.ai/news/announcing-mistral-7b/).

2. Take a pre-trained base model and **fine-tune it on a task-specific dataset**. For example, if you are trying to create a helpful dialog model like ChatGPT, you would want to tune your model on a dataset of conversational dialog, so that your model's outputs sound more like parts of a conversation and less like a Wikipedia page. In this stage, we still use the next word prediction task, and the fine-tuning procedure updates our model to make predictions that more closely align with the high-quality task-specific examples we are feeding it. Examples of fine-tuned models in this stage are [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html), [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/) and [Mistral-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1).

3. Finally, we **fine-tune our model in accordance with human preferences**. For a while, this was done using RLHF. Recently, RLFH has been largely surplanted by Direct Preference Optimization (DPO). LLMs that have been tuned using human preferences include [Llama 2 Chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [ChatGPT-4](https://cdn.openai.com/papers/gpt-4-system-card.pdf), [Claude 2.1](https://www-cdn.anthropic.com/files/4zrzovbb/website/5c49cc247484cecf107c699baf29250302e5da70.pdf), and [Gemini Ultra](https://blog.google/technology/ai/google-gemini-ai/#availability).

## Tuning LLMs on preference data
It is hard work to create high-quality demonstrations of the desired behavior we want our LLM to mimic. And it is expensive if we want to hire lablers to help us create such data. However, once we have a model that is "good enough" at demonstrating our behavior, we can shift into high-gear. Given a prompt, we can set our LLM's temperature to value > 0 and thus sample two different responses. Now, it is cheap and easy to have a labeler express a preference for one of the two completions.

While using ChatGPT, you may have noticed that OpenAI will occasionally ask you to provide such a preference between two similar answers! Similarly, [Chatbot Arena](https://chat.lmsys.org/) collects preference data for the purpose of computing ELO scores to compare LLMs.

![LMSys Chatbot Arena](/assets/img/chatbot-arena.png)

There are many publically available preference datasets, such as LMSys' [Chatbot Arena Conversations dataset](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations), OpenAI's [WebGPT Comparisons datataset](https://huggingface.co/datasets/openai/webgpt_comparisons?row=1), and Anthropic's [Helpfulness-Harmlessness RLHF dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf) (explicit/offensive content warning).

### The Bradley-Terry Model
So what do we do with all this preference data? We want to leverage it to modify our LLM to output responses that better conform to the preferences. We begin with a simple model for this:
$$
p^*(i \succ j) = \frac{s_i}{s_i + s_j}
$$

This is the Bradley-Terry Model, which is a probability model for the outcome of pairwise comparisons. In plain english it says "we model the true[^star] probability that outcome i is prefered to outcome j as the score of i over the combined scores of i and j".

[^star]: This is the reason for the "star" in $p^*$: to indicate that we are modeling the true underlying distribution of human preferences.

<!-- TODO: explain where the term "reward" comes from -->
If we choose to parameterize the score as $s=e^r$ (which is a common choice), where $r$ is a "reward", then our model starts to look pretty nice - a simple difference in reward values passed through the sigmoid function[^sigmoid].
$$
p^*(i \succ j) = \frac{s_i}{s_i + s_j} = \frac{e^{r_i}}{e^{r_i} + e^{r_j}} = \frac{1}{1+e^{-(r_i-r_j)}} = \sigma(r_i - r_j)
$$

[^sigmoid]: ![Sigmoid Function](/assets/img/sigmoid_fn_transparent.jpg)

Now, we want to take this model of our data and leverage it to improve our LLM. In order to slot our LLM into this, we need to parameterize the reward using our LLM. We will call this reward $r_\theta(x, y)$, which just means that the reward is a function of the context/prompt ($x$) and the completion/answer ($y$) while being determined by the parameters of the LLM ($\theta$).

So after adapting our preference model to use our parameterized reward function, we have:
$$
p^*(y_1 \succ y_2 | x) = \sigma(r_\theta(x, y_1) - r_\theta(x, y_2))
$$

From here, given that we have access to a dataset of preference comparisons $\mathcal{D}=\{x^{(i)},y_w^{(i)},y_l^{(i)}\}_{i=1}^N$ (where $y_w$ is the winning completion from the pair and $y_l$ is the losing completion), we can frame this as a binary classification problem where our objective is to minimize the following [negative log-likelihood loss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html) function on our preference dataset:[^expectation1]
$$
\mathcal{L}_R(r_\theta) = -\mathbb{E}_{(x,y_w,y_l)\sim \mathcal{D}}[\log(\sigma(r_\theta(x,y_w) - r_\theta(x, y_l)))]
$$

[^expectation1]: {-} $\mathbb{E}_{(x,y_1,y_2)\sim \mathcal{D}}[f(x,y_w,y_l)]$ is just a formal way of saying "the expected value of function f on datapoints sampled from our preference dataset".

### The probability of a completion
It is time to introduce a new probabiltiy function: $\pi(y|x)$. In RL notation, $\pi$ indicates a policy - and policies are optimized to maximize reward. In this setting, our "policy" function's output is literally the output of our LLM. Specifically, $\pi_\theta(y|x)$ is the probability of the completion y based on an LLM with parameters $\theta$ given that we start with prompt x.

What do we mean by "the probability of the completion y"? Our LLM is an auto-regressive text generator, and, upon each auto-regressive step, it computes a probability value for every word[^token] in its vocabulary.

[^token]: In practice, modern LLMs operate on tokens, not words. For our purposes, the difference doesn't really matter. You can learn more by playing with an [online tokenizer demo](https://platform.openai.com/tokenizer) or digging through Karparthy's [minbpe](https://github.com/karpathy/minbpe) repo.

![Next Word Prediction Graphic](/assets/img/next-word-prediction.png)
So - proceeding in order through every word in completion y - we compute the probability of the next word in the completion given all of the proceeding words. Now, we have a probability value for every word in y! So we can compute the joint probability of the sequence of words as the product of the individual probabilities of observing each word along the way[^logprobs]:

$$
\pi_\theta(y|x)=\prod_{t=0}^{|y|}p_{LLM_\theta}(y_t|x,y_{0:t})
$$

[^logprobs]: In practice, multiplying probabilities can result in numerical underflow. It is common to instead work with logprobs: $\prod_i p_i=e^{\sum_i log p_i}$. Since every term in the summation of logprobs increases the magnitude of its output, underflow is avoided. OpenAI has a nice [guide to using token logprobs](https://cookbook.openai.com/examples/using_logprobs) returned by an LLM.

Another way to think about it is that there is a tree of possible completions and we are computing the probability of tracing one specific path from the root (prompt) to a leaf (stop-token).

![Probability of Sequence Graphic](/assets/img/sequence-prediction.png)

When traning, we know the entire text completion ahead of time, so, by applying a causal mask we can can calculate all of the the individual next-word probabilities (and thus $\pi_\theta(y|x)$) in a single forward-pass through our LLM.

### Optimizing our LLM based on preferences
Ok, so now that we've got our framework in place. Lets remind ourselves of our goal: to improve the outputs of our LLM. Stated another way, we want the completion (y) our LLM provides for a prompt (x) to generate a large reward $r_\theta(x, y)$. With this in mind, we can formulate an optimization problem where we want to find the parameters of our LLM ($\theta$) that maximize our expected reward for prompts similar to those we see in practice.[^expectation2]
$$
\max_{\theta}\mathbb{E}_{x\sim \mathcal{D},y\sim \pi_\theta(y|x)}[r_\theta(x, y)]
$$

[^expectation2]: {-} $\mathbb{E}_{x\sim \mathcal{D},y\sim \pi_\theta(y|x)}[r(x, y)]$ is just a formal way of saying "the expected reward attained by completions generated/sampled from our model ($y\sim \pi_\theta(y|x)$) based on prompts sampled from our dataset ($x\sim \mathcal{D}$)".

This is a bit too simplistic, however. In practice, we are starting with the parameters of our fine-tuned base model, and we have some belief that our fine-tuned base model is pretty good, so we don't want the outputs of our model to change too much unless they really do improve the reward. With that in mind, we amend our optimization problem to include a constraint[^kldiv] to help enforce this belief.
$$
\max_{\theta}\mathbb{E}_{x\sim \mathcal{D},y\sim \pi_\theta(y|x)}[r_\theta(x, y)] - \beta\mathbb{D}_{KL}[\pi_\theta(y|x) \ \Vert \ \pi_{ft}(y|x)]
$$
<!-- TODO: switch pi_ft to pi_ref? -->

[^kldiv]: $\mathbb{D}_{KL}[P \Vert Q]$ is the Kullback-Leibler divergence, a statistical distance measure. It quantifies how the probability distribution P differs from probability distribution Q. $\mathbb{D}_{KL}[P \Vert Q] = \mathbb{E}_{P}[log(\frac{P(x)}{Q(x)})]$

This constraint just encodes the idea that we want to penalize outputs from our model ($\pi_\theta$) more the more they differ from outputs from fine-tuned model we started with ($\pi_{ft}$).

Now, we want to derive the optimal solution to this optimization problem. The derivation will rely on the fact that $\mathbb{D}_{KL}[P \Vert Q]\geq0$ and $\mathbb{D}_{KL}[P \Vert Q]=0$ if and only if $P=Q$[^gibbs].

[^gibbs]: See [Gibb's Inequality](https://en.wikipedia.org/wiki/Gibbs%27_inequality). The intuition here is that the KL-divergence is a distance measure (kind of), and there is no distance between P and Q if they are equal, and there must be some distance if they are not equal.

$$
\max_{\pi_\theta}\mathbb{E}_{x\sim \mathcal{D},y\sim \pi_\theta(y|x)}[r_\theta(x, y)] - \beta\mathbb{D}_{KL}\left[\pi_\theta(y|x) \ \Vert \ \pi_{ft}(y|x)\right] \\[10pt]
=\max_{\\pi_\theta}\mathbb{E}_{x\sim \mathcal{D},y\sim \pi_\theta(y|x)}[r_\theta(x, y)] - \beta\mathbb{E}_{y\sim \pi_\theta(y|x)}\left[\log\frac{\pi_\theta(y|x)}{\pi_{ft}(y|x)}\right] \\[10pt]
= \max_{\pi_\theta}\mathbb{E}_{x\sim \mathcal{D}}\mathbb{E}_{y\sim \pi_\theta(y|x)}\left[r_\theta(x,y) - \beta\log\frac{\pi_\theta(y|x)}{\pi_{ft}(y|x)}\right] \\[10pt]
= \min_{\pi_\theta}\mathbb{E}_{x\sim \mathcal{D}}\mathbb{E}_{y\sim \pi_\theta(y|x)}\left[\log\frac{\pi_\theta(y|x)}{\pi_{ft}(y|x)} - \frac{1}{\beta}r_\theta(x,y)\right] \\[10pt]
= \min_{\pi_\theta}\mathbb{E}_{x\sim \mathcal{D}}\mathbb{E}_{y\sim \pi_\theta(y|x)}\left[\log\frac{\pi_\theta(y|x)}{\frac{1}{Z(x)}\pi_{ft}(y|x)e^{\frac{1}{\beta}r_\theta(x,y)}} - \log Z(x)\right] = ...
$$
where $Z(x)=\sum_y\pi_{ft}(y|x)e^{\frac{1}{\beta}r_\theta(x,y)}$. Importantly, this $Z(x)$ term depends only on x and $\pi_{ft}$ and not on y or $\pi_\theta$. This lets us do a bit of reorganizing from where we just left off.
$$
...= \min_{\pi_\theta}\mathbb{E}_{x\sim \mathcal{D}}\left[\mathbb{E}_{y\sim \pi_\theta(y|x)}\left[log\frac{\pi_\theta(y|x)}{\frac{1}{Z(x)}\pi_{ft}(y|x)e^{\frac{1}{\beta}r_\theta(x,y)}}\right] - logZ(x)\right] \\[10pt]
= \min_{\pi_\theta}\mathbb{E}_{x\sim \mathcal{D}}\left[\mathbb{D}_{KL}\left(\pi_\theta(y|x)\ \Vert\ \frac{1}{Z(x)}\pi_{ft}(y|x)e^{\frac{1}{\beta}r_\theta(x,y)}\right) - logZ(x)\right]
$$
And we have nearly arrived! Since $Z(x)$ does not depend on $\pi_\theta$, we can just ignore it for the purpose of deriving the optimal solution. We can now rely on the property of the KL divergence mentioned above: it is minimized at 0 if and only if the two distributions are identical. So the optimal solution (denoted as $\pi^*$) to our optimization problem is:
$$
\pi^*(y|x)=\pi_\theta(y|x)=\frac{1}{Z(x)}\pi_{ft}(y|x)e^{\frac{1}{\beta}r_\theta(x,y)}
$$


### Direct Preference Optimization
So we know the optimal solution to our optimization problem, but can we compute it? No. Even if we had access to the ground-truth reward function, computing the term $Z(x)=\sum_y\pi_{ft}(y|x)e^{\frac{1}{\beta}r_\theta(x,y)}$ is intractable - it would require computing the probability of observing every single possible string of words.

Instead, we can reorganize the optimal solution from above such that we isolate the reward function:
$$
r_\theta(x,y) = \beta\log{\frac{\pi_\theta(y|x)}{\pi_{ft}(y|x)}} + \beta\log{Z(x)}
$$
<!-- TODO: optimality astricts on pi? -->

<!-- TODO: better explanation connecting the optimization problem and bradley-terry -->
If we plug this into our Bradley-Terry preference model from above, we notice that the intractable $Z(x)$ function cancels out!
$$
p^*(y_1 \succ y_2 | x) = \sigma(r_\theta(x, y_1) - r_\theta(x, y_2)) \\[10pt]
= \sigma\left(\beta\log{\frac{\pi_\theta(y_1|x)}{\pi_{ft}(y_1|x)}} + \beta\log{Z(x)} - \left(\beta\log{\frac{\pi_\theta(y_2|x)}{\pi_{ft}(y_2|x)}} + \beta\log{Z(x)}\right)\right) \\[10pt]
= \sigma\left(\beta\log{\frac{\pi_\theta(y_1|x)}{\pi_{ft}(y_1|x)}} - \beta\log{\frac{\pi_\theta(y_2|x)}{\pi_{ft}(y_2|x)}}\right)
$$

Now we can formulate a loss function analogous to the NLL loss function from above.
$$
\mathcal{L}_{DPO}(\pi_\theta;\pi_{ft}) = -\mathbb{E}_{(y_w,y_l,x)\sim \mathcal{D}}\left[\log\left(\sigma\left(\beta\log{\frac{\pi_\theta(y_w|x)}{\pi_{ft}(y_w|x)}} - \beta\log{\frac{\pi_\theta(y_l|x)}{\pi_{ft}(y_l|x)}}\right)\right)\right]
$$

This loss function allows us to directly optimize our LLM (via gradient descent) using preference data, thus "Direct Preference Optimization"[^rlhf].

[^rlhf]: "Direct" is in contrast to RLHF, which has the same optimization objective, but achieves it in a roundabout manner - by learning a reward model and then using reinforcement learning on the rewards from the reward model to update the LLM.

It is illustrative to examine the gradient of the DPO loss function with respect to the parameters of our LLM:
$$
\nabla_\theta\mathcal{L}_{DPO}(\pi_\theta;\pi_{ft}) \\[10pt]
=-\nabla_\theta\mathbb{E}_{(y_w,y_l,x)\sim \mathcal{D}}\left[\log\left(\sigma\left(\beta\log{\frac{\pi_\theta(y_w|x)}{\pi_{ft}(y_w|x)}} - \beta\log{\frac{\pi_\theta(y_l|x)}{\pi_{ft}(y_l|x)}}\right)\right)\right] \\[10pt]
=-\nabla_\theta\mathbb{E}_{(y_w,y_l,x)\sim \mathcal{D}}\left[\log\left(\sigma(r_\theta(x, y_1) - r_\theta(x, y_2)))\right)\right]  \\[10pt]
=-\mathbb{E}\left[\frac{\sigma(r_\theta(x, y_1) - r_\theta(x, y_2))}{\sigma(r_\theta(x, y_1) - r_\theta(x, y_2))}\nabla_\theta(r_\theta(x, y_1) - r_\theta(x, y_2))\right]
$$
<!-- TODO: finish gradient section -->

## Experiments with DPO
TODO

# References

[1] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct Preference Optimization: Your Language Model is Secretly a Reward Model, 2023. arXiv:2305.18290.
