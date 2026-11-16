---
title: "How Modern LLM Text Watermarks Work"
subtitle: Keyed sampling, exponential races, and speculative decoding
date: 2026-11-15T00:00:00-08:00
blurb: "A visual walkthrough of token-level watermarking for language models, with derivations and Qwen3-8B samples."
tags: ["post", "llm", "watermarking", "sampling", "speculative-decoding", "textseal", "synthid"]
math: true
code: true
margin_image: /assets/img/watermark-burning-fuses-anim.gif
margin_image_alt: "Four candidate tokens, warm, crisp, foggy, and 72°F, drawn as fuses that burn toward zero at speeds equal to their probabilities. Warm reaches zero first."
---

Over the past two years, major model providers have turned on invisible text watermarks. Google deployed [SynthID-Text](https://www.nature.com/articles/s41586-024-08025-4) in Gemini. Then [Article 50 of the EU AI Act](https://artificialintelligenceact.eu/article/50/) took effect in August 2026, and since then [Anthropic has watermarked Claude](https://www.anthropic.com/news/claude-text-watermark) and [OpenAI has rolled out textGrain](https://openai.com/index/eu-text-provenance/) for ChatGPT in Europe.

Meanwhile, researchers worked out most of the statistical problems that were left in token-level watermarking, in a burst of papers that ran from Meta's [TextSeal](https://arxiv.org/abs/2605.12456) in May 2026 to its successor, [TextSeal2](https://arxiv.org/abs/2610.40613), in October. This post builds up those methods one problem at a time. I'll follow one running example, "The morning was \_\_\_," and test each idea on real completions from Qwen3-8B.

## Why watermarking is controversial

Explaining how these watermarks work isn't the same as endorsing them everywhere. On *Stratechery*, [Ben Thompson argues](https://stratechery.com/2026/anthropics-watermarking-how-it-probably-works-worse-than-it-seems/) that universal watermarking gets authorship backwards, because it flags human drafts that a model merely edited. Researchers have also shown that statistical watermarks can be stripped by paraphrasing ([Zhang et al., 2023](https://arxiv.org/abs/2311.04378)) or learned from API queries and forged onto human writing ([Jovanović et al., 2024](https://arxiv.org/abs/2402.19361)).

Proponents, including [Anthropic](https://www.anthropic.com/news/claude-text-watermark) and [OpenAI](https://openai.com/index/eu-text-provenance/), point out that a watermark comes with a calibrated false-positive rate, unlike the [style classifiers that mislabel non-native English writing as AI-generated](https://arxiv.org/abs/2304.02819). They also argue that watermarks help keep synthetic text out of future training data ([Shumailov et al., 2024](https://www.nature.com/articles/s41586-024-07566-4)), raise the cost of automated spam, and meet the marking requirement in [Article 50](https://artificialintelligenceact.eu/article/50/).

## Why one random number isn't enough

Policy questions aside, the technical problem is a neat one. At detection time, the verifier gets only the finished text and a secret key. It can't see the prompt, the model weights, or the token probabilities.

We'll follow a single next-token step throughout the post. Suppose the text so far is "The morning was \_\_\_," and the model spreads its probability across four candidates:

- "warm" with \(p_{\text{warm} } = 0.50\)
- "crisp" with \(p_{\text{crisp} } = 0.30\)
- "foggy" with \(p_{\text{foggy} } = 0.15\)
- "72°F" with \(p_{72^\circ\mathrm{F} } = 0.05\)

Ordinary sampling lays those four probabilities end to end on \([0, 1]\) and draws one uniform random number \(U\). Whichever bucket \(U\) lands in is the next token. A natural first idea for a watermark is to compute \(U\) by hashing the preceding words with a secret key \(k\), instead of drawing it from a hardware random number generator. Anyone who holds the key could then recompute \(U\) and check it against the text. The left half of the figure below tests this idea, and the right half shows the fix.

<figure class="fullwidth">
  <img src="/assets/img/watermark-one-dart-vs-keyed.png" sizes="(max-width: 760px) 150vw, 1300px" alt="Two panels. Left, one shared random number: the sampler hashes 'The morning was' with the key and gets U = 0.63, which lands in the crisp bucket of a bar that splits [0, 1] into warm, crisp, foggy, and 72°F. The verifier hashes the same words with the key, also gets U = 0.63, and sees that the next word was crisp. Its copy of the bar is blank and marked p = ?, because without the model's probabilities it can't tell where the crisp bucket sits. Right, one keyed number per candidate: the sampler hashes the prefix, each candidate v, and the key to get R_v, which is 0.64 for warm, 0.60 for crisp, 0.80 for foggy, and 0.85 for 72°F. Raising each R_v to the power 1/p_v gives 0.41, 0.18, 0.23, and 0.04, so warm wins. The verifier hashes 'The morning was', warm, and the key and gets R = 0.64 directly.">
</figure>

The left half shows the problem. Suppose the sampler's hash gives \(U = 0.63\), which lands in the "crisp" bucket. Later, a verifier reads "The morning was crisp." Hashing "The morning was" with the key gives the same \(U = 0.63\), and the next word is "crisp." To confirm that the two agree, the verifier would have to check that 0.63 falls inside the "crisp" bucket. But the edges of that bucket come from the model's probabilities \(p\), which the verifier doesn't have. That's why the verifier's copy of the bar in the figure is blank and marked \(p = ?\).

The fix, on the right, is to **give every candidate token its own keyed random number**. For each token \(v\) in the vocabulary, we feed three inputs to a pseudorandom function (PRF), a keyed hash whose outputs look like uniform draws on \((0, 1)\). The inputs are a short window of preceding tokens \(x_{\text{prev} }\) (the "prefix" in the figure), the candidate \(v\), and the key \(k\):

\[
R_v = \mathrm{PRF}_k(x_{\text{prev} }, v)
\]

The sampler still has to turn these numbers into a choice. The rule [derived below](#sampling-without-distortion) raises each \(R_v\) to the power \(1/p_v\) and picks the largest result. In the figure, that's "warm," at \(0.64^2 = 0.41\) (★). The two halves of the figure pick different words because they use different random numbers.

Checking the result is now simple. A verifier reading "The morning was warm" hashes ("The morning was", "warm", \(k\)) on a CPU and gets \(R_{\text{warm} } = 0.64\) directly, without knowing \(p\). The window covers only the last three or four tokens, so the verifier never needs the prompt. It can slide across a quoted paragraph and recover \(R_{w_t}\) for the token \(w_t\) at almost every position \(t\).[^dedup]

[^dedup]: If the same \((x_{\text{prev} }, w_t)\) pair shows up twice in a passage, the hash returns the same number both times. Counting the repeat as a fresh draw would inflate false positives on repetitive human writing, such as song lyrics or code loops, so detectors score each distinct pair only once.

On human text, these recovered values are spread uniformly on \((0, 1)\). If the sampler tends to pick tokens whose \(R_v\) is close to 1, the verifier can spot that shift across a paragraph in microseconds. We'd like that sampler to have three properties at once:

1. **Distortion-free:** Averaged over keys, the probability of emitting token \(v\) is still exactly \(p_v\).
2. **Varied reruns:** Asking the same question twice under the same deployment key still gives different answers.
3. **Fast serving:** The rule works with speculative decoding and doesn't add slow processing after every forward pass.

Getting one or two of these is easy. Getting all three took almost four years.

## Shifting the logits: green and red lists

The best-known early LLM watermark, from [Kirchenbauer et al. (2023)](https://arxiv.org/abs/2301.10226), is also the simplest way to favor keyed tokens. Instead of giving each token a keyed number, it gives each token a keyed color. At each step, the sampler hashes the preceding token with the key to split the vocabulary at random into a _green list_ \(G\), which holds a fraction \(\gamma\) of the vocabulary (say half), and a _red list_ with everything else. Then it adds a constant \(\delta\) to the logit of every green token before the softmax:

\[
\tilde{p}_v = \frac{p_v \, e^{\delta \, \mathbf{1}[v \in G]} }{\sum_{u} p_u \, e^{\delta \, \mathbf{1}[u \in G]} }
\]

Detection is a simple count. A writer who doesn't know the key lands on a green token with probability \(\gamma\) at each step. So over \(T\) scored tokens, the number of green tokens \(|S|_G\) has mean \(\gamma T\), and the detector reports how many standard deviations above that mean the count sits:

\[
z = \frac{|S|_G - \gamma T}{\sqrt{T \gamma (1 - \gamma)} }
\]

The drawback is that adding \(\delta\) changes the model's probabilities, and the change doesn't average out over keys. In our running example, suppose "crisp" and "72°F" land on the green list and \(\delta = 2.5\). Then "crisp" jumps from 0.30 to 0.74, and "warm" falls from 0.50 to 0.10.[^green-red-fig]

[^green-red-fig]: {-+} ![Bar chart of the four candidates before and after the green-list bonus, with gray bars for the original probabilities and colored bars for the new ones. Warm, a red token, falls from 0.50 to 0.10. Crisp, a green token, rises from 0.30 to 0.74. Foggy, red, falls from 0.15 to 0.03. 72°F, green, rises from 0.05 to 0.12.](/assets/img/watermark-green-red.png) Gray bars show the model's probabilities \(p\). Colored bars show \(\tilde{p}\) after adding \(\delta = 2.5\) to the logits of the two green tokens, "crisp" and "72°F." The red tokens shrink because the softmax renormalizes.

That distortion has a cost. A \(\delta\) large enough to detect a 150-token paragraph raises perplexity and nudges the model toward odder phrasing. On code, the tradeoff is especially sharp. When one token has \(p_v = 0.999\), a mild \(\delta\) barely changes the choice and adds little signal, while a large one introduces bugs.

To see the signal on real text, I asked Qwen3-8B to explain why the sky changes color at noon, at sunset, and during a lunar eclipse, using \(\gamma = 0.5\) and \(\delta = 2.5\). (I ran every method in this post on the same prompt and collected the results in [one table](#comparing-the-full-lineage-on-qwen3-8b).) Of the 214 scored tokens, 61.2% landed on the green list. That gives a \(p\)-value of \(5.08 \times 10^{-4}\), which means that text written without the key would show this much green only about once in 2,000 tries. That's strong detection, but it comes from warping the softmax at every step.

## Sampling without distortion

Green and red lists get their signal by bending the model's probabilities. A better sampler would favor tokens with large \(R_v\) and still pick each token with probability **exactly \(p_v\)**. [Scott Aaronson (2023)](https://scottaaronson.blog/?p=6823) found such a rule while building an early watermarking prototype at OpenAI.

Simply picking the token with the largest \(R_v\) won't work. That would make every word in a 150,000-token vocabulary equally likely, which ignores the model entirely. We need a handicap that sinks unlikely tokens unless their draw is extremely close to 1. Raising each draw to the power \(1/p_v\) turns out to be exactly the right handicap:

\[
w_t = \arg\max_v \; R_v^{1/p_v}
\]

This is the rule behind the ★ in the first figure. Because \(R_v\) is between 0 and 1, a large exponent drags it toward zero. In the morning example, "warm" only squares its draw, so \(0.64^2 = 0.41\). "72°F" has a luckier draw of 0.85, but it has to raise it to the twentieth power, and \(0.85^{20} = 0.04\). So "warm" wins. Over random keys, the chance that token \(v\) wins this contest is exactly \(p_v\):[^gumbel-proof]

\[
\Pr\left(R_v^{1/p_v} > \max_{u \neq v} R_u^{1/p_u}\right) = p_v
\]

[^gumbel-proof]: Fix \(R_v = r\). A rival \(u\) loses when \(R_u^{1/p_u} < r^{1/p_v}\), which is the same as \(R_u < r^{p_u / p_v}\). Since \(R_u\) is uniform, that happens with probability \(r^{p_u / p_v}\). Multiplying over all the independent rivals gives \(r^{(1 - p_v)/p_v}\), and integrating over \(r\) from 0 to 1 gives \(p_v\).

### Burning fuses

Exponents like \(R^{20}\) are hard to picture, especially once a draft model and a target model start to interact later in this post. Logarithms make the rule easier to see. Define \(E_v = -\ln R_v\). This turns a draw near 1 into a short length near 0, and because \(R_v\) is uniform, \(E_v\) follows a standard exponential distribution, \(\mathrm{Exp}(1)\). Since \(-\ln R_v^{1/p_v} = E_v / p_v\), picking the largest \(R_v^{1/p_v}\) is the same as picking the smallest \(E_v / p_v\).

Picture each candidate as a fuse of length \(E_v\) that burns toward zero at speed \(p_v\), so that it runs out at time \(E_v / p_v\). The first fuse to reach zero wins, and its token is emitted. The animation shows the morning race, with a clock \(t\) that counts the time since the fuses were lit:

![Four horizontal fuses with sparks burning toward zero on a shared axis. Warm, with probability 0.50, starts at length 0.45. Crisp, 0.30, starts at 0.51. Foggy, 0.15, starts at 0.22. 72°F, 0.05, starts at 0.16. A clock counts up from t = 0. Warm burns fastest and reaches zero first at t = 0.89, while the other three fuses still have length left.](/assets/img/watermark-burning-fuses-anim.gif)

"72°F" drew a short fuse (0.16), and "warm" drew one nearly three times as long (0.45). But "warm" burns ten times faster, so it reaches zero first, at \(t \approx 0.89\). "72°F" wouldn't finish until \(t = 3.25\).

This race is the classic Gumbel-Max trick for sampling. The quantity \(-\ln E_v\) follows a standard Gumbel distribution, and the winning token is the one with the largest \(\ln p_v - \ln E_v\). That's why Aaronson's scheme is usually called the **Gumbel-Max watermark**.

The verifier never sees the race. It sees only the winners, so it scores each one by how short its fuse was. Detectors that follow [Aaronson (2023)](https://scottaaronson.blog/?p=6823) and [Fernandez et al. (2023)](https://arxiv.org/abs/2308.00113) give the token \(w_t\) at position \(t\) the score

\[
s_t = -\ln\left(1 - R_{w_t}\right)
\]

which is large when \(R_{w_t}\) is close to 1, that is, when the winning fuse was short. On human text, each \(s_t\) follows \(\mathrm{Exp}(1)\) and has a mean of 1, so the total over \(n\) scored tokens follows a Gamma distribution:

\[
\sum_{t=1}^{n} s_t \sim \mathrm{Gamma}(n, 1)
\]

That gives an exact \(p\)-value without the model. Under the watermark, short fuses win more often than chance would predict, so the average score rises above 1. How far it rises depends on how close the races are:

**High-entropy prose.** When several words have similar probabilities, their fuses burn at similar speeds, so the shortest fuse usually wins. That's the token with the largest \(R_v\), and the scores climb.

**Low-entropy code and facts.** When one token holds nearly all the probability, such as \(p_v = 0.999\) for a colon in Python, its fuse burns a thousand times faster than any rival's. It wins almost regardless of its length. The key barely changes the outcome, so the token carries almost no signal. Asking Qwen3-8B for a Python `binary_search` shows the extreme case. It wrote the exact same 88-token function with ordinary sampling and with TextSeal2, the distortion-free scheme that this post builds toward. The watermarked copy had a mean score of 0.924 and a \(p\)-value of 0.75:

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

There were almost no close races in those 88 tokens. The key rarely changed which token won, so a distortion-free sampler had no choice but to emit the canonical code.[^code-green]

[^code-green]: The green-list method can't do much better here without breaking things. Even with a strong bias of \(\delta = 4\), its run on the same prompt reached a \(p\)-value of only 0.22.

### Why pure Gumbel-Max fails in production

Gumbel-Max has the first property: averaged over keys, one completion is indistinguishable from ordinary sampling. It adds almost no work to sampling, and its verifier needs only the key. Deploying it behind an API, though, exposes two flaws, one for each of the other two properties.

**Flaw 1: one key, one answer.** A provider usually serves every request with one fixed key, since the verifier has to know which key to check. With the key fixed, every fuse length is a deterministic function of the last few tokens, and every burn speed is a deterministic function of the prefix. No randomness is left. Two users who send the same prompt get the same completion word for word. And if a short phrase ever repeats inside a long passage, the sampler faces the same fuses again and can fall into a loop.

I ran Qwen3-8B three times with a fixed key, asking for a two-sentence mystery opening about a foggy lighthouse. All three runs returned this exact text:

> "The lighthouse stood like a ghost in the swirling fog, its beam slicing through the darkness with an eerie, rhythmic pulse, as if the tower itself were breathing. A single footstep echoed down the spiral stairs, lost in the silence, but the wind carried the whisper of something watching from the shadows beyond the door."

**Flaw 2: rejected drafts.** Fast inference servers often use speculative decoding. A small draft model proposes several tokens, and the large target model checks them all in one pass. When the target rejects a draft token, it samples a replacement from a corrected distribution. There's no obvious way to run that replacement step as a keyed race, so the replacements usually lose the watermark. [A later section](#surviving-speculative-decoding) shows how TextSeal2 keeps them.

## Restoring variety

Fixing flaw 1 means bringing back fresh randomness at runtime without washing out the signal that the verifier checks. Three ideas appeared over two years, and each one fixed a weakness of the one before.

### Tournament sampling (SynthID-Text)

Google DeepMind's [SynthID-Text](https://www.nature.com/articles/s41586-024-08025-4) (Dathathri et al., 2024) splits the choice into two stages. Fresh randomness picks the candidates, and the key picks among them. At each step, the sampler draws \(2^m\) candidates from \(p\) and seats them in an \(m\)-round knockout bracket. In each round, every surviving token gets a keyed score of 0 or 1. The token with the higher score advances, and a coin flip breaks ties.

Because the candidates are drawn fresh on every call, reruns vary. The winners still tend to collect extra 1s, and the verifier counts them. On the sky prompt, with \(m = 30\), the emitted tokens averaged 15.56 ones out of their 30 keyed scores, against 15 expected by chance, for a \(p\)-value of \(2.49 \times 10^{-3}\).

With two candidates per match, SynthID-Text is exactly distortion-free over keys. (Larger matches give a stronger signal but distort the output.) It has two costs, though. The bracket adds 30 rounds of work after every forward pass, and its detection drops sharply under speculative decoding in [TextSeal2's benchmarks](https://arxiv.org/abs/2610.40613).

### Flipping a coin between keys (TextSeal)

In May 2026, Sander, Fernandez, and colleagues at Meta returned to the single race, which needs no extra work after the forward pass. Their method, [TextSeal](https://arxiv.org/abs/2605.12456), comes with a proof that \(s_t\) is the optimal score for a verifier that doesn't have the model. To restore variety, TextSeal picks one of two secret keys at random for each token. A routing probability sets how often the second key is used, and I used a fair coin, which gives the most variety. Every step stays exactly distortion-free, and \(T\) tokens have \(2^T\) possible key paths.

The price is detection power. The verifier doesn't know which key chose the token at position \(t\), so it has to average the scores from both keys. With a fair coin, one of those two scores is pure \(\mathrm{Exp}(1)\) noise. Averaging cuts the excess above 1 in half but shrinks the standard deviation by only \(\sqrt{2}\), so the signal-to-noise ratio drops by a factor of \(\sqrt{2}\), or \(\sqrt{K}\) for \(K\) keys. On the sky prompt, the \(p\)-value was \(1.04 \times 10^{-2}\) for single-key Gumbel-Max and \(3.70 \times 10^{-2}\) for TextSeal.

### Mixing noise into the fuses (AutoMark and TextSeal2)

The coin flip loses signal because each token is tied to only one of the two keys, and the verifier can't tell which. In September 2026, [Gloaguen, Staab, and Vechev](https://arxiv.org/abs/2609.37310) at ETH Zurich proposed a fix in AutoMark: keep a single key, and blend a controlled amount of fresh noise into every fuse before the race starts. The blend is a **Gaussian copula**.[^autoresearch]

[^autoresearch]: The AutoMark authors found this construction with an automated search loop. They gave a coding agent an evaluation harness that measured generation quality, diversity, and detection rate, and they let the agent propose and test changes to the sampler. The agent converged on blending the keyed variable with fresh Gaussian noise.

First, map each keyed draw \(R_v\) to a standard normal value \(G_v = \Phi^{-1}(R_v)\), where \(\Phi\) is the standard normal CDF and \(\Phi^{-1}\) is its inverse. Then draw fresh noise \(\varepsilon_v\) from a standard normal, \(N(0, 1)\), and mix the two with a dial \(\rho\) between 0 and 1:

\[
Z_v = \rho \, G_v + \sqrt{1 - \rho^2} \, \varepsilon_v
\]

\[
E_v = -\ln \Phi(Z_v)
\]

Because \(G_v\) and \(\varepsilon_v\) are independent standard normals, \(Z_v\) is also exactly standard normal for every \(\rho\). Passing it back through \(\Phi\) gives a uniform number, so \(E_v\) is still a proper \(\mathrm{Exp}(1)\) fuse. Averaged over keys and noise, the output distribution doesn't move. The variety comes from the fresh \(\varepsilon_v\), and because every token is generated and scored under the same key, none of the signal is split across keys.

The top row of the figure below shows this recipe, with the step from \(R_v\) to \(G_v\) folded into the hash. The grid below it runs the morning race with one fixed key. Each row is a setting of \(\rho\), and each column is a separate call. The bars show each token's burn time \(E_v / p_v\), the moment when its fuse reaches zero. The shortest bar wins, so it's colored and marked with ★. A "›" means that a bar runs past the edge.

<figure class="fullwidth">
  <img src="/assets/img/watermark-copula-mix.png" sizes="(max-width: 760px) 150vw, 1400px" alt="Top: a keyed hash of the prefix, token, and key gives G_v, and fresh noise from N(0, 1) gives ε_v. The two are blended as ρ·G_v + √(1 − ρ²)·ε_v and turned into a fuse E_v = −ln Φ(·). Below, a grid with three rows of ρ and three calls shows the burn times of warm, crisp, foggy, and 72°F, with the shortest bar colored and starred as the winner. At ρ = 1, every call has identical bars and warm wins every time. At ρ = 0.9, the bars shift a little from call to call, and the winners are warm, foggy, and warm. At ρ = 0, the bars change completely, and the winners are 72°F, warm, and crisp.">
</figure>

At \(\rho = 1\), there's no noise. All three calls have identical bars, and "warm" wins every time, which is deterministic Gumbel-Max again. At \(\rho = 0\), the key drops out, and the winners (72°F, warm, crisp) are ordinary unwatermarked samples. At \(\rho = 0.9\), the bars shift a little from call to call. That's enough for "foggy" to win one call, while each fuse stays closely tied to its keyed value. So \(\rho\) works as a dial between variety and watermark strength.

Rerunning the real lighthouse prompt three times at \(\rho = 0.9\), with the same fixed key, gives three different openings instead of three clones:[^short-text]

1. "The lighthouse stood like a skeletal sentinel against the churning sea, its beam slicing through the thick, ghostly fog that clung to the rocks like a shroud. A single, unmarked envelope slipped from the rusted mailbox, its contents whispering of secrets buried in the mist."
2. "The lighthouse stood like a ghost in the swirling fog, its beam slicing through the gray like a blade carving through silence. Somewhere in the dark, a sound echoed—soft, deliberate, and far too close for comfort."
3. "The lighthouse stood like a ghost against the swirling fog, its beam cutting through the mist with an eerie, pulsing rhythm that echoed the heartbeat of the storm. Somewhere in the dark, a light had gone out—leaving only the whispers of the sea and the silence of an abandoned keep."

[^short-text]: Two of the three runs still open with "The lighthouse stood like a ghost," just like the Gumbel-Max clone, because each fuse stays correlated with its keyed value. Texts this short carry little evidence, though. The three runs had \(p\)-values of 0.33, 0.050, and 0.0071 on 59, 46, and 61 tokens. For comparison, the 65-token Gumbel-Max clone had a \(p\)-value of \(2.67 \times 10^{-3}\).

In October, OpenAI published [textGrain](https://cdn.openai.com/pdf/e9508624-d767-41b6-a26d-e34ca798ada6/textgrain-entropy-calibrated-watermarking-for-language-model-text.pdf) (Li et al., 2026), which treats the tradeoff between variety and signal as an entropy budget. Tying each choice to the key removes some of the randomness that would otherwise make reruns differ. The parameter \(\beta\) caps how much of the model's next-token entropy the watermark may remove, so \(\beta = 0.5\) allows it to remove up to half. To stay within that budget, textGrain solves a small optimal-transport problem with Sinkhorn iterations at every token. At \(\beta = 0.5\), it reached a \(p\)-value of \(7.23 \times 10^{-3}\) on the sky prompt. The extra step costs time, though. The TextSeal2 authors measured 1–3 ms per token for it in their implementation (0.2–0.4 ms in a minimal one), and that work has to finish before the next token can start.

Later in October, the TextSeal team released [TextSeal2](https://arxiv.org/abs/2610.40613), which replaces its coin flip with AutoMark's copula.[^virtual-key] The authors showed that an optimization like textGrain's has little room to improve on the copula. On 5,169 real next-token distributions from Qwen3.5-9B, the copula reached, on average, about 97% of the best possible signal-to-noise ratio per token for the entropy that it removes. Without speculative decoding, TextSeal2 detected about as many answers as textGrain in their benchmarks, with no optimization step after the forward pass. TextSeal2 also fixes flaw 2, as the next section shows.

[^virtual-key]: TextSeal2 actually keeps two keys, but for a different reason. One key can be made public so that anyone can run a weaker check, while the provider keeps the other private. Before adding the runtime noise, TextSeal2 merges the two keys in normal space as \((G_v^{(1)} + G_v^{(2)}) / \sqrt{2}\), which is still a single standard normal. Every fuse depends on both keys, so the full detector, which checks both, loses nothing to a coin flip.

## Surviving speculative decoding

TextSeal2's second idea fixes [flaw 2](#why-pure-gumbel-max-fails-in-production). Under speculative decoding, a fast draft model with probabilities \(q\) proposes several tokens ahead, and the target model, with probabilities \(p\), checks them all in one parallel pass. In my runs, Qwen3-0.6B drafted for Qwen3-8B.

The trouble comes from **rejection**. Suppose the drafter runs the morning race at speeds \(q_v\), with \(q_{\text{warm} } = 0.35\), \(q_{\text{crisp} } = 0.50\), \(q_{\text{foggy} } = 0.10\), and \(q_{72^\circ\mathrm{F} } = 0.05\). Because the drafter favors "crisp," its fuse finishes first, at time \(\tau = E_{\text{crisp} } / q_{\text{crisp} } = 1.02\). The target gives "crisp" only \(p_{\text{crisp} } = 0.30\), so it accepts the draft with probability \(0.30 / 0.50 = 0.6\). Otherwise, it samples a replacement from the residual distribution

\[
r_v \propto \max(0, \; p_v - q_v)
\]

which works out to \(r = (0.75, 0, 0.25, 0)\) for warm, crisp, foggy, and 72°F.

This replacement step is where earlier systems lost the key. The original fuses can't simply race again at speeds \(r_v\). Every rival already lost to "crisp," so we know that its fuse was longer than \(\tau q_v\), and reusing the full fuses would bias the result.

Earlier systems therefore sampled the replacement without a watermark or switched to a second key. Either way, they gave up the signal of the main key on the target-sampled tokens, which made up 38–44% of the text in my runs. The TextSeal2 authors found that replacements drawn under a second key even score 0.84 on average under the main key. That's below the human average of 1, because each replacement lost the draft race, so its fuse under the main key tends to be long. These tokens actively dilute the evidence.

TextSeal2's **residual race** avoids that loss by reusing what's left of each fuse at time \(\tau\):

\[
E'_v = E_v - \tau \, q_v
\]

Exponential variables are **memoryless**. Knowing that a fuse was longer than \(\tau q_v\) says nothing about how much longer it was, so the stubs \(E'_v\) behave like fresh \(\mathrm{Exp}(1)\) fuses that are still tied to the key. The target then burns those stubs at speeds \(r_v\). This race picks token \(v\) with probability exactly \(r_v\), so the output still follows \(p\), and the replacement carries the watermark too. In TextSeal2's experiments, target-sampled tokens score 1.34 on average, close to the 1.42 of the drafted tokens that were kept.

The figure below shows both stages. In the left panel, the draft race runs at speeds \(q\) until "crisp" finishes at \(\tau = 1.02\). Each gray bar is a full fuse, and the colored part is the stub that's left at \(\tau\). The number beside each bar gives the length of that stub. "Crisp" has nothing left, and the ✗ marks the target's rejection. In the right panel, "crisp" is crossed out, and the target burns the remaining stubs at speeds \(r\). Here, the numbers give the extra time that each stub needs. "Warm" finishes after +0.12, ahead of "foggy" at +0.48, so it wins (★). "72°F" has \(r = 0\), so its stub never finishes (∞).

<figure class="fullwidth">
  <img src="/assets/img/watermark-leftover-fuses.png" sizes="(max-width: 760px) 150vw, 1300px" alt="Two panels. Left, the draft model burns the fuses at speeds q: warm q = 0.35, crisp q = 0.50, foggy q = 0.10, and 72°F q = 0.05. Crisp reaches zero first at τ = 1.02, and the target rejects it, marked 0 ✗. The other fuses have stubs left: warm 0.09, foggy 0.12, and 72°F 0.11. Right, the target burns those same stubs at the residual speeds r ∝ max(0, p − q): warm r = 0.75, foggy r = 0.25, and 72°F r = 0, with crisp crossed out. Warm reaches zero after +0.12, before foggy at +0.48, and 72°F never finishes.">
</figure>

Real drafters propose a block of several tokens at a time. [Block verification (Sun et al., 2024)](https://arxiv.org/abs/2403.10444) decides how much of the block to accept by looking at the whole block at once, instead of one token at a time. TextSeal2 extends the residual race to these blocks without changing the output distribution or the acceptance rate.

### Speculative decoding on Qwen3

To see how much the residual race matters in practice, I ran speculative decoding on the same sky prompt, with Qwen3-0.6B drafting 8-token blocks for Qwen3-8B. I compared three runs. The first was unwatermarked. The second was a fallback baseline that uses AutoMark's copula but switches to a second key on rejected steps, and its detector checks both keys. The third was TextSeal2's residual race. The last two columns of the table count the emitted tokens by source and give the average number of tokens per call to the target model:

| Method | Mean score | \(p\)-value | Drafted / target-sampled | Tokens per target call |
|:---|:---|:---|:---|:---|
| Unwatermarked | 1.029 | 0.333 | 129 / 79 | 2.60 |
| AutoMark + second-key fallback | 1.052 | 0.150 | 134 / 81 | 2.62 |
| **TextSeal2 residual race** | **1.167** | **0.0155** | 105 / 81 | 2.27 |

In all three runs, about 40% of the emitted tokens came from the target model. The fallback, which moves those tokens to a second key, reached a \(p\)-value of 0.150. The residual race, which keeps them on the main key, reached 0.0155.[^spec-split] One run can't isolate the effect of the residual race, but the direction matches TextSeal2's own benchmarks. There, the residual race detected 28 to 33 percentage points more answers than drawing those tokens under an independent key.[^throughput]

[^spec-split]: The target-sampled tokens weren't the strong ones in this run. Drafted tokens averaged a score of 1.39, and target-sampled tokens averaged only 0.88, well below the 1.34 that TextSeal2 reports. The setups differ, and a single 186-token answer is noisy. Even so, three of the ten highest-scoring tokens were target-sampled, and over many answers, these tokens add evidence under the residual race instead of diluting it.

[^throughput]: The residual race also emitted fewer tokens per target call in this run, 2.27 against 2.60 without a watermark. That gap comes from a single 186-token answer. By construction, the residual race keeps the acceptance rate of block verification, and over 480 prompts, the TextSeal2 authors measured no drop in tokens per round.

Here's one target-sampled token that did carry a strong signal. The drafter had pruned " scatter" from its top-\(p\) nucleus, so \(q_{\text{scatter} } = 0\), and its fuse didn't burn at all during the draft race. When Qwen3-8B rejected the draft, " scatter" had \(p_{\text{scatter} } = 0.679\) and still had its whole fuse. It won the residual race with a score of 3.93.[^scatter-match]

[^scatter-match]: The same token gets the exact same score, 3.93, in the non-speculative TextSeal2 sample below. In both runs, " scatter" follows the same three tokens, "blue and violet," and its hash depends only on those tokens and the key.

## Comparing the full lineage on Qwen3-8B

With both flaws fixed, we can line up every method on the sky prompt, along with unwatermarked sampling. Every run used Qwen3-8B with the same key and settings, without speculative decoding.[^lineage-setup] The full prompt was: *"Explain why the sky is blue at noon but turns deep orange-red at sunset, and why the Moon still looks faintly copper during a total lunar eclipse. Write two lively paragraphs (about 160 words total)."*

[^lineage-setup]: All runs used `temperature = 1.0`, `top_p = 0.95`, and key 42. The scored-token counts are the distinct (prefix, token) pairs that the detector checked. The green-list method reports a fraction of green tokens and SynthID reports a tournament score, so the \(p\)-value is the only measure of detection strength that's comparable across all methods.

In the table, the three columns after the method name track the three properties from the start of the post. "Partial" in the speculative decoding column means that a method runs under speculative decoding but loses some of its signal on the tokens that the target model samples. Green/red lists lose nothing, because their watermark is only a change to \(p\), and speculative decoding samples exactly from the target model's distribution, warped or not. The last column gives the \(p\)-value of each method's answer:

<div class="fullwidth">

| Method | Distortion-free | Varied reruns | Speculative decoding | Scored tokens | \(p\)-value |
|:---|:---|:---|:---|:---|:---|
| Unwatermarked | Yes | Yes | Yes | 208 | 0.281 |
| [Green/red lists](https://arxiv.org/abs/2301.10226), \(\delta = 2.5\) | **No** | Yes | Yes | 214 | \(5.08 \times 10^{-4}\) |
| [Gumbel-Max](https://scottaaronson.blog/?p=6823) | Yes | **No** | Partial | 215 | \(1.04 \times 10^{-2}\) |
| [SynthID-Text](https://www.nature.com/articles/s41586-024-08025-4), 30 rounds | Yes | Yes | Partial | 190 | \(2.49 \times 10^{-3}\) |
| [TextSeal](https://arxiv.org/abs/2605.12456), two keys | Yes | Yes | Partial | 208 | \(3.70 \times 10^{-2}\) |
| [textGrain](https://cdn.openai.com/pdf/e9508624-d767-41b6-a26d-e34ca798ada6/textgrain-entropy-calibrated-watermarking-for-language-model-text.pdf), \(\beta = 0.5\) | Yes | Yes | Partial | 193 | \(7.23 \times 10^{-3}\) |
| **[TextSeal2](https://arxiv.org/abs/2610.40613), \(\rho = 0.9\)** | **Yes** | **Yes** | **Yes** | 220 | **\(1.69 \times 10^{-4}\)** |

</div>

Any single answer involves some luck in which tokens come up. On the sky prompt, for example, Gumbel-Max got a weaker \(p\)-value than TextSeal2, even though deterministic Gumbel-Max, with no fresh noise at all, should carry more signal per token. So I also ran all seven samplers on 24 open-ended explanation prompts and tracked how detection builds up token by token.

The figure below plots the median \(p\)-value after \(t\) scored tokens. The vertical axis uses a log scale, with stronger evidence higher up. The two dashed curves are methods that give up one of the first two properties: green/red lists, which warp \(p\), and Gumbel-Max, which repeats itself. The solid curves are the methods that are both distortion-free and varied. Unwatermarked text, in gray, hovers near a \(p\)-value of 0.5. The dashed horizontal line marks a \(p\)-value of 0.01, which corresponds to a 1% false-positive rate (FPR).

<figure class="fullwidth">
  <img src="/assets/img/watermark-confidence-by-tokens.png" sizes="(max-width: 760px) 150vw, 1200px" alt="Line chart of the median p-value against the number of scored tokens, from 0 to 250, across 24 Qwen3-8B explanation prompts. The vertical axis is a log scale from 1 at the bottom to 10⁻⁵ at the top, and a dashed horizontal line marks a 1% false-positive rate at p = 0.01. Unwatermarked text, in gray, stays near p = 0.5. Dashed curves for green/red lists and Gumbel-Max cross the line first, at about 85 and 105 tokens, and reach about 2.5 × 10⁻⁵ and 3.7 × 10⁻⁵ by 250 tokens. Solid curves for SynthID-Text and TextSeal2 cross at about 125 and 135 tokens and reach about 2.9 × 10⁻⁴ and 1.9 × 10⁻⁴. textGrain crosses at about 155 tokens and reaches 8.4 × 10⁻⁴. Two-key TextSeal crosses last, at about 170 tokens, and reaches 2.8 × 10⁻³.">
</figure>

For the first 50 tokens, every curve stays well short of the line. Green/red lists cross it first, at about 85 tokens, and Gumbel-Max follows at about 105. That speed comes at the cost of one of those two properties. SynthID-Text and TextSeal2 cross at about 125 to 140 tokens, roughly one long paragraph, while staying distortion-free and varied. OpenAI's textGrain crosses at about 155 tokens. Two-key TextSeal trails at about 170 to 185, as expected from the coin-flip dilution.

Reading the unwatermarked and TextSeal2 sky answers next to each other shows what "distortion-free" looks like in practice. Both are ordinary Qwen3-8B prose, and both open with the same sentence: "At noon, the sky appears blue because sunlight travels a shorter path through the atmosphere." Then they split:

> **Unwatermarked** (\(p\)-value 0.281): "During the day, when the Sun is high, its light passes through less of the Earth’s atmosphere, allowing more blue wavelengths to reach our eyes. The shorter wavelengths of blue light scatter more easily in the atmosphere, creating the brilliant blue sky we see. …"

> **TextSeal2** (\(p\)-value \(1.69 \times 10^{-4}\)): "As sunlight enters Earth's atmosphere, shorter wavelengths like blue and violet scatter more easily due to Rayleigh scattering. Since our eyes are more sensitive to blue than violet, the sky looks blue. …"

Nothing in the wording gives the watermark away. The verifier finds it only by hashing each token with the key.

The figure below shows what the verifier sees. The top panel gives the score \(s_t\) of each of the first 16 TextSeal2 tokens after the two answers split. The dashed line at 1 marks the average score on human text, and bars above it are blue. The bottom panel tracks the running total \(\sum (s_t - 1)\) over the full TextSeal2 and unwatermarked answers. On human text, each term has mean 0 and variance 1, so after \(t\) tokens the total usually stays within about \(\pm 2\sqrt{t}\) of zero. The gray band marks that range.

<figure class="fullwidth">
  <img src="/assets/img/watermark-detector-trace.png" sizes="(max-width: 760px) 150vw, 1200px" alt="Top: the score s_t of each of the first 16 TextSeal2 tokens after the answers split, 'As sunlight enters Earth 's atmosphere , shorter wavelengths like blue and violet scatter more easily', with a dashed line at the human average of 1. The scores are 0.37, 1.38, 1.68, 2.08, 0.61, 1.14, 1.95, 0.09, 0.02, 1.11, 1.32, 0.84, 0.13, 3.93, 0.06, and 1.21. Bars above 1 are blue. Bottom: the running total Σ(s_t − 1) over about 220 scored tokens, with a gray ±2√t band that shows the typical range for human text. The unwatermarked answer wanders inside the band and ends near 8, with p = 0.28. The TextSeal2 answer climbs steadily out of the band and ends near 57, with p = 1.7 × 10⁻⁴.">
</figure>

Individual scores are noisy, just as they would be on human text: "wavelengths" scores 0.02, and "scatter" scores 3.93. The evidence comes from the running total. The two curves overlap for the first 17 or so tokens, while the answers share their opening sentence. After that, the unwatermarked total wanders inside the band and ends near 8 (\(p\)-value 0.28), and the TextSeal2 total climbs steadily out of the band and ends near 57 (\(p\)-value \(1.7 \times 10^{-4}\)).

## Finding watermarks in messy documents

The trace follows one clean answer that the model wrote from start to finish. Real documents are messier. They mix boilerplate with prose, or paste one AI-written paragraph into a long human report. The TextSeal papers pair the sampler with three detector extensions for those cases:

- **Entropy weighting.** Predictable function words and punctuation carry almost no signal and only add noise to the sum. At detection time, the verifier can run an auxiliary language model to estimate how predictable each position is, and then weight the unpredictable positions more heavily. That model can be small: the first TextSeal paper used a 4-bit model with about 0.8B parameters. A Gamma approximation turns the weighted sum back into a \(p\)-value. In TextSeal2's experiments, entropy weighting raised the share of 200-token answers detected at a 0.1% false-positive rate from 91% to 97%, and from 89% to 95% under speculative decoding.
- **Sliding windows.** Averaging over a 12,000-token report buries a 400-token pasted span in noise. With precomputed running sums, the detector can check windows of every power-of-two length in milliseconds, and a Bonferroni correction keeps the false-positive rate calibrated.
- **Distillation.** Students fine-tuned on watermarked outputs, especially long reasoning traces, pick up faint preferences for the tokens that the teacher favored ([Sander et al., 2024](https://arxiv.org/abs/2402.14904)). After deduplicating every scored pair across prompts, the TextSeal authors detected the watermark in a 3B student model that had been distilled on watermarked math reasoning traces. The \(p\)-value was \(3.7 \times 10^{-110}\).

## What watermarks still cannot do

The problems with the sampler now have clean answers. The limits that remain come from information theory and from how people use text.

Low-entropy text is the first limit. A distortion-free watermark can only hide its signal in randomness that the model already has. Short factual answers, arithmetic, and canonical code like `binary_search` have almost none, so they can't carry a watermark without a loss in quality.

Rewriting is the second. The detector checks exact windows of a few tokens, so rewording a passage, for example with an unwatermarked local model, breaks the windows and erases the trace. The TextSeal2 authors found that paraphrasing with an LLM removes almost every watermark, as expected for token-level schemes.

The last limit is about who holds the key ([Milis et al., 2026](https://arxiv.org/abs/2610.12106)). If the provider keeps the key private, outsiders have to take its word for every verdict. If it publishes a detection key, anyone can use that key to strip or forge the watermark offline. TextSeal2's two keys offer a middle ground. Anyone can run a weaker check with the public key, while the private key supports full-strength verification and can expose tampering with the public signal.

The harder questions, about trust and authorship, are the ones that this post started with.

## References

{% raw %}
<textarea id="bibtex_input" style="display:none;">
@inproceedings{kirchenbauer2023watermark,
      title={A Watermark for Large Language Models},
      author={John Kirchenbauer and Jonas Geiping and Yuxin Wen and Jonathan Katz and Ian Miers and Tom Goldstein},
      booktitle={Proceedings of the 40th International Conference on Machine Learning},
      year={2023},
      url={https://arxiv.org/abs/2301.10226}
}
@misc{aaronson2023watermarking,
      title={Watermarking of Large Language Models},
      author={Scott Aaronson},
      year={2023},
      howpublished={Simons Institute Talk and Blog},
      url={https://scottaaronson.blog/?p=6823}
}
@inproceedings{fernandez2023bricks,
      title={Three Bricks to Consolidate Watermarks for Large Language Models},
      author={Pierre Fernandez and Antoine Chaffin and Karim Tit and Vivien Chappelier and Teddy Furon},
      booktitle={IEEE International Workshop on Information Forensics and Security (WIFS)},
      year={2023},
      url={https://arxiv.org/abs/2308.00113}
}
@inproceedings{sander2024watermarking,
      title={Watermarking Makes Language Models Radioactive},
      author={Tom Sander and Pierre Fernandez and Alain Durmus and Matthijs Douze and Teddy Furon},
      booktitle={Advances in Neural Information Processing Systems},
      year={2024},
      url={https://arxiv.org/abs/2402.14904}
}
@inproceedings{sun2024block,
      title={Block Verification Accelerates Speculative Decoding},
      author={Ziteng Sun and Uri Mendlovic and Yaniv Leviathan and Asaf Aharoni and Jae Hun Ro and Ahmad Beirami and Ananda Theertha Suresh},
      booktitle={Advances in Neural Information Processing Systems},
      year={2024},
      url={https://arxiv.org/abs/2403.10444}
}
@article{dathathri2024synthid,
      title={Scalable Watermarking for Identifying Large Language Model Outputs},
      author={Sumanth Dathathri and Abigail See and Sumedh Ghaisas and Po-Sen Huang and Rob McAdam and Johannes Welbl and Vandana Bachani and Alex Kaskasoli and Robert Stanforth and Tatiana Matejovicova and Jamie Hayes and Pushmeet Kohli},
      journal={Nature},
      volume={634},
      pages={818--823},
      year={2024},
      url={https://www.nature.com/articles/s41586-024-08025-4}
}
@misc{sander2026textseal,
      title={TextSeal: A Localized LLM Watermark for Provenance \& Distillation Protection},
      author={Tom Sander and Hongyan Chang and Sylvestre-Alvise Rebuffi and Tom{\'a}{\v{s}} Sou{\v{c}}ek and Tuan Tran and Valeriu Lacatusu and Alexandre Mourachko and Surya Parimi and Christophe Ropers and Rashel Moritz and Vanessa Stark and Hady Elsahar and Pierre Fernandez},
      year={2026},
      eprint={2605.12456},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.12456}
}
@misc{gloaguen2026automark,
      title={AutoMark: Enabling Autoresearch to Discover Better LLM Watermarks},
      author={Thibaud Gloaguen and Robin Staab and Martin Vechev},
      year={2026},
      eprint={2609.37310},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.37310}
}
@misc{li2026textgrain,
      title={textGrain: Entropy-Calibrated Watermarking for Language Model Text},
      author={Xiang Li and Garrett Wen and Xiaohong Chen and Qi Long and Arzav Jain and Florent Joly and Mike Lam and Qingquan Song and Weijie Su},
      year={2026},
      howpublished={OpenAI Technical Report},
      url={https://cdn.openai.com/pdf/e9508624-d767-41b6-a26d-e34ca798ada6/textgrain-entropy-calibrated-watermarking-for-language-model-text.pdf}
}
@misc{sander2026textseal2,
      title={TextSeal2: Distortion-Free LLM Watermarking with Copula Mixing and Memoryless Races},
      author={Tom Sander and Pierre Fernandez and Itai Gat and Haiming Bao and Jiaxin Shi and Jason Park and Valeriu Lacatusu and Sylvestre-Alvise Rebuffi and Hongyan Chang and Hady Elsahar and Tom{\'a}{\v{s}} Sou{\v{c}}ek and Tuan Tran and Surya Parimi and Alexandre Mourachko},
      year={2026},
      eprint={2610.40613},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2610.40613}
}
@misc{milis2026public,
      title={Could LLM Watermark Detection Be Public?},
      author={Georgios Milis and Tom Sander and Tom{\'a}{\v{s}} Sou{\v{c}}ek and Po-Sen Huang and Pierre Fernandez},
      year={2026},
      eprint={2610.12106},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2610.12106}
}
@misc{thompson2026watermarking,
      title={Anthropic's Watermarking, How It (Probably) Works, Worse Than It Seems},
      author={Ben Thompson},
      year={2026},
      howpublished={Stratechery},
      url={https://stratechery.com/2026/anthropics-watermarking-how-it-probably-works-worse-than-it-seems/}
}
@misc{anthropic2026watermark,
      title={Introducing Invisible Text Watermarking for Claude},
      author={{Anthropic}},
      year={2026},
      url={https://www.anthropic.com/news/claude-text-watermark}
}
@misc{zhang2023sand,
      title={Watermarks in the Sand: Impossibility of Strong Watermarking for Generative Models},
      author={Hanlin Zhang and Benjamin L. Edelman and Danilo Francati and Daniele Venturi and Giuseppe Ateniese and Boaz Barak},
      year={2023},
      eprint={2311.04378},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2311.04378}
}
@inproceedings{jovanovic2024stealing,
      title={Watermark Stealing in Large Language Models},
      author={Nikola Jovanovi{\'c} and Robin Staab and Martin Vechev},
      booktitle={Proceedings of the 41st International Conference on Machine Learning},
      year={2024},
      url={https://arxiv.org/abs/2402.19361}
}
@article{liang2023gptdetectors,
      title={GPT Detectors Are Biased Against Non-Native English Writers},
      author={Weixin Liang and Mert Yuksekgonul and Yining Mao and Eric Wu and James Zou},
      journal={Patterns},
      volume={4},
      number={7},
      year={2023},
      url={https://arxiv.org/abs/2304.02819}
}
@article{shumailov2024collapse,
      title={AI Models Collapse When Trained on Recursively Generated Data},
      author={Ilia Shumailov and Zakhar Shumaylov and Yiren Zhao and Nicolas Papernot and Ross Anderson and Yarin Gal},
      journal={Nature},
      volume={631},
      pages={755--759},
      year={2024},
      url={https://www.nature.com/articles/s41586-024-07566-4}
}
</textarea>
{% endraw %}
