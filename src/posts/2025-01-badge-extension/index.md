---
title: An Extension to BADGE Active Learning for Variable-Sized Batches
subtitle: BADGE (like other greedy acquisition algorithms) can be trivially extended to provide an ordering over the pool of unlabeled data
date: 2025-01-05T00:00:00-08:00
blurb: We show how BADGE's batch selection strategy can be adapted to handle flexible batch sizes without compromising its ability to select diverse, informative samples - enabling more practical active learning workflows.
tags: ["post", "machine-learning", "active-learning", "BADGE", "batch-active-learning", "k-means++", "MNIST", "multidimensional-scaling", "uncertainty-sampling"]
---

[Batch Active learning by Diverse Gradient Embeddings (BADGE)](https://arxiv.org/abs/1906.03671) is a simple and effective batch-based [active learning](https://en.wikipedia.org/wiki/Active_learning_(machine_learning)) method for neural networks that has been empirically shown to outperform other active learning approaches in a variety of settings.

BADGE selects data points by examining their gradient embeddings - vectors that combine both the model's uncertainty (through gradient magnitude) and the data's semantic content (through gradient direction). By choosing points with diverse, high-magnitude gradients, BADGE identifies samples that are both informative (high uncertainty) and representative of different regions of the input space.[^uncertainty] This helps the model learn efficiently by focusing on challenging examples while maintaining good coverage of the dataset distribution.

[^uncertainty]: [Uncertainty sampling](https://lilianweng.github.io/posts/2022-02-20-active-learning/#uncertainty-sampling) is a common classical active learning strategy that only leverages the model's uncertainty to select points for labeling.
    In a batch setting, this can degenerate into selecting points that are all very similar, leading to suboptimal performance. Take this example on MNIST, where uncertainty sampling selects eight similar-looking `1`s in a batch of nine:
    ![Uncertainty sampling on MNIST](/assets/img/uncertainty_selected_digits.png)

One limitation of BADGE is that it requires the user to specify the batch size in advance. This can be a problem in practice, as the user may not know how many data points to label in each batch. In this post, I will show how BADGE can be extended to support variable-sized batches to enable applications where the next batch size is unknown a priori.

## BADGE Overview

Before we dive into the extension, let's review how BADGE works. BADGE is a batch active learning method that selects a batch of data points that are diverse and high-magnitude in the gradients they may induce once labeled. The algorithm proceeds as follows:

### 1. Compute Gradient Embeddings
First, we compute the gradients of the loss with respect to the final layer of the neural network. These gradients capture both the model's uncertainty (magnitude) and the data's semantic content (direction).
The first issue that arises is that - since our overall goal is to solicit labels for the most informative *unlabeled* data points - we don't have access to the true labels for these data points.
In order to compute the gradients, we need to "hallucinate" labels for the unlabeled data points. This is typically done by using the current model to predict the expected label for each data point.[^hallucinate]
The expected label is then used to compute the gradient.

[^hallucinate]: This approach of using the model's predicted label provides a lower bound on the true gradient magnitude - since the model would view any other label as even more unexpected.

BADGE uses the gradient of the final linear layer's weights, which can be computed efficiently without needing to backpropagate through the entire network.

Here is a reference implementation for computing the gradient embedding of a data point:

```python
from torch import Tensor
from torch.nn import functional as F

@torch.no_grad()
def compute_gradient_embedding(model: nn.Module, data_point: Tensor) -> Tensor:
    """Compute gradient embedding for a single data point using BADGE.

    Args:
        model: A trained neural network model
        data_point: Input tensor of shape [*feature_dims]

    Returns:
        Gradient embedding tensor of shape [hidden_dim * num_classes]
    """
    model.eval()
    x = data_point.unsqueeze(0)  # add batch dimension

    # Compute the output and final layer activations of the model
    logits, final_acts = model(x)
    probs = F.softmax(logits, dim=1)

    # Use predicted class to simulate what gradient would be if this were the true label
    hallucinated_class = logits.argmax(dim=1)
    y_hat = F.one_hot(hallucinated_class, num_classes=logits.shape[1]).float()

    # Compute gradient of cross entropy loss wrt final layer weights
    # For cross entropy loss, this is simply (prediction - target)
    loss_gradient = (probs - y_hat)  # [num_classes]

    # Compute gradient embedding by outer product of:
    # - final_acts: the last layer's activations (semantic information)
    # - loss_gradient: the gradient of loss w.r.t. logits (uncertainty)
    # This is equivalent to the gradient w.r.t the final layer weights
    grad_embedding = torch.outer(
        final_acts.squeeze(0),  # [hidden_size]
        loss_gradient           # [num_classes]
    ).flatten()  # [hidden_size * num_classes]

    return grad_embedding
```

### 2. Select Diverse and Informative Points
Finally, BADGE selects the next batch of data points by choosing points that maximize the diversity of their gradient embeddings. This is achieved using [k-means++ initialization](http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf), a greedy algorithm originally designed for seeding k-means clustering[^dpp]. The core idea of k-means++ is to iteratively select points that are far away from the points already chosen.

[^dpp]: It's worth noting the connection between k-means++ and [Determinantal Point Processes (DPPs)](https://en.wikipedia.org/wiki/Determinantal_point_process). DPPs are probabilistic models that encourage the selection of diverse subsets of items. While finding the optimal diverse subset according to a DPP can be computationally expensive, k-means++ can be viewed as a computationally efficient, greedy approximation to sampling from a DPP where the similarity between items is inversely related to the distance between their gradient embeddings.

The k-means++ algorithm proceeds as follows:
1. Initialize the first point by selecting one uniformly at random from the unlabeled pool.
2. For each subsequent point to be selected, compute the squared Euclidean distance from each unlabeled point to the nearest point that has already been selected.
3. Select the next point with a probability proportional to this squared distance. That is, a point $x_i$ is selected with probability $\frac{d^2(x_i, S)}{\sum_{x_j \in U} d^2(x_j, S)}$, where $S$ is the set of already-selected points, $U$ is the set of unlabeled points, and $d^2(x_i, S) = \min_{x_j \in S} ||\mathbf{g}(x_i) - \mathbf{g}(x_j)||^2$, with $\mathbf{g}(x_i)$ being the gradient embedding of point $x_i$.
4. Repeat steps 2 and 3 until the desired number of points have been selected.

If a batch size of $k$ is specified, BADGE uses k-means++ to greedily select $k$ diverse points from the unlabeled pool based on their gradient embeddings.

The strength of k-means++ lies in its ability to efficiently select a set of points that are well-spread out in the gradient embedding space. By probabilistically favoring points that are far from the already selected points, k-means++ encourages the selection of a diverse set of examples. This diversity is crucial for active learning because it ensures that the newly labeled data points cover a broader range of the data distribution, leading to more informative updates to the model. Furthermore, the tendency to select points far from existing selections *implicitly favors points with higher magnitude embeddings*, as these points are likely to reside in less densely populated regions of the embedding space.

Scikit-learn provides [an implementation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.kmeans_plusplus.html) of the k-means++ centroid initialization algorithm, which can be readily adapted for BADGE.

### A Demonstration with MNIST
Here we demonstrate BADGE on the [MNIST dataset](https://huggingface.co/datasets/ylecun/mnist). We randomly sample half of the MNIST dataset as our training set, subsequently dropping all `0` and `1` class examples (in order to help illustrate BADGE collecting useful examples).
We then train a simple convolutional neural network on this training set[^confusion] and use BADGE to select the next batch of data points to label from a large pool of unlabeled data points.

[^confusion]: The confusion matrix shows that the model consistently misclassifies 0s and 1s, as expected since these classes were excluded from training.
    0s are most often misclassified as 6s and 1s as 4s.
    ![Confusion matrix showing model predictions](/assets/img/mnist_confusion_matrix.png)

We compute gradient embeddings for a heldout set of 15,000 unlabeled points and use [Multidimensional Scaling (MDS)](https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#multidimensional-scaling)[^MDS] to reduce the dimensionality of the embeddings to 2D for visualization.
The points are arranged by their underlying class labels and plotted below:

[^MDS]: Multidimensional Scaling (MDS) is a technique used to visualize the similarity between data points in a lower-dimensional space. The property that makes it attractive
    for this task is that it tries to respect the distances between points in the original high-dimensional space. Technically, we use a variant called [Landmark MDS](https://cannoodt.dev/2019/11/lmds-landmark-multi-dimensional-scaling/), which
    is more scalable to large datasets.

![Gradient embeddings of unlabeled MNIST digits](/assets/img/mds_gradient_embedding_viz.png)

Notably the magnitudes of the gradient embeddings are not uniform across the classes, with examples from the `0` and `1` classes generally having greater magnitudes[^density]. This is expected, as the model
saw no examples of these classes during training, and is thus more uncertain about them. The plot visually demonstrates that the gradient embeddings effectively capture semantic information, as points from the same digit class tend to cluster together.

[^density]: It is a bit hard to tell from the plot, but >90% the 15k data points are concentrated on in a tight cluster around (0, 0).

We then use k-means++ to sample the next batch of data points to label. For a batch of size 10, the selected points are highlighted below:

![Gradient embeddings of the examples selected by BADGE to label](/assets/img/badge_selections_plotted.png)

As expected, BADGE predominantly selects examples from the `0` and `1` classes - the classes absent from the training data and where the model shows the highest uncertainty. Within these classes, BADGE strategically chooses points with diverse gradient embeddings, ensuring the selected examples capture different aspects of these unfamiliar classes (in contrast to uncertainty sampling, as we observed above). Intuitively, we hope this diversity will help the model develop a more comprehensive understanding of the previously unseen digits.

## Extending BADGE to Variable-Sized Batches
While the standard BADGE algorithm is effective, its requirement for a fixed batch size can be limiting in real-world scenarios. Consider situations where labeling resources fluctuate, or where the optimal number of data points to label before retraining varies.
In these cases, the inflexibility of a fixed batch size can lead to inefficient use of resources.
To address this, we introduce the concept of a priority ordering over the entire set of unlabeled data points.
Instead of selecting a fixed batch size upfront, this ordering tells us which points would be most valuable to label first,
second, third, and so on - allowing us to be flexible about how many points we ultimately select.

### When Are Variable-Sized Batches Helpful?

Variable-sized batches are particularly valuable in real-world machine learning workflows where labeling resources need to be managed efficiently. Let's explore a few practical scenarios:

#### 1. **Optimizing Labeler Productivity**
Consider a team of full-time data labelers who need to maintain consistent productivity. In settings where model training and evaluation are relatively
quick compared to labeling time, the number of examples needed before a model update can vary significantly—perhaps anywhere from 10 to 10,000 examples.
Variable-sized batches allow us to keep labelers continuously engaged by adapting to these fluctuating requirements.

#### 2. **Dynamic Budget Management**
Organizations frequently need to adapt to changing resource constraints throughout a project's lifecycle:

| Constraint Type     | Example Scenario                                                                                      |
| ------------------- | ----------------------------------------------------------------------------------------------------- |
| Budget Changes      | Labeling budgets may be adjusted mid-project                                                          |
| Service Limitations | Third-party labeling services often use time- or cost-based constraints rather than fixed batch sizes |
| Priority Shifts     | Project priorities may change, requiring resource reallocation                                        |

Variable-sized batches provide the flexibility to adapt to these changing constraints while maintaining the quality of selected examples.

#### 3. **Facilitating Online Learning**
In [online learning](https://en.wikipedia.org/wiki/Online_machine_learning) scenarios, data arrives sequentially, and models need to be updated incrementally.
Variable-sized batches are naturally suited for this setting. Instead of waiting for a fixed number of new labels to become available,
the model can be updated as soon as a sufficient number of high-priority examples have been labeled. This allows the model to adapt
more quickly to changes in the data distribution and leverage new information as it arrives, leading to potentially faster convergence
and improved performance over time. More on this below.


### Generating a Prioritized Ordering

As mentioned earlier, the key to supporting variable-sized batches in BADGE is to generate a priority ordering over the entire set of unlabeled data points.

K-means++ provides an elegant way to generate this priority ordering, as it naturally selects diverse and informative points
in a sequential manner before lumping them into a batch. By running k-means++ on the unlabeled pool and recording the order
of selection, we obtain a ranked list where earlier points are both more informative and well-distributed across the feature space.
We can then work through this priority queue of points, stopping whenever we've reached our desired batch size. This approach also lends itself well to online active learning scenarios, where new unlabeled data points may become available over time. The priority ordering allows us to continuously select the most valuable points for labeling as they appear.

### Demonstrating on MNIST (Continued from above)

To demonstrate this extension, we use the same setup as before, but this time we run k-means++ on the entire pool of unlabeled data points to generate an ordering. We then select the next data point to label by following this ordering.

We visualize the ordering generated by k-means++ below:

![BADGE selection proceeding in order](/assets/img/badge_selection_with_digits_transparent.gif)

### Optimizing Computational Efficiency of K-means++
The time complexity of k-means++ initialization is $O(nkd)$ where $n$ is the number of data points, $k$ is the number of centroids to select,
and $d$ is the dimensionality of the data. The $d$ factor comes from computing pairwise distances between points in $d$-dimensional space.
When we generate a complete ordering over the pool of unlabeled data points, we effectively run k-means++ $k$ times, where $k$ is the size of the pool.
This means that the overall time complexity of is $O(n^2d)$, which can be expensive for large datasets or high-dimensional data.

There are several ways to optimize this process. The first we will explore is the use a GPU,
combined with incrementally tracking minimum distances to avoid redundant computations:

```python
import torch

@torch.no_grad()
def kmeans_pp_pytorch(data: torch.Tensor, k: int = -1) -> torch.Tensor:
    """
    K-means++ implementation with distance caching.
    Leverages a GPU and only computes distances to the newest center each iteration.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    n_samples = data.shape[0]
    if k == -1:
        k = n_samples

    # Initialize output indices and distance cache
    selected_indices = torch.zeros(k, dtype=torch.long, device=device)
    min_squared_distances = torch.full((n_samples,), float('inf'), device=device)

    # Pick first point randomly
    selected_indices[0] = torch.randint(n_samples, (1,), device=device)

    # Process remaining points
    for i in range(1, k):
        # Get the last selected center and compute squared distances to all points
        last_center = data[selected_indices[i-1]].unsqueeze(0)
        new_squared_distances = torch.cdist(data, last_center, p=2).square()[:, 0]

        # Update minimum squared distances if new distances are smaller
        min_squared_distances = torch.minimum(min_squared_distances, new_squared_distances)

        # Sample next center with probability proportional to squared distance
        probs = min_squared_distances / min_squared_distances.sum()
        cumprobs = torch.cumsum(probs, dim=0)
        r = torch.rand(1, device=device)
        selected_indices[i] = torch.searchsorted(cumprobs, r)

        # Set distance to zero for selected point to avoid reselection
        min_squared_distances[selected_indices[i]] = 0

    return selected_indices
```

This GPU-accelerated implementation significantly improves performance. For a dataset of 15,000 embeddings with 2,560 dimensions,
it generates a complete ordering in just 20 seconds - compared to 15 minutes using scikit-learn's CPU implementation.

Additionally, since data labeling is time-consuming, we can optimize further by *extracting points from the ordering one at a time* (and only as-needed)
rather than computing the entire ordering upfront. This lazy evaluation approach allows us to stop once we've labeled our desired
number of points, avoiding unnecessary computation. It also allows us to add new (and potentially useful) data points to the pool
of unlabeled data on-the-fly as those points become available! See [this (longer) code snippet](https://gist.github.com/tyler-romero/1aba44c529c64fb3c29026ec906bcd9c) for an example of how to lazily generate the ordering.

## Can Other Popular Batch Active Learning Methods Be Similarly Extended?
Not all batch active learning methods can be extended to support variable-sized batches.
Let's consider a two categories of batch active learning methods and discuss their potential for extension:

### CORESET and other Greedy-Selection Algorithms ✅

[CORESET](https://arxiv.org/abs/1708.00489) is a batch active learning method that selects diverse data points by maximizing the minimum distance to already selected points. Like BADGE, it can be extended to generate a priority ordering over the unlabeled pool.

The extension is straightforward: instead of stopping after selecting k points, run the standard greedy
CORESET selection until all unlabeled points are ordered. Starting with an initial point (chosen randomly or as the furthest from existing labeled data),
each iteration selects the point with the maximum minimum distance to all previously selected points.

This naturally prioritizes points that cover unexplored regions of the feature space - earlier points represent more crucial additions to the diverse subset.
The ordering maintains CORESET's goal of representative sampling while enabling variable batch sizes.

### BatchBALD and Other Methods That Require a Global View ❌

Unlike BADGE and CORESET, [BatchBALD](https://arxiv.org/abs/1906.08158) doesn't lend itself well to variable-sized batches. BatchBALD works by selecting points that
maximize mutual information between selected and remaining points, using predictive entropy across the entire unlabeled pool.

The core issue is that BatchBALD's mutual information calculation is inherently global - it depends on the full set of unlabeled data. This makes it difficult to generate a
meaningful priority ordering, since selecting points one at a time would break the mutual information maximization that makes BatchBALD effective.

In short, greedy methods like BADGE naturally extend to variable batches, while global methods like BatchBALD would need major modifications to support this use case.

## Wrapping Up
Thanks for reading! Hopefully this post has provided you with a deeper understanding of the BADGE active learning method and how it can be extended to support variable-sized batches. I also hope you
found the visualizations and code snippets fun and helpful in understanding the concepts discussed. If you have any questions or feedback, feel free to [reach out](mailto:tyler.alexander.romero@gmail.com)!

## References

[1] Ash, J. T., Zhang, C., Krishnamurthy, A., Langford, J., & Agarwal, A. (2020). BADGE: Batch Active learning by Diverse Gradient Embeddings. *International Conference on Learning Representations*. [https://arxiv.org/abs/1910.11945](https://arxiv.org/abs/1910.11945)

[2] LeCun, Y., Cortes, C., & Burges, C. J. C. (1998). The MNIST Database of Handwritten Digits. [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

[3] De Silva, V., & Tenenbaum, J. B. (2004). Sparse Multidimensional Scaling using Landmark Points. *Technical Report, Stanford University*. [https://graphics.stanford.edu/courses/cs468-05-winter/Papers/Landmarks/Silva_landmarks5.pdf](https://graphics.stanford.edu/courses/cs468-05-winter/Papers/Landmarks/Silva_landmarks5.pdf)

[4] Sener, O., & Savarese, S. (2018). Active Learning for Convolutional Neural Networks: A Core-Set Approach. *International Conference on Learning Representations*. [https://arxiv.org/abs/1708.00489](https://arxiv.org/abs/1708.00489)

[5] Kirsch, A., van Amersfoort, J., & Gal, Y. (2019). BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning. *Advances in Neural Information Processing Systems*. [https://arxiv.org/abs/1906.08158](https://arxiv.org/abs/1906.08158)

[6] Motta, D. (2023). MDS and LMDS implementation. *GitHub Repository*. [https://github.com/danilomotta/LMDS](https://github.com/danilomotta/LMDS)

[7] De Moriarty, A. (2023). fast_pytorch_kmeans. *GitHub Repository*. [https://github.com/DeMoriarty/fast_pytorch_kmeans](https://github.com/DeMoriarty/fast_pytorch_kmeans)
