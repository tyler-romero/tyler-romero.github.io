---
title: An Extension to BADGE Active Learning for Variable-Sized Batches
subtitle: BADGE can be trivially extended by providing an ordering over the pool of unlabeled data
date: 2025-01-02T00:00:00-08:00
blurb: BADGE, a useful active learning technique, can be extended to support variable-sized batches (where the next batch size is unknown a priori) while maintaining its benefits.
tags: ["post", "machine-learning", "active-learning", "BADGE"]
---

Batch Active learning by Diverse Gradient Embeddings (BADGE) is a simple and effective active learning method for neural networks that has been shown to outperform other active learning approaches in a variety of settings. Published in 2019 by Ash et al., BADGE is still commonly used today due to its simplicity and empirical effectiveness.

BADGE selects data points by examining their gradient embeddings - vectors that combine both the model's uncertainty (through gradient magnitude) and the data's semantic content (through gradient direction). By choosing points with diverse, high-magnitude gradients, BADGE identifies samples that are both informative (high uncertainty) and representative of different regions of the input space. This helps the model learn efficiently by focusing on challenging examples while maintaining good coverage of the dataset distribution.

One limitation of BADGE is that it requires the user to specify the batch size in advance. This can be a problem in practice, as the user may not know how many data points to label in each batch. In this post, I will show how BADGE can be extended to support variable-sized batches, where the next batch size is unknown a priori.

## BADGE Overview

Before we dive into the extension, let's briefly review how BADGE works. BADGE is a batch active learning method that selects a batch of data points that are diverse in their gradients. The algorithm proceeds as follows:

### 1. Compute Gradient Embeddings
First, we compute the gradients of the loss with respect to the penultimate layer of the neural network. These gradients capture both the model's uncertainty (magnitude) and the data's semantic content (direction). The first issue that arises is that - since our overall goal is to solicit labels for the most informative *unlabeled* data points - we don't have access to the true labels for these data points. In order to compute the gradients, we need to "hallucinate" labels for the unlabeled data points. This is typically done by using the current model to predict the expected label for each data point. The expected label is then used to compute the gradient.

BADGE uses the penultimate layer gradients, which can be computed efficiently using backpropagation, without needing to backpropagate through the entire network.

Here's a PyTorch implementation for computing the gradient embedding of a data point:

<!-- TODO: add correct code snippet -->
```python
def compute_gradient_embedding(model, loss_fn, data_point):
    model.eval()

    # Get the penultimate layer's parameters
    penultimate_params = list(model.parameters())[-2]  # last two layers are final classification layers
    penultimate_params.retain_grad()

    output = model(data_point)
    predicted_label = output.argmax(dim=1)  # "hallucinated" label
    loss = loss_fn(output, predicted_label)
    loss.backward()

    # Get gradient w.r.t penultimate layer
    gradient_embedding = penultimate_params.grad.clone().flatten()

    # Clear gradients
    model.zero_grad()

    return gradient_embedding
```

### 2. Select Diverse and Informative Points
Finally, BADGE selects the next batch of data points by choosing the points with the highest diversity in their gradient embeddings. This is done using kmeans++ initialization, a well-known greedy algorithm for seeding k-means clustering. The algorithm iteratively selects the next point to add to the batch based on its distance from all the points already selected.

Kmeans++ proceeds as follows:
1. Initialize the first point randomly.
2. For each subsequent point, iteratively compute the distance to the nearest point already selected and choose the next point with probability proportional to this distance squared: $d^2(x_i, S) = \min_{x_j \in S} ||\mathbf{g}(x_i) - \mathbf{g}(x_j)||^2$. Where $S$ is the set of already-selected points, and $\mathbf{g}(x_i)$ is the gradient embedding of point $x_i$.
3. Stop when the desired number of points have been selected.

If a batch size of `k` is specified, BADGE uses kmeans++ to select `k` points from the unlabeled pool.

Kmeans++ is chosen for BADGE because it selects points that are far apart from each other, ensuring that the selected points are both diverse and informative. By prioritizing points that are distant from each other in the gradient embedding space, kmeans++ also tends to select points with high-magnitude embeddings.

Scikit-learn provides [an implementation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.kmeans_plusplus.html) of the kmeans++ centroid initialization algorithm.

## Extending BADGE to Variable-Sized Batches
In some settings, such as when labeling data is expensive or time-consuming, it may be beneficial to select a variable number of data points to label in each batch. This allows the user to adapt the batch size based on factors such as the current model performance, the available budget, or the difficulty of the data points.

### When is this helpful?

As a practical example, consider a scenario where labelers are paid a fixed salary and we want to maximize their productivity. In this setting, training and evaluation are relatively inexpensive compared to the cost of labeling. For instance, a new model might be available after labeling anywhere from 10 to 1000 examples. To keep the labelers continuously engaged and productive, we need to adapt to the varying number of data points required before a new model is ready to generate another batch for labeling. This is where variable-sized batches can be beneficial.

Another such example would be when the labeling budget is changed mid-stream. Or if a third party labeling service is used with a pre-set budget. In these cases, the batch size can be adjusted based on the available budget.

<!-- TODO: discuss how this is advantagious to an online learning setting -->

### Generating a Prioritized Ordering

To extend BADGE to support variable-sized batches, we need to modify the selection process to allow for a dynamic batch size. The key insight is that we can maintain the benefits of BADGE by providing an ordering over the entire pool of unlabeled data points. This ordering can be used to select the next data point to label, regardless of the batch size.

Kmeans++ is a natural choice for this ordering, as it already selects points based on their diversity and informativeness. By running kmeans++ on the entire pool of unlabeled data points, and tracking the order in which points are selected, we can obtain an ordering that reflects the diversity and informativeness of the data points. We can then select the next data point to label by following this ordering.

<!-- TODO: diagram -->
<!-- TODO: time complexity analysis -->
<!-- TODO: actual time benchmarking -->

##

## Extending Other Batch Active Learning Methods
Not all batch active learning methods can be easily extended to support variable-sized batches...