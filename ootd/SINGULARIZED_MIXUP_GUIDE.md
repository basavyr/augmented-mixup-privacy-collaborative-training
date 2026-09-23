# Singularized Mixup: Questions and Answers

This guide explains the singularized-mixup design, its intended collaborative-learning workflow, and how it relates to the current implementation. I use **you** for a data-holding party and **we** for the parties collectively.

## 1. What does one singularized mixup sample look like?

Starting from two local examples $(x_i,y_i)$ and $(x_{\pi(i)},y_{\pi(i)})$, you release

$$
\tilde{x}_i = w_{i1}x_i+w_{i2}\bigl(x_{\pi(i)}+e_i\bigr),
\qquad
\tilde{y}_i = w_{i1}y_i+w_{i2}y_{\pi(i)}.
$$

Here:

- $\pi(i)$ is the partner index;
- $w_{i1},w_{i2}\geq 0$ are mixing weights with $w_{i1}+w_{i2}=1$;
- $e_i$ is random noise with a calibrated $\ell_2$ norm;
- the label is mixed, but no noise is added to the label.

In feature-space training, replace image $x$ with a feature vector $f(x)$; the construction is otherwise the same.

## 2. Why is noise added only to the “second” sample?

“Second” is a role, not an intrinsically special data point. You could rename the two operands and exchange their weights. What matters is the asymmetric design: one component is clean and one component is noise-bearing.

This gives a released representation with one useful signal component and one perturbed private component. It limits distortion for training while making inversion harder. If an attacker knew the partner and weights, a direct inversion for $x_i$ would retain unknown noise:

$$
\frac{\tilde{x}_i-w_{i2}x_{\pi(i)}}{w_{i1}}
= x_i+\frac{w_{i2}}{w_{i1}}e_i.
$$

Adding noise to both inputs is a different mechanism. It may be viable, but the paper’s utility and reconstruction analysis are derived for one clean and one noisy component, not for two noisy components.

## 3. Is the noise random or crafted from the data?

Its **direction is random**; its **magnitude is calibrated from the data**.

We first estimate a typical distance between random samples, $r_{\mathrm{avg}}$, along with global variance $\hat v$ and dimensionality $d$. The code computes

$$
c=\frac{r_{\mathrm{avg}}^2}{2d\hat v},
$$

then, for mixing coefficient $\alpha$ and privacy parameter $\tau$, computes

$$
m_f(\tau)=
\sqrt{
\frac{
\max\!\left(
\frac{\alpha^2}{\tau(1-\alpha)^2}-1,\;0
\right)
}{2c}
}.
$$

The target noise norm is

$$
\lVert e\rVert_2=m_f(\tau)r_{\mathrm{avg}}.
$$

For each protected sample, we draw a new random Gaussian direction and rescale it to this target norm. The direction is not selected to hide a particular feature of the current image pair. Smaller $\tau$ produces larger noise and therefore stronger signal attenuation.

## 4. How should I choose partners for a single protected training dataset?

Choose the partner mapping once, then keep it fixed for that protected dataset. A natural construction is a permutation $\pi$ of the local indices: every local sample appears once as a primary sample and once as a partner.

The paper’s Algorithm 1 samples a uniform random permutation. That allows:

- **fixed points**: $\pi(i)=i$, a self-pair;
- **two-cycles**: $i\to j$ and $j\to i$;
- longer cycles.

A self-pair gives $\tilde{x}_i=x_i+w_{i2}e_i$. If you want every mix to contain two distinct private examples, use a derangement, which forbids fixed points. If you also want to avoid using the same unordered pair in both directions, forbid two-cycles. Those are reasonable implementation choices, but they are stricter than the paper’s stated algorithm and should be evaluated as such.

For the collaborative setting, create partners **within each party’s local dataset**. Do not require parties to share raw examples merely to form cross-party pairs.

## 5. Should partner assignments stay the same in every epoch?

It depends on whether you are following the **one-shot privacy protocol** or doing ordinary local augmentation.

For the intended one-shot collaborative protocol:

1. You create the partner mapping, weights, noise, and protected representations once.
2. You send that protected dataset to the central server once.
3. The server can train for many epochs by reusing that same protected dataset.

Do not send new protected versions of the same private records every epoch. Fresh releases create additional noisy linear observations of the same private data. An attacker may combine those observations; even repeated releases with the same partners and independently sampled noise can reduce effective noise by averaging.

If mixup is used only inside a trusted local training process and no intermediate protected representations leave that process, regenerating pairs/noise every epoch is ordinary data augmentation. It is not the same privacy setting as releasing a protected dataset to another party.

## 6. Should I use the same partners when comparing different $\tau$ values?

Yes, for a fair tau ablation. Keep the following fixed across tau values:

- train/validation/test split;
- client partition;
- partner mapping;
- mixing weights;
- random noise directions, if practical;
- training initialization and optimization schedule.

Then change only the noise magnitude prescribed by $\tau$. This is a paired comparison: an accuracy or reconstruction difference can be attributed more confidently to tau rather than to different partners or random training variation.

Afterward, repeat the entire sweep with multiple independent seeds and report variation across runs. In a new independent run, you may choose a different partner mapping.

## 7. What happens if I generate another protected dataset with different random partners?

For an offline experiment, this is fine: it is another random trial.

For a privacy-sensitive release of the same private training set, it is risky. The central recipient receives another set of equations involving the same unknown records but different combinations. That can improve reconstruction opportunities. The same concern applies to a new release with the same partners but fresh noise: independent noise can be combined or averaged down.

The safest interpretation of the protocol is one protected release per private training dataset and privacy budget/context.

## 8. Do we train and test on mixed images?

We train on mixed representations, but we evaluate the final classifier on ordinary, unmixed test images.

Training uses a synthetic example and its soft label:

$$
\tilde f=w_1f(x_i)+w_2\bigl(f(x_j)+e\bigr),
\qquad
\tilde y=w_1y_i+w_2y_j.
$$

At inference, the real task is still “classify one normal image.” We therefore compute $f(x)$ for a plain test image $x$ and feed it to the trained classifier. We compare its hard predicted class to the normal test label.

Testing on mixed examples would measure a different task: classification of artificial mixtures rather than normal-image classification.

## 9. How does this replace federated learning?

Each party can transform its local data once, send the protected mixed representations and soft labels to a central server, and let that server train one classifier on the pooled protected datasets.

This avoids the usual federated-learning loop of repeated local model updates, gradient/model transmission, and server aggregation. Raw training images remain local.

This is not cryptographic security or differential privacy. The server still receives transformed information derived from private training data. The paper’s claim is a reduced reconstruction risk under its specified threat model.

## 10. What does the current codebase do?

`collaborative_training.py` currently performs feature-space mixup on each in-memory training batch:

- a fixed ResNet extracts features from plaintext local images;
- partners are made by a cyclic shift within the current batch;
- weights and noise are freshly sampled during training;
- the classifier trains on mixed features and soft labels;
- `evaluate_classifier` evaluates plain test images with hard labels and no test-time mixup/noise.

Therefore, the reported accuracy is standard clean-test accuracy. The current training loop behaves like online augmentation. It does not yet materialize one protected, fixed dataset and simulate transmitting it exactly once to a separate central server. That distinction matters when evaluating the paper’s one-shot collaborative privacy protocol.
