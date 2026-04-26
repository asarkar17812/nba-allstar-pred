# NBA All-Star Prediction

This project models NBA All-Star selection as a **structured, competitive ranking problem** rather than independent classification.

Selections are:
- **relative within (season, conference)**
- **constrained by roster rules**
- **dependent across players**

The pipeline is built to preserve that structure from preprocessing through evaluation.

---

## Problem Framing

Instead of predicting:
> “Is this player an All-Star?”

we model:
> “How does this player rank within their competitive pool?”

Formally, for a group $G_{s,c}$ (season $s$, conference $c$), we learn a scoring function:
$$
f(x_i) \rightarrow \mathbb{R}
$$

and selection is:
$$
\text{TopK}(f(x_i)) \quad \text{subject to roster constraints}
$$

---

## Data Pipeline

The preprocessing pipeline is designed around a single constraint:

> **Preserve relative performance within each season while preventing leakage across seasons.**

---

## Preprocessing Design

### Canonical Team Mapping

Franchise identity is normalized across time:
- resolves relocations
- ensures grouping consistency

Grouping key:
$$
G = (\text{Season}, \text{Conference})
$$

---

### Structural Missingness

Certain NaNs are deterministic:
$$
\text{3P\%} =
\begin{cases}
0 & \text{if } \text{3PA} = 0 \\
\text{observed} & \text{otherwise}
\end{cases}
$$

Handled before imputation.

---

### Season-wise Normalization

Raw stats are not comparable across eras. We normalize within season:

$$
x' = \frac{x - \mu_s}{\sigma_s}
$$

where:
- $\mu_s$ = season mean  
- $\sigma_s$ = season standard deviation  

This preserves **relative standing**.

---

### Season-wise Imputation

For each season $s$:

1. Standardize features:
$$
X_s \rightarrow \tilde{X}_s
$$

2. Apply KNN imputation:
$$
\tilde{X}_s^{\text{imputed}} = \text{KNN}(\tilde{X}_s)
$$

3. Inverse transform:
$$
X_s^{\text{final}} = \sigma_s \tilde{X}_s^{\text{imputed}} + \mu_s
$$

This prevents leakage across seasons.

---

### Filtering

Players are kept if:
$$
(\text{Games} \ge 20 \land \text{Minutes} \ge 10) \lor (\text{AllStar} = 1)
$$

---

## Models

### Hybrid Logistic Regression

We combine pointwise and pairwise predictions:

$$
p_i = \alpha \cdot p_i^{\text{point}} + (1 - \alpha) \cdot p_i^{\text{pair}}
$$

Pairwise model learns:
$$
P(i > j) = \sigma(w^\top (x_i - x_j))
$$

Temperature scaling:
$$
p_i' = \sigma\left(\frac{\log\left(\frac{p_i}{1-p_i}\right)}{T}\right)
$$

---

### Neural Network

Two-head architecture:

- starter score: $s_i$
- reserve score: $r_i$

Final score:
$$
\hat{y}_i = \sigma(0.3 \cdot s_i + 0.7 \cdot r_i)
$$

---

### Soft Selection (Training Objective)

Instead of hard top-$k$, we use:

$$
z_i = k \cdot \frac{e^{s_i / \tau}}{\sum_j e^{s_j / \tau}}
$$

Target distribution:
$$
t_i = \frac{k \cdot y_i}{\sum_j y_j}
$$

Selection loss:
$$
\mathcal{L}_{\text{select}} = \frac{1}{n} \sum_i (z_i - t_i)^2
$$

---

### Pairwise Loss

$$
\mathcal{L}_{\text{pair}} =
\mathbb{E}_{i \in \text{pos}, j \in \text{neg}}
\left[ \max(0, m - (s_i - s_j)) \right]
$$

---

### Full Objective

$$
\mathcal{L} =
\mathcal{L}_{\text{select}} +
\lambda_1 \mathcal{L}_{\text{pair}} +
\lambda_2 \mathcal{L}_{\text{BCE}}
$$

---

## Structured Selection

For each group $G_{s,c}$:

$$
\begin{aligned}
\text{Starters}_{BC} &= \text{Top-2}_{BC} \\
\text{Starters}_{FC} &= \text{Top-3}_{FC} \\
\text{Reserves} &= \text{Top-7 remaining}
\end{aligned}
$$

---

## Evaluation

### Global Metrics

- AUC:
$$
\text{AUC} = P(f(x^+) > f(x^-))
$$

- Precision:
$$
\frac{TP}{TP + FP}
$$

- Recall:
$$
\frac{TP}{TP + FN}
$$

---

### Structured Metrics

Top-K Recall:
$$
\frac{\text{# correctly selected All-Stars}}{\text{# true All-Stars}}
$$

Top-12 Accuracy:
$$
\frac{\text{# correct selections}}{12}
$$

---

## Results

### Neural Network

| Metric    | Value  |
|----------|--------|
| AUC      | 0.9827 |
| Accuracy | 0.9740 |
| Precision| 0.8333 |
| Recall   | 0.7547 |
| F1 Score | 0.7921 |

---

### Logistic Regression

| Metric    | Value  |
|----------|--------|
| AUC      | 0.9872 |
| Accuracy | 0.9678 |
| Precision| 0.7813 |
| Recall   | 0.7075 |
| F1 Score | 0.7426 |

---

## Interpretation

- Logistic regression optimizes:
$$
\max P(f(x^+) > f(x^-))
$$

- Neural network optimizes:
$$
\text{selection consistency under constraints}
$$

This explains:
- higher AUC for logistic regression  
- better final rosters for neural network  

---

## Key Takeaways

- The task is fundamentally:
$$
\text{constrained ranking}, \not\; \text{classification}
$$

- Correct preprocessing is critical:
$$
\text{preserve structure} \gg \text{maximize raw signal}
$$

- Selection-aware training improves real-world performance

---
