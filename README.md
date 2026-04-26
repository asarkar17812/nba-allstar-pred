# NBA All-Star Prediction

## Overview

This project models NBA All-Star selection as a **structured ranking problem**, rather than a standard binary classification task.

Each season is partitioned by conference, and selections are made under fixed constraints:
- 2 backcourt starters  
- 3 frontcourt starters  
- 7 reserves  

The objective is not simply to predict probabilities, but to **rank players within competitive groups and construct valid rosters**.

Two modeling approaches are implemented:
- A structured neural network with attention and multi-objective loss  
- A hybrid logistic regression model combining pointwise and pairwise learning  

---

## Data Pipeline

The dataset is constructed from multiple historical sources and normalized to ensure consistency across eras.

### Key preprocessing decisions

- Team identities are canonicalized to handle relocations and naming changes  
- Conference labels are deterministically assigned from team mappings  
- Position is reduced to **Backcourt vs Frontcourt** to reflect roster constraints  
- Low-signal players (few games / low minutes) are filtered unless selected  
- Missing values are imputed **within each season** using kNN (k = 5)  
- Numerical features are standardized within-season to remove era effects  

### Example normalization

Instead of comparing raw scoring across eras:
  x’ = (x − μ_season) / σ_season
This preserves **relative performance within a season**, which is what voting implicitly reflects.

---

## Problem Formulation

Let each group be defined by:
  G = (season, conference)
  
For players \( i ∈ G \), the model produces scores:
  s_i = starter score
  r_i = reserve score

Final ranking score:
  score_i = 0.3 * s_i + 0.7 * r_i

Selection is then performed under constraints:
  Top 2 backcourt (starters)
  Top 3 frontcourt (starters)
  Top 7 remaining (reserves)

---

## Neural Network Model

### Architecture

- Feature MLP → representation
- Embeddings:
  - Conference
  - Position group
  - Season
- Multi-head self-attention across players within a group
- Two output heads:
  - Starter head
  - Reserve head

This allows the model to capture **relative interactions between players**, not just independent scores.

---

### Training Objectives

The model is trained using a combination of structured losses.

#### 1. Soft Selection Loss

Approximates top-k selection using a temperature-scaled softmax:
  z_i = k * softmax(s_i / τ)

Target distribution:
  target_i = k * y_i / sum(y)

Loss:
  L_select = mean((z − target)^2)

---

#### 2. Pairwise Ranking Loss

Encourages correct ordering between selected and non-selected players:
  L_pair = mean(max(0, m − (s_pos − s_neg)))

---

#### 3. Binary Classification Loss

Standard calibration term:
  L_bce = BCE(s_i + r_i, y_i)

---

#### 4. Constraint Regularization

Encodes roster structure:

- Backcourt proportion constraint
- Starter/reserve overlap penalty

---

### Final Loss
  L = L_select + λ_pair * L_pair + λ_bce * L_bce + regularization

---

## Logistic Regression Model

### Design

A hybrid model combining:

1. **Pointwise model**
   P(y = 1 | x)
2. **Pairwise model**
   P(i > j) = σ(w^T (x_i − x_j))

---

### Pairwise Dataset Construction

For each group:
  (x_i − x_j, 1) : if i is All-Star, j is not (x_j − x_i, 0)

This transforms ranking into a binary classification problem.

---

### Final Prediction
  p = α * p_point + (1 − α) * p_pair

Temperature scaling:
  logit = log(p / (1 − p)) / T

---

## Evaluation Methodology

Evaluation mirrors the actual selection process.

### Step 1: Score all players  
### Step 2: Apply structured selection  
### Step 3: Compute metrics  

---

### Metrics

#### Global
- AUC (ranking quality)
- Accuracy
- Precision
- Recall
- F1

#### Structured Metrics
- **Top-12 Recall**: fraction of true All-Stars selected  
- **Top-12 Accuracy**: fraction of selected players who are correct  

#### Grouped Evaluation
- Per (season, conference)
- Per season overall
- Average across seasons

---

## Results

### Neural Network

#### Training Summary

| Metric        | Value |
|---------------|-------|
| Initial Loss  | 18.02 |
| Final Loss    | 4.63  |
| Best Val AUC  | 0.9888 |
| Epochs        | 80    |

#### Final Performance

| Metric | Value |
|--------|------|
| AUC | 0.9827 |
| Accuracy | 0.9740 |
| Precision | 0.8333 |
| Recall | 0.7547 |
| F1 | 0.7921 |

---

### Logistic Regression

#### Model Selection

| Metric | Value |
|--------|------|
| Best C | 2.1544 |
| Val AUC Range | 0.9850 – 0.9862 |
| Val Top-K Range | 0.7468 – 0.7595 |

#### Final Performance

| Metric | Value |
|--------|------|
| AUC | 0.9872 |
| Accuracy | 0.9678 |
| Precision | 0.7813 |
| Recall | 0.7075 |
| F1 | 0.7426 |

---

## Comparison (To be updated)

| Metric | Neural Network | Logistic Regression |
|--------|--------------|---------------------|
| AUC | 0.9827 | **0.9872** |
| Accuracy | **0.9740** | 0.9678 |
| Precision | **0.8333** | 0.7813 |
| Recall | **0.7547** | 0.7075 |
| F1 | **0.7921** | 0.7426 |
| Top-12 Recall | **0.7553** | 0.7083 |
| Top-12 Accuracy | **0.8333** | 0.7813 |

---

## Interpretation

- Logistic regression achieves slightly higher AUC, indicating strong ranking calibration  
- The neural network consistently outperforms in **selection-based metrics**  
- This suggests that:
  - Linear models capture global signal well  
  - Structured models better capture **competition and roster constraints**

---

## Key Takeaways

- All-Star selection is fundamentally a **structured ranking problem**
- Group-relative features are critical for performance
- Pairwise learning significantly improves ranking quality
- Explicit constraint modeling leads to better real-world outcomes

---

## Future Work

- Incorporate voting data (fan/media/player splits)  
- Model temporal dynamics explicitly  
- Explore graph-based player interaction models  
- Improve calibration between starter and reserve heads  

---

## Repository Structure (To be updated)
  source/
  cleaned/
    cleaned_data.csv
  uncleaned/
    NBA ALL STAR DATA.xlsx

notebooks/
  neural_net.ipynb
  log_reg.ipynb
  svm.ipynb

---

## Notes

This project prioritizes:
- Interpretability of modeling decisions  
- Alignment with real-world selection constraints  
- Robustness across eras and seasons  

The goal is not just prediction accuracy, but **faithfully modeling how selections are actually made**.
