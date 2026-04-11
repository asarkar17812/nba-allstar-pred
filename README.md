# NBA Player All-Star Predictor:

# Motivation and Inspiration:

<!-- 
Briefly describe:
- Why NBA data?
- What question are you trying to answer?
- Why ML is appropriate here
-->

---

# Methodology:

<!-- 
Outline high-level pipeline:
1. Data collection
2. Cleaning + preprocessing
3. Feature engineering
4. Model training
5. Evaluation
-->

We construct a unified dataset by combining multiple sources:

$$
D = \bigcup_{i=1}^{n} T(D_i)
$$

where $T(\cdot)$ denotes preprocessing and alignment transformations.

---

# Data Cleaning and Processing:

<!-- 
To be completed after implementation
Consider including:
- Missing value handling
- Normalization / scaling
- Feature construction
- Train/test splits
-->

---

# Models:

We plan to implement and compare the following models:

- **Neural Networks**

  <!-- Architecture, loss function, optimizer to be added -->
- **Logistic Regression (with Boosting)**

  <!-- Specify boosting method (e.g., AdaBoost, Gradient Boosting) -->
- **Support Vector Machines (SVM)**

  <!-- Specify kernel (e.g., linear, RBF), regularization parameter C, and gamma -->  
x`
<!-- 
Optional additions:
- Hyperparameter tuning
- Cross-validation strategy
-->

---

# Results and Comparisons

<!-- 
To be completed after experiments

Include:
- Evaluation metrics (accuracy, MSE, etc.)
- Model comparisons
- Visualizations (confusion matrix, ROC curves, etc.)
-->

---

# Conclusions and Final Remarks:

<!-- 
Summarize:
- Key findings
- Model performance
- Limitations
- Future work
-->

---

# Data Sources

We aggregate multiple datasets to build a unified dataset of NBA team and player statistics.

## Team Data

- **Team Records Dataset**Source: :contentReference[oaicite:0]{index=0}File: `Team_Records`
- **Team Abbreviation Mapping**
  Source: :contentReference[oaicite:1]{index=1}
  File: `TeamHistories.csv`

## Player Data

- **Modern Player Data (1996–2023)**Source: :contentReference[oaicite:2]{index=2}File: `all_seasons.csv`
- **Historical Player Data (1950–2017)**
  Source: :contentReference[oaicite:3]{index=3}
  File: `player_data.csv`

## Supplementary Sources

- :contentReference[oaicite:4]{index=4}
- :contentReference[oaicite:5]{index=5}

These sources were used to:

- Fill missing seasons
- Validate inconsistencies
- Cross-check player and team statistics

---

## Data Coverage

We combine datasets across overlapping time periods:

$$
\text{Total Coverage} = [1950, 2023]
$$

with:

- Historical dataset: \(1950 \leq t \leq 2017\)
- Modern dataset: \(1996 \leq t \leq 2023\)

---

## Data Cleaning Notes

- Resolved team naming inconsistencies using `TeamHistories.csv`
- Imputed missing values using external references
- Standardized feature formats across datasets
