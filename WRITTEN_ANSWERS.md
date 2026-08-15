# MovieIQ — Written Answers

## Stage 0 — Problem Statement

### 1. What makes a movie successful?
A movie is classified as **successful** when its revenue is greater than its budget:

`success = 1 if revenue > budget, else 0`.

### 2. Why is predicting film success valuable?
**Movie studios:** Predictions can support production-budget, marketing, release and resource-allocation decisions.

**Investors:** Predictions can help assess potential financial risk before investing in a movie.

### 3. Project objective
The objective is to analyse historical movie performance data and build a binary classification model that predicts whether a movie will be successful.

Main steps:
1. Clean and prepare the dataset and create the success target.
2. Perform exploratory and statistical analysis.
3. Train and evaluate a Random Forest classifier.
4. Deploy the analysis and predictor through Streamlit.

### 4. Why is this a classification problem?
The target has two categories:
- `1` = successful
- `0` = unsuccessful

Therefore, this is a **binary classification** problem.

## Stage 1 — Data Preparation

Original dataset: **2,000 rows × 7 columns**.

Exact duplicate rows found: **0**.

Rows with zero budget: **0**.

Rows with zero revenue: **0**.

After removing duplicate rows, invalid zero-budget/zero-revenue records and rows missing required numeric modelling fields, the analysis contains **2,000 rows**.

Success distribution:
- Successful: **1,614 (80.7%)**
- Unsuccessful: **386 (19.3%)**

The target is therefore imbalanced toward successful movies.

### Why handle zero budget/revenue?
The success label is directly based on revenue compared with budget. Zero or unavailable financial values can therefore create misleading target labels and are excluded before creating `success`.

### Genre processing
The `genres` field is stored as a stringified list of TMDB-style dictionaries, e.g. `"[{'id': 18, 'name': 'Drama'}]"`. It is parsed with `ast.literal_eval` (not a comma/pipe split, which would shred the dict syntax into garbage fragments) to extract each movie's genre name(s), then exploded for genre-level analysis. Movies with no genre entry are labeled `"Unknown"`.

## Stage 2 — Exploratory Data Analysis

### Budget vs Revenue
The analysis compares budget and revenue using a scatter plot and correlation matrix. Higher budget and revenue show a positive association in the dataset, but correlation does not imply causation.

### Genre
Genre frequencies and success rates are compared across the 9 genres present in the data (plus an "Unknown" bucket for the small number of movies with no genre listed). All 9 named genres have 190+ observations, so no group is too small to interpret. Frequencies are fairly even (~190–220 movies per genre, see `genre_frequency.png`), and success rates are tightly clustered between 78.5% and 82.2% (see `genre_success.png`) — no genre stands out as meaningfully more or less successful, consistent with the non-significant chi-square result below.

### Popularity, runtime and vote average
The distributions of these variables are compared between successful and unsuccessful movies using box plots and group summaries.

### Correlation heatmap
The heatmap shows relationships among the numeric variables. **Revenue is not used as a model feature** because it is used to construct the target and would cause target leakage.

## Stage 3 — Statistical Testing

### T-Test
Feature tested: **popularity**.

- Null hypothesis: mean popularity is the same for successful and unsuccessful movies.
- Alternative hypothesis: the means differ.
- t-statistic: **2.0617**
- p-value: **0.039682**

At α = 0.05, the result is **statistically significant**.

A p-value below 0.05 means the observed difference would be relatively unlikely if the null hypothesis were true. Statistical significance does not prove causation.

### Chi-Square Test
Variables: **genre and success**.

- Null hypothesis: genre and success are independent.
- Alternative hypothesis: genre and success are associated.
- Chi-square statistic: **1.773**
- Degrees of freedom: **9**
- p-value: **0.995**

At α = 0.05, we **fail to reject** the null hypothesis. Genre success rates are tightly clustered — from 78.5% (Science Fiction) to 82.2% (Drama) — so genre carries essentially no signal about success in this dataset.

## Stage 4 — Predictive Modeling

### Features
The Random Forest uses:
- budget
- popularity
- runtime
- vote_average

`revenue` is excluded because it is used to create `success`; using it would leak information from the target into the predictors.

### Train/test split
The data is split into 80% training and 20% testing using a fixed random seed and stratification.

### Random Forest
A Random Forest Classifier with 300 trees and balanced class weights is trained.

### Performance
- Accuracy: **80.50%**
- Precision: **80.70%**
- Recall: **99.69%**

Confusion matrix:

```text
[[  0  77]
 [  1 322]]
```

Because the target is imbalanced, accuracy should not be interpreted alone. Precision, recall and the confusion matrix provide additional information.

**Worth flagging directly:** a naive rule that always predicts "success" would also score ~80.75% accuracy on this test set — matching this model's 80.50%. The confusion matrix shows why: the model correctly identifies 0 of the 77 actual unsuccessful movies (top-left cell). Its apparently strong 99.7% recall is mechanically produced by almost never predicting "not successful," not by genuinely learning what separates flops from hits. This lines up with the weak/insignificant EDA and statistical-test results above — on the features available, there just isn't a strong signal for the model to find.

### Feature importance
The Random Forest feature-importance ranking is:

- **popularity**: 26.14%
- **budget**: 25.79%
- **vote_average**: 25.77%
- **runtime**: 22.29%

## Stage 5 — Streamlit

Run:

```bash
pip install -r requirements.txt
streamlit run MovieIQ.py
```

The dashboard provides genre and vote filters, EDA visualizations, statistical-test information and an interactive movie-success prediction form.

## Limitations
The dataset does not contain every factor that can influence movie success, such as marketing expenditure, cast, franchise strength, release timing, competition and distribution. The target is also imbalanced, so model performance should be interpreted using multiple metrics.

A future version could add richer pre-release features, cross-validation, hyperparameter tuning, probability calibration and comparison with other classification algorithms.
