# TrueML Challenge: Debt Payment Propensity

End-to-end solution for the TrueML Data Science exercise: predict a customer's
probability of paying their next bill (`made_last_payment_on_time`) from a sample
of debt accounts.

The project covers exploratory analysis, statistical comparison across product
types, feature engineering and selection, model selection with cross-validation,
model interpretation with SHAP, experiment tracking and model registry with
MLflow, and a Streamlit application that scores the full portfolio using the
registered champion model.

---

## Contents

- [Exercise requirements](#exercise-requirements)
- [Repository structure](#repository-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Champion model](#champion-model)
- [Streamlit application](#streamlit-application)
- [Getting started](#getting-started)
- [Tests and CI](#tests-and-ci)
- [Notes and limitations](#notes-and-limitations)

---

## Exercise requirements

The original brief is included as [Data Science Exercise.pdf](Data%20Science%20Exercise.pdf).
It asks for three things, each addressed in the notebook:

| Requirement | Where it is addressed |
|---|---|
| Explore and summarize the data with descriptive statistics and plots, and explain what the patterns mean | [data_analysis_model_creation.ipynb](src/data_analysis_model_creation.ipynb), exploratory section |
| Write a function (or set of functions) to run comparisons across product type, generalizable to more than two products, with a unit test | `combination` and `ks_test_product` in the notebook; extracted to [combination_function.py](src/unit_test/combination_function.py) with tests in [test_for_functions.py](src/unit_test/test_for_functions.py) |
| Build a model to predict the probability of paying the next bill, showing the steps taken | Notebook modeling section, plus MLflow tracking and the Streamlit app |

---

## Repository structure

```
TrueML_Challenge/
├── README.md
├── requirements.txt
├── Data Science Exercise.pdf                Original brief
├── mlflow.db                                MLflow backend store (experiments + model registry)
├── .github/workflows/test.yml               CI: runs pytest on push and pull request
└── src/
    ├── app.py                               Streamlit scoring application
    ├── data_analysis_model_creation.ipynb   EDA, modeling and MLflow tracking
    ├── debt_sample.csv                      Raw dataset (671 accounts)
    ├── woe_encoder.pkl                      Fitted Weight-of-Evidence encoder
    ├── trueml_logo.webp                     Asset used by the app
    ├── mlflow.png                           Screenshot of the MLflow UI used to pick the champion
    ├── mlruns/                              Logged model artifacts referenced by the registry
    ├── databases/
    │   ├── database.ipynb                   Loads the CSV into SQLite
    │   ├── trueml_database.db               Input table  `database_challenge`
    │   └── trueml_predictions.db            Output table `predictions`
    └── unit_test/
        ├── combination_function.py          Recursive combination generator
        └── test_for_functions.py            pytest unit tests
```

---

## Dataset

`src/debt_sample.csv` contains 671 accounts and 16 columns, with no missing
values. The target is `made_last_payment_on_time` (373 negatives / 298
positives), so the data is only mildly imbalanced.

| Column | Type | Notes |
|---|---|---|
| `account_id` | int | Unique identifier, dropped for modeling |
| `made_last_payment_on_time` | int | Target, 0/1 |
| `latest_communication_channel` | categorical | Email (454) / SMS (217) |
| `minimum_payment` | float | |
| `previous_payment_amount` | float | |
| `product` | categorical | Credit Card (513) / Deferred Repayment (158) |
| `last_reminder_sent_days` | int | Days since the last reminder |
| `debt_to_income_ratio` | float | |
| `opened_last_communication` | int | Binary |
| `used_chat_feature` | int | Binary |
| `count_comms_sent_last_30d` | int | |
| `homeowner` | int | Binary |
| `ever_missed_payment` | int | Binary |
| `age_of_debt_yrs` | float | |
| `total_balance` | float | |
| `latest_communication_dow` | categorical | Day of week of the latest communication |

---

## Methodology

### 1. Exploratory analysis

- `df.info()` and `df.describe()` to establish types, ranges and the absence of
  missing values.
- Correlation heatmap over the continuous variables (binary columns excluded) to
  check for multicollinearity. No strong pairwise correlation was found, so a
  linear model remains a defensible choice.
- A reusable `diagnostic_plots(df, variable)` helper produces, for every
  continuous variable, a histogram (distribution), a Q-Q plot (linearity and
  normality) and a boxplot (outliers).
- Every continuous variable is right-skewed, and the skew persists when the data
  is split by target class. `total_balance` spans several orders of magnitude and
  is therefore inspected on a log scale.
- Cardinality of the categorical and binary columns is reviewed with
  `value_counts`.

### 2. Comparisons across product type

The brief asks for a function that generalizes beyond the two products present in
the sample. The approach is:

- `combination(lista)` is a recursive generator of the power set of a list. It is
  the piece under unit test, and it is what makes the comparison generic: with N
  products it produces every subset, and the caller keeps the subsets of size 2.
- `ks_test_product(df, variable)` builds all product pairs from
  `combination(df['product'].unique())`, filters to pairs, and runs a two-sample
  Kolmogorov-Smirnov test per pair, reporting whether the null hypothesis of
  equal distributions is rejected at alpha = 0.05.

Result: the distribution of the continuous variables differs by product for every
variable except `minimum_payment` and `last_reminder_sent_days`. That is a signal
that product-specific models could be worth exploring.

For the categorical inputs, a chi-square test of independence against the target
ranks the features by p-value. The strongest associations are
`opened_last_communication`, `ever_missed_payment`, `latest_communication_dow`
and `product`.

### 3. Feature engineering

- Train/test split first (80/20, `random_state=0`) so that all fitting happens on
  training data only.
- Categorical variables are encoded with Weight of Evidence
  (`category_encoders.woe.WOEEncoder`), fitted on the training split:

  ```
  WoE_i = ln( %distribution of goods_i / %distribution of bads_i )
  ```

  A `woe_table` helper prints the category-to-WoE mapping, which is what makes
  the SHAP plot readable later on.

### 4. Feature selection

Two complementary methods:

- **Permutation importance** on a `StandardScaler` + `LogisticRegression`
  pipeline, scored on precision. Top signals: `opened_last_communication`,
  `count_comms_sent_last_30d`, `last_reminder_sent_days` and, to a lesser degree,
  `ever_missed_payment`.
- **Recursive Feature Elimination**, sweeping the number of features from 2
  upward and scoring each candidate with 8-fold stratified cross-validation. Ten
  features gave the best precision at acceptable variance.

Selected features, in training order:

```
previous_payment_amount, product, last_reminder_sent_days, debt_to_income_ratio,
opened_last_communication, count_comms_sent_last_30d, ever_missed_payment,
age_of_debt_yrs, total_balance, latest_communication_dow
```

### 5. Metric choice

The guiding question is: which error hurts more, predicting that someone will pay
when they will not, or predicting that they will not pay when they will? The
first is costlier in a collections context, so **precision** is the primary
metric, with PR-AUC used for model selection during cross-validation.

### 6. Models evaluated

Given the small sample (671 rows), the search deliberately stays with simple,
low-variance models. Every candidate is a `StandardScaler` + estimator pipeline.

| Candidate | Outcome |
|---|---|
| Logistic Regression, default hyperparameters | Strong baseline: train precision 0.940, test precision 0.880, roughly a 6 percent train/test gap |
| Logistic Regression + `RandomizedSearchCV` (40 iterations, 8-fold stratified, scored on PR-AUC) | Comparable performance; this is the configuration registered as champion |
| k-NN, default | Worse than logistic regression at the same overfitting gap |
| k-NN + randomized search (`n_neighbors=31`, manhattan, distance-weighted) | Clear overfitting, discarded |
| Logistic Regression on IQR-winsorized features (`feature_engine.outliers.Winsorizer`, right tail, fold 1.5) | No improvement over the untreated features |

### 7. Decision threshold

The threshold is not left at 0.5. It is chosen on the training set from the
precision-recall curve, by maximizing F1:

```python
prec, rec, thr = precision_recall_curve(y_train, y_score)
f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
best_f1_thr = thr[f1s.argmax()]
```

The resulting threshold is stored as an MLflow tag on the registered model
version, so serving code reads it from the registry instead of hard-coding it.

### 8. Interpretation

`shap.LinearExplainer` over the scaled features produces a beeswarm plot. Read
together with the WoE mapping table, the main effects are:

- `count_comms_sent_last_30d`: more communications push the prediction positive.
- `last_reminder_sent_days`: the longer since the last reminder, the lower the
  probability of payment.
- `debt_to_income_ratio`: higher ratio, lower probability of payment.
- `ever_missed_payment`: a history of missed payments pushes strongly negative.
- `age_of_debt_yrs`: older debt, lower probability of payment.
- `product`: Credit Card (WoE +0.2178) leans positive, Deferred Repayment
  (WoE -0.6919) leans negative.
- `latest_communication_dow` and `total_balance` contribute very little.

### 9. Experiment tracking

Every candidate is logged to MLflow with hyperparameters, the feature list, the
decision threshold, train and test metrics (precision, recall, F1, ROC-AUC,
PR-AUC), a model signature and an input example. The champion is then registered
under the name `trueml_collections_logr`, given the alias `champion_log`, and
tagged with its `decision_threshold`, which is exactly what the Streamlit app
resolves at startup.

---

## Champion model

| Item | Value |
|---|---|
| Registered name | `trueml_collections_logr` |
| Alias | `champion_log` (version 1) |
| Run ID | `626d2cf870c84b2b951398dbb02e8953` |
| Pipeline | `StandardScaler` + `LogisticRegression` |
| Hyperparameters | `C=0.1274`, `penalty='l2'`, `solver='saga'`, `class_weight='balanced'`, `max_iter=700`, `random_state=0` |
| Features | 10 (listed above) |
| Decision threshold | 0.5254 |

| Metric | Train | Test |
|---|---|---|
| Precision | 0.944 | 0.880 |
| Recall | 0.944 | 0.880 |
| F1 | 0.944 | 0.880 |
| ROC-AUC | 0.982 | 0.966 |
| PR-AUC | 0.980 | 0.949 |

The train/test gap is around 6 percent, which is the main reason a simple linear
model was preferred over anything more expressive on 671 rows.

---

## Streamlit application

[src/app.py](src/app.py) is a scoring front end for the registered champion. It:

1. Reads the `database_challenge` table from `src/databases/trueml_database.db`.
2. Applies the fitted WoE encoder from `src/woe_encoder.pkl` and keeps the ten
   selected features in training order.
3. Resolves the champion through the MLflow registry
   (`trueml_collections_logr@champion_log`), reading the model version and its
   `decision_threshold` tag. Because the registry stores absolute Windows paths
   from the training machine, the loader re-resolves the artifact folder locally
   by searching for the matching `MLmodel` file under the repository root.
4. Scores the full portfolio, showing records scored, predicted positives and the
   mean probability, plus a sortable table with a probability progress column.
5. Offers a CSV download and a "Save to database" button that writes the
   `predictions` table to `src/databases/trueml_predictions.db`. The write is
   guarded, since it only succeeds when running locally.
6. Provides single-account scoring by `account_id`.

Run it with:

```bash
streamlit run src/app.py
```

The tracking URI defaults to `sqlite:///mlflow.db` at the repository root and can
be overridden with the `MLFLOW_TRACKING_URI` environment variable.

---

## Getting started

Requires Python 3.14. The CI workflow and the logged model artifacts both target
3.14, and the champion is serialized with scikit-learn 1.9.0, so a different
minor version of scikit-learn may fail to load it.

```bash
git clone https://github.com/331269/TrueML_Challenge.git
cd TrueML_Challenge

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

Extras used only inside the notebook and deliberately not pinned in
`requirements.txt`: `matplotlib`, `seaborn`, `scipy`, `statsmodels`, `shap`,
`skops`, `jupyter`.

### Rebuilding the SQLite input database

If `src/databases/trueml_database.db` needs to be regenerated from the CSV, run
[database.ipynb](src/databases/database.ipynb), which loads `debt_sample.csv`
into the `database_challenge` table.

### Browsing the experiments

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

`src/mlflow.png` is a snapshot of that UI at the moment the champion was chosen.

---

## Tests and CI

The unit tests cover the recursive `combination` function, which is the base of
the generic product comparison:

```bash
cd src
pytest
```

[.github/workflows/test.yml](.github/workflows/test.yml) runs the same command on
every push and pull request using Python 3.14.

---

## Notes and limitations

- **Sample size.** 671 rows is the binding constraint. It is why simple models
  win here, and why more data would help more than more feature engineering.
- **Product-specific models.** The KS tests show that most continuous variables
  are distributed differently by product. Fitting one model per product is the
  most promising next step that the data itself suggests.
- **Outlier treatment.** Only IQR winsorization on the right tail was tried.
  Quantile capping, random capping and other strategies remain unexplored.
- **Encoder scope.** `woe_encoder.pkl` is the artifact the app uses at serving
  time; the notebook fits its own encoder on the training split for evaluation.
- **Simulated data.** As the brief states, this is simulated data, so some
  relationships may be counterintuitive by construction.
- **Portability of artifact paths.** MLflow recorded absolute paths from the
  training machine. The app works around this at load time, but a real deployment
  would use a shared artifact store instead.
