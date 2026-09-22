# Pima Indians Diabetes — Binary Classification Study

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/status-complete-success)

A comparative study of three binary classifiers: **Logistic Regression**, **Random Forest** and **C-Support Vector Classification (SVC)**. These are trained to predict diabetes from eight routine clinical measurements in the Pima Indians Diabetes dataset.

Each model is tuned with `GridSearchCV`, trained twice (on raw and on standardized features), and compared on F-score, ROC AUC, precision, recall and specificity, with 5-fold stratified cross-validation used as an overfitting check.

> Academic project - Machine Learning (*Aprendizagem Automática*), BSc in Computer Science and Multimedia Engineering (LEIM), ISEL.

---

## Dataset

The Pima Indians Diabetes dataset was compiled by the US National Institute of Diabetes and Digestive and Kidney Diseases, which has monitored the Pima population since 1965 due to its high diabetes incidence. It contains **768 female patients**, of whom **268 are diabetic** and **500 are not**.

| Feature | Range |
| :-- | :-: |
| Number of pregnancies | 0 – 17 |
| Plasma glucose concentration (2 h oral glucose tolerance test) | 0 – 199 |
| Diastolic blood pressure (mm Hg) | 0 – 122 |
| Triceps skin fold thickness (mm) | 0 – 99 |
| 2-hour serum insulin (µU/ml) | 0 – 846 |
| Body mass index (kg/m²) | 0 – 67.1 |
| Diabetes pedigree function | 0.078 – 2.42 |
| Age (years) | 21 – 81 |

The data ships with the repository as a Python pickle (`src/pimaDiabetes.p`) holding `data`, `target` and `feature_names`.

## Approach

1. **Exploratory analysis:** A correlation matrix across all eight features. The strongest pair (pregnancies / age, $ρ = 0.54$) was not strong enough to justify dropping a feature, so all eight were kept.
2. **Stratified split:** $80\%$ train $/$ $10\%$ validation $/$ $10\%$ test, stratified on the target to preserve the 65/35 class balance.
3. **Hyperparameter search:** `GridSearchCV` over a per-model parameter grid (regularization strength and iteration budget for `Logistic Regression`; tree count, depth and split/leaf minimums for `Random Forest`; C, kernel and gamma for `SVC`).
4. **Two feature regimes:** Every model is fitted twice: once on raw features and once on features standardized to zero mean and unit variance (`StandardScaler`), to measure how much each algorithm depends on feature scaling.
5. **Evaluation:** Confusion matrix, recall, specificity, precision, F-score and ROC AUC on the held-out test set, plus 5-fold stratified cross-validation over the full dataset as an overfitting check.

## Results

Test-set performance ($\%$), raw features vs. standardized features:

| Model | Features | F-score | ROC AUC | Precision | Recall | Specificity | Errors |
| :-- | :-- | :-: | :-: | :-: | :-: | :-: | :-: |
| Logistic Regression | raw | $82.87$ | $78.92$ | $75.4$ | $92.0$ | $42.3$ | $19$ |
| Logistic Regression | standardized | $80.35$ | $80.15$ | $75.4$ | $86.0$ | $46.2$ | $21$ |
| Random Forest | raw | $81.09$ | $79.14$ | $73.8$ | $90.0$ | $38.5$ | $21$ |
| **Random Forest** | **standardized** | **$83.80$** | **$81.46$** | **$80.0$** | $88.0$ | **$57.7$** | **$17$** |
| SVC | raw | $79.62$ | $73.38$ | $71.4$ | $90.0$ | $30.8$ | $23$ |
| SVC | standardized | $80.35$ | $79.14$ | $75.0$ | $84.0$ | $46.2$ | $22$ |

5-fold cross-validated accuracy ($\%$):

| Features | Logistic Regression | Random Forest | SVC |
| :-- | :-: | :-: | :-: |
| raw | $77.2$ | $77.3$ | $76.0$ |
| standardized | $83.0$ | $83.0$ | $77.0$ |

#### **Conclusion** 
The **Random Forest on standardized features** is the best model overall: it leads on F-score, ROC AUC, precision and specificity, and makes the fewest test-set errors. Logistic Regression is a close and much cheaper alternative, and its strong showing on raw features suggests the classes are close to linearly separable. SVC trails on every metric. Cross-validation accuracies sit within a few points of the test-set results, so none of the models shows obvious overfitting.

All models share the same defect: specificity is far below recall, *i.e.* diabetic cases are frequently missed. Standardization is especialy helpful in this matter, raising Random Forest specificity from $38.5\%$ to $57.7\%$.

## Repository Structure

```
.
├── doc/
│   └── pimaDiabetesEnunciado_25-26.pdf   # assignment brief (PT)
├── src/
│   ├── A48630A51038A51811TP1.ipynb       # main notebook: analysis, training, evaluation
│   ├── drafts.ipynb                      # exploratory scratch notebook
│   ├── pimaDiabetes.p                    # dataset (pickle)
│   └── tools.py                          # helper library: metrics, ROC, grid search, k-fold
├── requirements.txt
└── README.md
```

`tools.py` is the reusable part of the project and contains:

| Function | Purpose |
| :-- | :-- |
| `show_corr_matrix` | Prints the feature correlation matrix |
| `get_best_params` | Wraps `GridSearchCV` and returns the best parameter set |
| `normalize_data` | Standardizes features to zero mean, unit variance |
| `predition_stats` | Confusion matrix plus recall, specificity, precision, F-score and G-score |
| `ROC_curve` | Plots the ROC curve and prints the AUC |
| `kfold_cross_validation` | Stratified k-fold accuracy and AUC, with standard deviations |

## Known Limitations

- `StandardScaler` is fitted independently on the train, validation and test splits instead of being fitted on the training set and applied to the others. The correct procedure is a single scaler fitted on train only.
- The SVC ROC curves are plotted with the Random Forest probability estimates, so the two SVC ROC AUC values above are not trustworthy.
- The validation split is created but never used; model selection relies on `GridSearchCV`'s internal cross-validation and results are reported on the test split.

## Authors

Group project for Machine Learning (T52D), ISEL — DEI, 2025/26. Supervised by Prof. Gonçalo Xufre Silva.

- João Madeira (48630)
- Renata Góis (51038)
- Bruno Pereira (51811)
