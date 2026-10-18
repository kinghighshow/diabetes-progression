# Diabetes Progression Screening

![Python](https://img.shields.io/badge/Python-3.10%2B-2a78d6?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-eb6834?logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-interactive-1baf7a?logo=plotly&logoColor=white)

Predicting one-year diabetes progression from baseline measurements and flagging high-risk patients, using the [scikit-learn diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) (442 patients, 10 features).

**Sultan Akinola Atanda** · CSCN8010 Foundations of Machine Learning · Conestoga College

## Results

| Task | Model | Test performance |
|---|---|---|
| Score from BMI only | Polynomial, degree 1 | R² 0.28 · MAE 51.5 · MAPE 43.2% |
| Score from 8 features | kNN, k = 11 | R² 0.49 · MAE 42.8 · MAPE 36.3% |
| Flag high risk (top 25%) | Logistic regression, class-balanced | Recall 0.74 · ROC-AUC 0.88 |

<p align="center">
  <img src="images/chosen_model_fit.png" width="100%" alt="BMI model fit over train, validation and test data">
</p>

<table>
  <tr>
    <td width="50%"><img src="images/correlation_matrix.png" alt="Correlation matrix"></td>
    <td width="50%"><img src="images/logistic_confusion_matrix.png" alt="Logistic regression confusion matrix"></td>
  </tr>
</table>

## Key findings

- **More features help, more flexibility does not.** Eight features nearly double R² over BMI alone, while the degree-3 polynomial and deeper trees overfit.
- **The small validation set can mislead.** It favoured a degree-5 polynomial that cross-validation rejected.
- **For screening, classify directly.** The best regressor catches 6 of 19 high-risk test patients; the balanced logistic model catches 14.

## Project structure

```
├── diabetes_progression.ipynb   # analysis and report
├── dashboard.html               # standalone screening dashboard, open in any browser
├── src/
│   ├── data_pipeline.py         # loading, summaries, splits
│   ├── visuals.py               # Plotly charts and image export
│   ├── modeling.py              # model building, evaluation, inspection
│   ├── dashboard.py             # exports the models to a standalone dashboard
│   └── dashboard_template.html  # dashboard page layout
└── requirements.txt
```

## Run it

```bash
pip install -r requirements.txt
jupyter lab diabetes_progression.ipynb
```

Run all cells. Figures are saved to `images/`, and the last cell writes `dashboard.html`.

## Try the model

Open **`dashboard.html`** in any browser. No Python or installation needed. Adjust a patient's values and the predicted score and high-risk probability update instantly.

## Limitations

Small single-study sample, a high-risk cut-off chosen statistically rather than clinically, and models that show association, not cause. Not for clinical use.
