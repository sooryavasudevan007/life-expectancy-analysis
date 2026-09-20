# What Is Associated with Life Expectancy? A Mixed-Model Analysis of WHO Country Data (2000-2015)

Applied Data Analysis assignment. Reproducible code for the accompanying report.

## Research question
Which health and socioeconomic factors are most strongly associated with life expectancy across countries, and does the association with schooling differ between developed and developing countries?

## Data
Kaggle "Life Expectancy (WHO)" (`kumarajarshi/life-expectancy-who`), compiled from WHO Global Health Observatory and UN data.
2,938 country-year records, 22 variables, 2000-2015. After cleaning: 2,928 observations, 183 countries.

The data file is not committed. To reproduce, download the CSV from Kaggle and save it as `data/Life Expectancy Data.csv`.

## Repository structure
```
.
├── analysis.py            # full pipeline: cleaning, EDA, models, validation, figures
├── requirements.txt
├── data/                  # place the Kaggle CSV here (git-ignored)
└── outputs/
    ├── figures/           # fig1-fig6 used in the report
    ├── mixed_model_coefficients.csv
    ├── sensitivity_comparison.csv
    └── cv_results.csv
```

## How to run
```
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python analysis.py
```
Runs in well under a minute. All outputs are written to `outputs/`.

## Methods
1. **Cleaning:** normalised column names; rows without life expectancy dropped; zeros in schooling and income composition treated as missing; within-country linear interpolation, then status-year median imputation; log transforms for GDP and HIV/AIDS; BMI, under-five deaths, population and measles excluded; adult mortality excluded from models (near-circular with the outcome).
2. **Inference:** linear mixed-effects model with country-specific random intercepts and year slopes; predictors standardised (coefficients are years of life expectancy per 1 SD); schooling × development-status interaction.
3. **Diagnostics and sensitivity:** residual and Q-Q plots; refit after removing the 10 largest residuals.
4. **Predictive validation:** Lasso and random forest evaluated with country-grouped 5-fold cross-validation, so no country appears in both training and test folds.

## Key results
| Predictor | Effect (years per 1 SD) | p |
|---|---|---|
| HIV/AIDS (log) | -3.28 | <0.001 |
| Schooling | +2.43 | <0.001 |
| Year | +0.76 | <0.001 |
| Developed status | +7.80 | <0.001 |
| Schooling x developed | -0.84 | 0.13 |
| GDP, alcohol, health expenditure, polio, diphtheria | not significant | >0.05 |

Country-grouped CV: Lasso R² = 0.81 (RMSE 4.0 years), random forest R² = 0.81 (RMSE 4.0 years).
Conclusions for the main predictors are unchanged when the 10 largest residuals are removed.

## Limitations
- Associations, not causal effects; country-level data cannot be read as individual-level effects.
- Heavy imputation for GDP and hepatitis B; a few implausible source values (e.g. 89 years for Spain 2007, Italy 2004).
- Only about 32 developed countries, which limits power for the interaction test.
- Residuals are heavy-tailed, so p-values for small effects are approximate.

## Data source
WHO Global Health Observatory and UN data via Kaggle: https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who
