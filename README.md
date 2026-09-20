# What Is Associated with Life Expectancy? A Mixed-Model Analysis of WHO Country Data (2000-2015)

Applied Data Analysis assignment. Reproducible code for the accompanying report.

## Research question
Which health and socioeconomic factors are most strongly associated with life expectancy across countries, and does the association with schooling differ between developed and developing countries?

## Data
Kaggle "Life Expectancy (WHO)" (`kumarajarshi/life-expectancy-who`), compiled from WHO Global Health Observatory and UN data.
2,938 country-year records, 22 variables, 2000-2015. After cleaning: 2,883 observations from 183 countries.

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
    ├── schooling_slopes_by_status.csv
    ├── standardization_constants.csv
    └── cv_results.csv
```

## How to run
```
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python analysis.py
```
All outputs are written to `outputs/`.

## Methods
1. **Cleaning:** normalised column names; 10 rows without a life expectancy value dropped; 45 rows with life expectancy above 85 years dropped as data errors (no national value exceeded about 84 years in 2000-2015; all are in high-income countries); zeros in schooling and income composition treated as missing; within-country linear interpolation, then status-year median imputation; log transforms for GDP and HIV/AIDS; BMI, under-five deaths, population and measles excluded; adult mortality and income composition of resources excluded from the models (adult mortality is nearly circular with the outcome; income composition correlates 0.92 with schooling).
2. **Inference:** linear mixed-effects model with country-specific random intercepts and year slopes; predictors standardised (coefficients are years of life expectancy per 1 SD); schooling x development-status interaction. Standardisation constants are in `outputs/standardization_constants.csv`.
3. **Diagnostics and sensitivity:** residual and Q-Q plots; refit on (a) the full data including the 45 removed rows and (b) the cleaned data excluding rows with |residual| > 5 years.
4. **Predictive validation:** Lasso and random forest evaluated with country-grouped 5-fold cross-validation, so no country appears in both training and test folds.

## Key results (cleaned data, years per 1 SD)
| Predictor | Effect | 95% CI | p |
|---|---|---|---|
| HIV/AIDS (log) | -3.10 | -3.51, -2.69 | <0.001 |
| Schooling | +2.24 | 1.76, 2.73 | <0.001 |
| Year | +0.78 | 0.60, 0.96 | <0.001 |
| Developed status (binary) | +8.83 | 6.57, 11.08 | <0.001 |
| Schooling x developed | -1.63 | -2.65, -0.61 | 0.002 |
| GDP, health expenditure, alcohol, polio | not significant | | >0.05 |
| Diphtheria | +0.09 | 0.00, 0.18 | 0.050 |

Schooling slope by status: developing +2.24 (CI 1.76, 2.73); developed +0.61 (CI -0.35, 1.57).
The interaction is also significant when rows with |residual| > 5 are excluded (-1.50, p < 0.001) but not in the uncleaned data that contain the 45 impossible values (-0.84, p = 0.13), so it is sensitive to data quality.
Country-grouped CV: Lasso R² = 0.82 (RMSE 3.9 years), random forest R² = 0.82 (RMSE 3.9 years).

## Limitations
- Associations, not causal effects; country-level data cannot be read as individual-level effects.
- Heavy imputation for GDP and hepatitis B.
- About 43 further suspicious entries remain in the main data (whole-number values roughly 5-8 years above the model fit); the trimmed sensitivity model addresses them.
- Only about 32 developed countries, which limits power for the interaction test.
- Residuals are heavy-tailed, so p-values for small effects are approximate.

## Data source
WHO Global Health Observatory and UN data via Kaggle: https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who