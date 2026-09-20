"""
Applied Data Analysis: WHO Life Expectancy (2000-2015)

Question: Which health and socioeconomic factors are most strongly associated
with life expectancy across countries, and does the association with schooling
differ between developed and developing countries?

Run:  python analysis.py
Data: put "Life Expectancy Data.csv" (Kaggle: kumarajarshi/life-expectancy-who)
      in ./data/
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("data/Life Expectancy Data.csv")
OUT = Path("outputs")
(OUT / "figures").mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

# ----------------------------------------------------------------------------
# 1. Load and clean
# ----------------------------------------------------------------------------
df = pd.read_csv(DATA)
# Column names have stray spaces / mixed case -> snake_case
df.columns = [re.sub(r"[^0-9a-z]+", "_", c.strip().lower()).strip("_") for c in df.columns]
print("Raw shape:", df.shape)
print("Missing values (%):\n", (df.isna().mean() * 100).round(1).sort_values(ascending=False).head(8))

df = df.dropna(subset=["life_expectancy"])

# Zeros in these columns are placeholders for "missing", not real values
for col in ["income_composition_of_resources", "schooling"]:
    df.loc[df[col] == 0, col] = np.nan

# BMI is not plausible as a country-level average (values >60 and <10 are common)
# and the two child-mortality columns are near-duplicates -> drop for clarity
df = df.drop(columns=["bmi", "under_five_deaths", "population", "measles"])

# Impute within country by linear interpolation over time, then by status-year median
df = df.sort_values(["country", "year"])
num_cols = df.select_dtypes("number").columns.difference(["year", "life_expectancy"])
df[num_cols] = df.groupby("country")[num_cols].transform(lambda s: s.interpolate(limit_direction="both"))
df[num_cols] = df.groupby(["status", "year"])[num_cols].transform(lambda s: s.fillna(s.median()))
df = df.dropna()

df["log_gdp"] = np.log(df["gdp"].clip(lower=1))
df["log_hiv"] = np.log1p(df["hiv_aids"])
df["developed"] = (df["status"] == "Developed").astype(int)
df_full = df.copy()  # kept for the sensitivity analysis

# Remove impossible values: no country's national life expectancy exceeded ~84 years in 2000-2015
implausible = df["life_expectancy"] > 85
print(f"Dropping {implausible.sum()} rows with life expectancy > 85 (data errors). By country:")
print(df.loc[implausible, "country"].value_counts().head(15).to_string())
df = df[~implausible].copy()
print("Clean shape:", df.shape, "| countries:", df["country"].nunique())

# ----------------------------------------------------------------------------
# 2. Exploratory analysis
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["life_expectancy"], kde=True, ax=ax[0])
ax[0].set(title="Distribution of life expectancy", xlabel="Life expectancy (years)")
sns.lineplot(data=df, x="year", y="life_expectancy", hue="status", errorbar="sd", ax=ax[1])
ax[1].set(title="Trend by development status (mean ± SD)", xlabel="Year", ylabel="Life expectancy (years)")
sns.move_legend(ax[1], "lower right", title="Status")
plt.tight_layout(); plt.savefig(OUT / "figures/fig1_distribution_trend.png", dpi=200); plt.close()

feats = ["adult_mortality", "log_hiv", "schooling", "income_composition_of_resources",
         "log_gdp", "alcohol", "polio", "diphtheria", "hepatitis_b", "total_expenditure",
         "thinness_1_19_years"]
plt.figure(figsize=(8, 6.5))
sns.heatmap(df[["life_expectancy"] + feats].corr(), annot=True, fmt=".2f", cmap="vlag", center=0, annot_kws={"size": 7})
plt.title("Correlation matrix")
plt.tight_layout(); plt.savefig(OUT / "figures/fig2_correlations.png", dpi=200); plt.close()

g = sns.lmplot(data=df, x="schooling", y="life_expectancy", hue="status", height=4.5, aspect=1.4,
               scatter_kws={"alpha": 0.3, "s": 14}, facet_kws={"legend_out": False})
g.set_axis_labels("Schooling (years)", "Life expectancy (years)")
g.ax.set_title("Schooling vs life expectancy (descriptive OLS lines by status)")
g.savefig(OUT / "figures/fig3_schooling_scatter.png", dpi=200); plt.close("all")

# ----------------------------------------------------------------------------
# 3a. Inference: mixed-effects model (random intercept + random year slope per country)
#     Repeated yearly observations per country are not independent, so plain
#     OLS would understate standard errors; random slopes let each country
#     have its own time trend.
# ----------------------------------------------------------------------------
# adult_mortality is excluded: it is nearly the inverse of life expectancy by construction.
# year is included to absorb the general upward trend over time.
model_feats = ["year", "log_hiv", "schooling", "log_gdp", "polio", "diphtheria",
               "total_expenditure", "alcohol"]
formula = "life_expectancy ~ " + " + ".join(model_feats) + " + developed + schooling:developed"


def standardize(d):
    d = d.copy()
    d[model_feats] = (d[model_feats] - d[model_feats].mean()) / d[model_feats].std()
    return d


def fit(d):
    return smf.mixedlm(formula, d, groups=d["country"], re_formula="~year").fit(reml=True)


z = standardize(df)
sd = pd.DataFrame({"mean": df[model_feats].mean(), "sd": df[model_feats].std()}).round(3)
print("Standardization constants (model coefficients are per 1 SD of these):")
print(sd.to_string())
sd.to_csv(OUT / "standardization_constants.csv")
mixed = fit(z)
print(mixed.summary())

coef = pd.DataFrame({"coef": mixed.fe_params, "se": mixed.bse_fe, "p": mixed.pvalues[mixed.fe_params.index]})
coef["ci_low"] = coef["coef"] - 1.96 * coef["se"]
coef["ci_high"] = coef["coef"] + 1.96 * coef["se"]
coef.round(3).to_csv(OUT / "mixed_model_coefficients.csv")

c = coef.drop("Intercept").sort_values("coef")
plt.figure(figsize=(7, 5))
plt.errorbar(c["coef"], range(len(c)), xerr=1.96 * c["se"], fmt="o", capsize=3)
nice = {"year": "Year", "log_hiv": "HIV/AIDS (log)", "schooling": "Schooling", "log_gdp": "GDP (log)",
        "polio": "Polio coverage", "diphtheria": "Diphtheria coverage", "total_expenditure": "Health expenditure",
        "alcohol": "Alcohol", "developed": "Developed (vs developing)", "schooling:developed": "Schooling × developed"}
plt.yticks(range(len(c)), [nice.get(k, k) for k in c.index]); plt.axvline(0, color="grey", ls="--")
plt.xlabel("Effect on life expectancy (years)\nper 1 SD for continuous predictors")
plt.title("Mixed-effects model: fixed effects (95% CI)")
plt.tight_layout(); plt.savefig(OUT / "figures/fig4_coefficients.png", dpi=200); plt.close()

# Residual diagnostics
resid, fitted = mixed.resid, mixed.fittedvalues
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(fitted, resid, alpha=0.3, s=10)
ax[0].axhline(0, color="grey", ls="--")
ax[0].set(xlabel="Fitted values", ylabel="Residuals", title="Residuals vs fitted")
sm.qqplot(resid, line="s", ax=ax[1], markersize=3)
ax[1].set_title("Q-Q plot of residuals")
plt.tight_layout(); plt.savefig(OUT / "figures/fig6_residual_diagnostics.png", dpi=200); plt.close()

# Large residuals (|residual| > 5 years)
big = resid.abs() > 5
print(f"Rows with |residual| > 5: {int(big.sum())}")
print(df.assign(resid=resid).loc[big, ["country", "year", "life_expectancy", "resid"]]
        .sort_values("resid", ascending=False).to_string())

# Sensitivity 1: keep the implausible rows (full data)
m_full = fit(standardize(df_full))
# Sensitivity 2: additionally drop rows with |residual| > 5 from the cleaned model
m_trim = fit(z.loc[~big])

sens = pd.DataFrame({
    "main_coef": mixed.fe_params, "main_p": mixed.pvalues[mixed.fe_params.index],
    "full_data_coef": m_full.fe_params, "full_data_p": m_full.pvalues[m_full.fe_params.index],
    "trim_resid5_coef": m_trim.fe_params, "trim_resid5_p": m_trim.pvalues[m_trim.fe_params.index],
})
print(sens.round(3).to_string())
sens.round(3).to_csv(OUT / "sensitivity_comparison.csv")


# Schooling slope by development status (answers research question 2 directly)
def slopes(m, label):
    b = m.fe_params
    k = len(b)
    V = pd.DataFrame(np.asarray(m.cov_params())[:k, :k], index=b.index, columns=b.index)
    a, i = "schooling", "schooling:developed"
    rows = []
    for grp, est, var in [("developing", b[a], V.loc[a, a]),
                          ("developed", b[a] + b[i], V.loc[a, a] + V.loc[i, i] + 2 * V.loc[a, i])]:
        se = np.sqrt(var)
        rows.append({"model": label, "group": grp, "slope": est, "ci_low": est - 1.96 * se, "ci_high": est + 1.96 * se})
    return rows


sl = pd.DataFrame(slopes(mixed, "main") + slopes(m_full, "full_data") + slopes(m_trim, "trim_resid5")).round(3)
print(sl.to_string(index=False))
sl.to_csv(OUT / "schooling_slopes_by_status.csv", index=False)

# ----------------------------------------------------------------------------
# 3b. Validity check: predictive performance with country-grouped CV
#     (random splits would leak the same country into train and test)
# ----------------------------------------------------------------------------
X, y, groups = df[model_feats + ["developed"]], df["life_expectancy"], df["country"]
models = {
    "Lasso": make_pipeline(StandardScaler(), LassoCV(cv=5)),
    "Random forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1),
}
rows = []
for name, m in models.items():
    r2s, rmses = [], []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m.fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict(X.iloc[te])
        r2s.append(r2_score(y.iloc[te], pred))
        rmses.append(mean_squared_error(y.iloc[te], pred) ** 0.5)
    rows.append({"model": name, "R2": np.mean(r2s), "RMSE": np.mean(rmses)})
cv = pd.DataFrame(rows).round(3)
print(cv)
cv.to_csv(OUT / "cv_results.csv", index=False)

rf = models["Random forest"].fit(X, y)
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(6.5, 4.5)); imp.plot.barh()
plt.title("Random forest feature importance"); plt.xlabel("Impurity-based importance")
plt.tight_layout(); plt.savefig(OUT / "figures/fig5_rf_importance.png", dpi=200); plt.close()

print("Done. Results in ./outputs")